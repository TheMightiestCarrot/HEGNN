import os
import time
import json
import csv

import torch

METRIC_TARGET_FLOOR = 1e-3


def _normalize_epoch_metrics(epoch_output):
    """Support both legacy triple outputs and extended metric tuples."""
    if isinstance(epoch_output, dict):
        required = (
            epoch_output.get('loss'),
            epoch_output.get('pos_err'),
            epoch_output.get('pos_mae'),
            epoch_output.get('vel_pct_err', 0.0),
            epoch_output.get('vel_mae', 0.0),
            epoch_output.get('vel_rmse', 0.0),
            epoch_output.get('vel_loss', 0.0),
        )
        if required[0] is not None and required[1] is not None and required[2] is not None:
            return required
    elif isinstance(epoch_output, (list, tuple)):
        if len(epoch_output) == 7:
            return epoch_output
        if len(epoch_output) == 6:
            loss, pos_err, pos_mae, vel_mae, vel_rmse, vel_loss = epoch_output
            return loss, pos_err, pos_mae, 0.0, vel_mae, vel_rmse, vel_loss
        if len(epoch_output) == 3:
            loss, pos_err, pos_mae = epoch_output
            return loss, pos_err, pos_mae, 0.0, 0.0, 0.0, 0.0
    raise ValueError(
        "train_single_epoch returned unsupported metrics format; expected tuple of length 3, 6, or 7, "
        f"got type {type(epoch_output)!r}."
    )

def get_edges_in_mini_batch(batch_size, num_nodes_all, edge_index):
    correction_for_batch = num_nodes_all * torch.arange(batch_size, device=edge_index.device)  # [batch_size]
    correction_for_batch = correction_for_batch.repeat_interleave(edge_index.size(1) // batch_size, dim=0).unsqueeze(0)  # [1, edge_cnt]
    correction_for_batch = correction_for_batch.repeat_interleave(2, dim=0)
    edge_index_in_mini_batch = edge_index + correction_for_batch
    return edge_index_in_mini_batch


def kernel(x, y, sigma):
    dist = torch.cdist(x, y, p=2)
    k = torch.exp(- dist / (2 * sigma * sigma))
    return k


def train_single_epoch(model, loader, optimizer, loss, sigma, weight, epoch_index, backprop, tag, sample, device='cpu',
                      scheduler=None, scheduler_mode='none', integrator=None, integrator_dt=1.0, integrator_kwargs=None,
                      loss_vel_weight=0.0):
    if backprop:
        model.train()
    else:
        model.eval()

    if integrator_kwargs is None:
        integrator_kwargs = {}

    result = {
        'loss': 0.,
        'counter': 0.,
        'pos_err': 0.,
        'pos_mae': 0.,
        'vel_pct_err': 0.,
        'vel_mae': 0.,
        'vel_rmse': 0.,
        'vel_loss': 0.,
        'vel_counter': 0.,
    }
    for batch_index, data in enumerate(loader):
        # All to device
        data = data.to(device)
        data = data.detach()  # All detach

        # Parse data
        batch_size = data['ptr'].size(0) - 1
        edge_index, edge_attr = data['edge_index'], data['edge_attr']
        loc_0, vel_0, loc_t = data['loc_0'], data['vel_0'], data['loc_t']
        vel_t = getattr(data, 'vel_t', None)
        node_feat, node_attr = data['node_feat'], data['node_attr']

        row, col = edge_index
        edge_length_0 = torch.sqrt(torch.sum((loc_0[row] - loc_0[col])**2, dim=1)).unsqueeze(1)
        edge_attr = torch.cat([edge_attr, edge_length_0], dim=1)
        
        # detach from compute graph
        loc_0, vel_0, node_attr, node_feat = loc_0.detach(), vel_0.detach(), node_attr.detach(), node_feat.detach()
        edge_attr, edge_index = edge_attr.detach(), edge_index.detach()
        if vel_t is not None:
            vel_t = vel_t.detach()

        optimizer.zero_grad()

        # start_time = time.time()
        vel_predict = None
        if model.__class__.__name__ == 'TFNModel':
            loc_predict = model(loc=loc_0, h=node_feat, vel=vel_0, edge_index=edge_index, data_batch=data['batch'])
        elif model.__class__.__name__ == 'VNEGNN':
            loc_predict, virtual_node_loc = model(node_loc=loc_0, node_attr=None, node_feat=node_feat, edge_index=edge_index, 
                                                  virtual_node_loc=data['virtual_fibonacci'].detach(), data_batch=data['batch'], edge_attr=edge_attr)
        elif model.__class__.__name__ == 'EGNN':
            out = model(x=loc_0, h=node_feat, edge_index=edge_index, edge_fea=edge_attr, v=vel_0)
            loc_predict = out[0]
        elif model.__class__.__name__ == 'HEGNN':
            integrator_name = str(integrator).lower() if integrator is not None else None
            use_integrator = (
                integrator_name not in (None, 'none')
                and hasattr(model, 'symplectic_euler_step')
                and callable(getattr(model, 'symplectic_euler_step'))
            )
            if use_integrator:
                masses = getattr(data, 'masses', None)
                integrator_kwargs_local = dict(integrator_kwargs)
                if integrator_name == 'symplectic_euler':
                    pos_next, vel_next, accel = model.symplectic_euler_step(
                        node_feat, loc_0, vel_0, edge_index, edge_attr, masses=masses, dt=integrator_dt, **integrator_kwargs_local
                    )
                elif integrator_name in ('velocity_verlet', 'verlet'):
                    pos_next, vel_next, accel = model.velocity_verlet_step(
                        node_feat, loc_0, vel_0, edge_index, edge_attr, masses=masses, dt=integrator_dt, **integrator_kwargs_local
                    )
                else:
                    raise ValueError(f"Unknown integrator '{integrator}'.")
                loc_predict = pos_next
                vel_predict = vel_next
            else:
                loc_predict = model(node_feat, loc_0, vel_0, edge_index, edge_attr)
        elif model.__class__.__name__ == 'GNN':
            nodes = torch.cat([loc_0, vel_0], dim=1)
            loc_predict = model(h=nodes, edge_index=edge_index, edge_fea=edge_attr)
        elif model.__class__.__name__ == 'Linear_dynamics':
            loc_predict = model(x=loc_0, v=vel_0)
        elif model.__class__.__name__ == 'RF_vel':
            vel_norm = torch.sqrt(torch.sum(vel_0 ** 2, dim=1).unsqueeze(1)).detach()
            loc_predict = model(vel_norm=vel_norm, x=loc_0, edges=edge_index, vel=vel_0, edge_attr=edge_attr)
        elif model.__class__.__name__ == 'OurDynamics':  # TFN
            loc_predict = model(loc_0, vel_0, node_attr, edge_index)
        elif model.__class__.__name__ == 'GVPNet':
            h_V = (node_feat, torch.stack([loc_0, vel_0], dim=1))  # node_s, node_v
            row, col = edge_index
            h_E = (edge_attr, (loc_0[row] - loc_0[col]).unsqueeze(1))  # edge_s, edge_v
            out = model(h_V=h_V, edge_index=edge_index, h_E=h_E, batch=data['batch'])
            loc_predict = out[1][:, 0, :]  # get coord
        elif model.__class__.__name__ == 'SchNet':
            loc_predict = model(z=node_feat, pos=loc_0, batch=data['batch'], edge_index=edge_index)
        elif model.__class__.__name__ in ['ClofNet', 'ClofNet_vel', 'clof_vel_gbf']:
            nodes = torch.sqrt(torch.sum(vel_0 ** 2, dim=1)).unsqueeze(1).detach()
            rows, cols = edge_index
            loc_dist = torch.sum((loc_0[rows] - loc_0[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            n_node = torch.tensor([loc_0.size(0) // batch_size])
            loc_predict = model(nodes, loc_0.detach(), edge_index, vel_0, edge_attr, n_nodes=n_node)
        elif model.__class__.__name__ == 'MACEModel':
            loc_predict = model(loc=loc_0, h=node_feat, vel=vel_0, edge_index=edge_index, data_batch=data['batch'])
        elif model.__class__.__name__ == 'SEGNN':
            x = torch.cat([node_feat, loc_0, vel_0], dim=1)
            loc_predict = model(x=x, pos=loc_0, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr, batch=data['batch'])
        elif model.__class__.__name__ == 'SEGNNFull':
            scalar_attr = node_attr
            outputs = model(
                pos=loc_0,
                vel=vel_0,
                node_attr=scalar_attr,
                edge_index=edge_index,
                batch=data['batch'],
            )
            delta_pos = outputs[:, :3]
            loc_predict = loc_0 + delta_pos
            if loss_vel_weight > 0.0 and outputs.size(1) >= 6:
                vel_predict = outputs[:, 3:6]
        else:
            print(model.__class__.__name__)
            raise Exception('Wrong model')
        
        loss_pos = loss(loc_predict, loc_t)
        total_loss = loss_pos
        with torch.no_grad():
            loc_t_detached = loc_t.detach()
            err = (loc_predict - loc_t_detached).detach()
            err_l2 = torch.linalg.norm(err, dim=1)
            target_norm = torch.linalg.norm(loc_t_detached, dim=1)
            mae = err.abs().mean().item()
            rms = torch.sqrt(torch.mean(loc_t_detached.pow(2)))
            floor = torch.maximum(
                rms,
                torch.tensor(
                    METRIC_TARGET_FLOOR,
                    device=target_norm.device,
                    dtype=target_norm.dtype
                ),
            )
            denom = torch.clamp(target_norm, min=floor)
            pct_err = (err_l2 / denom).mean().item() * 100.0

        loss_vel_value = None
        if vel_predict is not None and vel_t is not None:
            loss_vel_value = loss(vel_predict, vel_t)
            if loss_vel_weight > 0.0:
                total_loss = total_loss + loss_vel_weight * loss_vel_value
            with torch.no_grad():
                vel_t_detached = vel_t.detach()
                vel_err = (vel_predict - vel_t_detached).detach()
                vel_mae = vel_err.abs().mean().item()
                vel_rmse = torch.sqrt(torch.mean(vel_err.pow(2))).item()
                vel_err_l2 = torch.linalg.norm(vel_err, dim=1)
                vel_target_norm = torch.linalg.norm(vel_t_detached, dim=1)
                vel_rms = torch.sqrt(torch.mean(vel_t_detached.pow(2)))
                vel_floor = torch.maximum(
                    vel_rms,
                    torch.tensor(
                        METRIC_TARGET_FLOOR,
                        device=vel_target_norm.device,
                        dtype=vel_target_norm.dtype,
                    ),
                )
                vel_denom = torch.clamp(vel_target_norm, min=vel_floor)
                vel_pct_err = (vel_err_l2 / vel_denom).mean().item() * 100.0
            result['vel_mae'] += vel_mae * batch_size
            result['vel_rmse'] += vel_rmse * batch_size
            result['vel_pct_err'] += vel_pct_err * batch_size
            result['vel_counter'] += batch_size
            result['vel_loss'] += loss_vel_value.item() * batch_size

        # record the loss
        result['loss'] += total_loss.item() * batch_size
        result['counter'] += batch_size
        result['pos_err'] += pct_err * batch_size
        result['pos_mae'] += mae * batch_size
        
        if backprop:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
            if scheduler is not None and scheduler_mode == 'batch':
                try:
                    scheduler.step()
                except Exception as e:
                    print(f'[warn] batch scheduler.step() failed: {e}')

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""

    if result['counter'] == 0:
        print(f'{prefix + tag} epoch: {epoch_index}, no batches processed (batch_size too large?).')
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    avg_loss = result["loss"] / result["counter"]
    avg_pos_err = result["pos_err"] / result["counter"]
    avg_pos_mae = result["pos_mae"] / result["counter"]

    if result['vel_counter'] > 0:
        avg_vel_pct_err = result['vel_pct_err'] / result['vel_counter']
        avg_vel_mae = result['vel_mae'] / result['vel_counter']
        avg_vel_rmse = result['vel_rmse'] / result['vel_counter']
        avg_vel_loss = result['vel_loss'] / result['vel_counter']
        print(f'{prefix + tag} epoch: {epoch_index}, avg loss: {avg_loss :.5f}, avg pos %err: {avg_pos_err :.4f}, avg pos MAE: {avg_pos_mae :.6f}, avg vel %err: {avg_vel_pct_err :.4f}, avg vel MAE: {avg_vel_mae :.6f}, avg vel RMSE: {avg_vel_rmse :.6f}')
    else:
        avg_vel_pct_err = 0.0
        avg_vel_mae = 0.0
        avg_vel_rmse = 0.0
        avg_vel_loss = 0.0
        print(f'{prefix + tag} epoch: {epoch_index}, avg loss: {avg_loss :.5f}, avg pos %err: {avg_pos_err :.4f}, avg pos MAE: {avg_pos_mae :.6f}')

    return avg_loss, avg_pos_err, avg_pos_mae, avg_vel_pct_err, avg_vel_mae, avg_vel_rmse, avg_vel_loss

def train(model, loader_train, loader_valid, loader_test, optimizer, loss, sigma, weight, log_directory, log_name,
          early_stop=float('inf'), device='cpu', test_interval=5, sample=3, config=None, wandb_run=None,
          scheduler=None, scheduler_mode='none', integrator=None, integrator_dt=1.0, integrator_kwargs=None,
          loss_vel_weight=0.0):
    log_dict = {
        'epochs': [],
        'loss': [],
        'loss_train': [],
        'pos_err': [],
        'pos_err_train': [],
        'pos_mae': [],
        'pos_mae_train': [],
        'vel_pct_err': [],
        'vel_pct_err_train': [],
        'vel_mae': [],
        'vel_mae_train': [],
        'vel_rmse': [],
        'vel_rmse_train': [],
        'vel_loss': [],
        'vel_loss_train': [],
    }
    best_log_dict = {
        'epoch_index': 0,
        'loss_valid': 1e8,
        'loss_test': 1e8,
        'loss_train': 1e8,
        'pos_err_valid': 1e8,
        'pos_err_test': 1e8,
        'pos_err_train': 1e8,
        'pos_mae_valid': 1e8,
        'pos_mae_test': 1e8,
        'pos_mae_train': 1e8,
        'vel_pct_err_valid': 1e8,
        'vel_pct_err_test': 1e8,
        'vel_pct_err_train': 1e8,
        'vel_mae_valid': 1e8,
        'vel_mae_test': 1e8,
        'vel_mae_train': 1e8,
        'vel_rmse_valid': 1e8,
        'vel_rmse_test': 1e8,
        'vel_rmse_train': 1e8,
        'vel_loss_valid': 1e8,
        'vel_loss_test': 1e8,
        'vel_loss_train': 1e8,
    }

    start =time.perf_counter()
    max_epochs = 2500
    try:
        if config is not None and hasattr(config, 'epochs'):
            max_epochs = int(getattr(config, 'epochs'))
    except Exception:
        max_epochs = 2500
    for epoch_index in range(1, max_epochs+1):
        train_epoch_output = train_single_epoch(
            model, loader_train, optimizer, loss, sigma, weight, epoch_index,
            backprop=True, tag='train', device=device, sample=sample,
            scheduler=scheduler, scheduler_mode=scheduler_mode,
            integrator=integrator, integrator_dt=integrator_dt, integrator_kwargs=integrator_kwargs,
            loss_vel_weight=loss_vel_weight,
        )
        (
            loss_train,
            pos_err_train,
            pos_mae_train,
            vel_pct_err_train,
            vel_mae_train,
            vel_rmse_train,
            vel_loss_train,
        ) = _normalize_epoch_metrics(train_epoch_output)
        log_dict['loss_train'].append(loss_train)
        log_dict['pos_err_train'].append(pos_err_train)
        log_dict['pos_mae_train'].append(pos_mae_train)
        log_dict['vel_pct_err_train'].append(vel_pct_err_train)
        log_dict['vel_mae_train'].append(vel_mae_train)
        log_dict['vel_rmse_train'].append(vel_rmse_train)
        log_dict['vel_loss_train'].append(vel_loss_train)
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch_index,
                'train/step': epoch_index,
                'train/loss': loss_train,
                'train/pos_perc_error': pos_err_train,
                'train/pos_mae': pos_mae_train,
                'train/vel_perc_error': vel_pct_err_train,
                'train/vel_mae': vel_mae_train,
                'train/vel_rmse': vel_rmse_train,
                'train/vel_loss': vel_loss_train,
                'lr': optimizer.param_groups[0]['lr']
            }, step=epoch_index)

        if epoch_index % test_interval == 0:
            valid_epoch_output = train_single_epoch(
                model, loader_valid, optimizer, loss, sigma, weight, epoch_index,
                backprop=False, tag='valid', device=device, sample=sample,
                integrator=integrator, integrator_dt=integrator_dt, integrator_kwargs=integrator_kwargs,
                loss_vel_weight=loss_vel_weight,
            )
            (
                loss_valid,
                pos_err_valid,
                pos_mae_valid,
                vel_pct_err_valid,
                vel_mae_valid,
                vel_rmse_valid,
                vel_loss_valid,
            ) = _normalize_epoch_metrics(valid_epoch_output)
            test_epoch_output = train_single_epoch(
                model, loader_test, optimizer, loss, sigma, weight, epoch_index,
                backprop=False, tag='test', device=device, sample=sample,
                integrator=integrator, integrator_dt=integrator_dt, integrator_kwargs=integrator_kwargs,
                loss_vel_weight=loss_vel_weight,
            )
            (
                loss_test,
                pos_err_test,
                pos_mae_test,
                vel_pct_err_test,
                vel_mae_test,
                vel_rmse_test,
                vel_loss_test,
            ) = _normalize_epoch_metrics(test_epoch_output)
            
            log_dict['epochs'].append(epoch_index)
            log_dict['loss'].append(loss_test)
            log_dict['pos_err'].append(pos_err_test)
            log_dict['pos_mae'].append(pos_mae_test)
            log_dict['vel_pct_err'].append(vel_pct_err_test)
            log_dict['vel_mae'].append(vel_mae_test)
            log_dict['vel_rmse'].append(vel_rmse_test)
            log_dict['vel_loss'].append(vel_loss_test)
            if wandb_run is not None:
                wandb_run.log({
                    'epoch': epoch_index,
                    'valid/step': epoch_index,
                    'valid/loss': loss_valid,
                    'valid/pos_perc_error': pos_err_valid,
                    'valid/pos_mae': pos_mae_valid,
                    'valid/vel_perc_error': vel_pct_err_valid,
                    'valid/vel_mae': vel_mae_valid,
                    'valid/vel_rmse': vel_rmse_valid,
                    'valid/vel_loss': vel_loss_valid,
                    'test/step': epoch_index,
                    'test/loss': loss_test,
                    'test/pos_perc_error': pos_err_test,
                    'test/pos_mae': pos_mae_test,
                    'test/vel_perc_error': vel_pct_err_test,
                    'test/vel_mae': vel_mae_test,
                    'test/vel_rmse': vel_rmse_test,
                    'test/vel_loss': vel_loss_test,
                    'lr': optimizer.param_groups[0]['lr']
                }, step=epoch_index)

            # Plateau scheduler uses validation loss; step here when we have it
            if scheduler is not None and scheduler_mode == 'plateau':
                try:
                    scheduler.step(loss_valid)
                except Exception as e:
                    print(f'[warn] plateau scheduler.step(loss) failed: {e}')
            
            if loss_valid < best_log_dict['loss_valid']:
                best_log_dict = {
                    'epoch_index': epoch_index,
                    'loss_valid': loss_valid,
                    'loss_test': loss_test,
                    'loss_train': loss_train,
                    'pos_err_valid': pos_err_valid,
                    'pos_err_test': pos_err_test,
                    'pos_err_train': pos_err_train,
                    'pos_mae_valid': pos_mae_valid,
                    'pos_mae_test': pos_mae_test,
                    'pos_mae_train': pos_mae_train,
                    'vel_pct_err_valid': vel_pct_err_valid,
                    'vel_pct_err_test': vel_pct_err_test,
                    'vel_pct_err_train': vel_pct_err_train,
                    'vel_mae_valid': vel_mae_valid,
                    'vel_mae_test': vel_mae_test,
                    'vel_mae_train': vel_mae_train,
                    'vel_rmse_valid': vel_rmse_valid,
                    'vel_rmse_test': vel_rmse_test,
                    'vel_rmse_train': vel_rmse_train,
                    'vel_loss_valid': vel_loss_valid,
                    'vel_loss_test': vel_loss_test,
                    'vel_loss_train': vel_loss_train,
                }
                name = None
                if config.dataset_name in ['5_0_0', '20_0_0', '50_0_0', '100_0_0']:
                    name = 'nbody'

                os.makedirs(f'./state_dict/{name}', exist_ok=True)
                torch.save(model.state_dict(), f'./state_dict/{name}/{model.__class__.__name__}_best_model.pth')
            print(f'*** Best Valid Loss: {best_log_dict["loss_valid"] :.5f} | Best Test Loss: {best_log_dict["loss_test"] :.5f} | Best Epoch Index: {best_log_dict["epoch_index"]}')

            if epoch_index - best_log_dict['epoch_index'] >= early_stop:
                best_log_dict['early_stop'] = epoch_index
                print(f'Early stopped! Epoch: {epoch_index}')
                break

        # Step epoch-based schedulers every epoch
        if scheduler is not None and scheduler_mode == 'epoch':
            try:
                scheduler.step()
            except Exception as e:
                print(f'[warn] epoch scheduler.step() failed: {e}')

        end = time.perf_counter() 
        time_cost = end - start
        best_log_dict['time_cost'] = time_cost
        
        json_object = json.dumps([best_log_dict, log_dict], indent=4)
        os.makedirs(log_directory, exist_ok=True)
        with open(f'{log_directory}/{log_name}', "w") as outfile:
            outfile.write(json_object)
    

    return best_log_dict, log_dict
