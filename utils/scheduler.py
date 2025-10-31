import os
from typing import Optional, Tuple

import torch
from torch import optim


def build_scheduler(opt: optim.Optimizer, args, train_loader_len: int) -> Tuple[Optional[optim.lr_scheduler._LRScheduler], str]:
    """
    Create a learning-rate scheduler based on CLI args.

    Returns (scheduler, mode) where mode in { 'none', 'batch', 'epoch', 'plateau' } indicates when to step.
    """
    sched_type = getattr(args, 'scheduler', 'none')
    if sched_type == 'none':
        return None, 'none'
    if sched_type == 'plateau':
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=args.lr_factor, patience=args.lr_patience,
            cooldown=args.lr_cooldown, min_lr=args.lr_min, verbose=True
        )
        return sch, 'plateau'
    if sched_type == 'cosine':
        sch = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.lr_t_max), eta_min=args.lr_min
        )
        return sch, 'epoch'
    if sched_type == 'cosine_restart':
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=max(1, args.lr_T_0), T_mult=max(1, args.lr_T_mult), eta_min=args.lr_min
        )
        return sch, 'epoch'
    if sched_type == 'step':
        sch = optim.lr_scheduler.StepLR(
            opt, step_size=max(1, args.lr_step_size), gamma=args.lr_gamma
        )
        return sch, 'epoch'
    if sched_type == 'exponential':
        sch = optim.lr_scheduler.ExponentialLR(
            opt, gamma=args.lr_gamma
        )
        return sch, 'epoch'
    if sched_type == 'warmup_cosine':
        total_warmup = int(getattr(args, 'warmup_steps', 0))
        scheds = []
        milestones = []
        if total_warmup > 0:
            warmup = optim.lr_scheduler.LinearLR(
                opt, start_factor=max(1e-6, args.warmup_start_factor), total_iters=total_warmup
            )
            scheds.append(warmup)
            milestones.append(total_warmup)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, args.lr_t_max), eta_min=args.lr_min
        )
        scheds.append(cosine)
        if len(milestones) == 0:
            return cosine, 'epoch'
        else:
            sch = optim.lr_scheduler.SequentialLR(opt, schedulers=scheds, milestones=milestones)
            return sch, 'epoch'
    if sched_type == 'onecycle':
        steps_per_epoch = max(1, train_loader_len)
        total_steps = max(1, steps_per_epoch * max(1, args.epochs))
        sch = optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=args.learning_rate,
            total_steps=total_steps,
            pct_start=args.onecycle_pct_start,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor,
            anneal_strategy='cos'
        )
        return sch, 'batch'
    raise ValueError(f'Unknown scheduler: {sched_type}')


def lr_range_test(
    model,
    optimizer: optim.Optimizer,
    loss_fn,
    loader_train,
    device: str,
    start_lr: float,
    end_lr: float,
    num_steps: int,
    log_directory: str,
    log_time_suffix: str,
) -> float:
    """Run a simple LR range test and save CSV. Returns suggested base LR.

    Heuristic: 10% of the LR at minimum smoothed loss. Supports HEGNN/EGNN paths.
    """
    model.train()
    print('Starting LR range test...')
    lr_mult = (end_lr / start_lr) ** (1 / max(1, num_steps - 1))
    for pg in optimizer.param_groups:
        pg['lr'] = start_lr
    avg_loss, best_loss = 0.0, float('inf')
    losses = []
    lrs = []
    step_count = 0
    for batch in loader_train:
        if step_count >= num_steps:
            break
        batch = batch.to(device)
        # detach() is supported by pyg Data; fallback if missing
        detach = getattr(batch, 'detach', None)
        if callable(detach):
            batch = batch.detach()
        optimizer.zero_grad()
        edge_index, edge_attr = batch['edge_index'], batch['edge_attr']
        loc_0, vel_0, loc_t = batch['loc_0'], batch['vel_0'], batch['loc_t']
        node_feat = batch['node_feat']
        row, col = edge_index
        edge_length_0 = torch.sqrt(torch.sum((loc_0[row] - loc_0[col])**2, dim=1)).unsqueeze(1)
        edge_attr = torch.cat([edge_attr, edge_length_0], dim=1)
        cls = model.__class__.__name__
        if cls == 'HEGNN':
            loc_predict = model(node_feat, loc_0, vel_0, edge_index, edge_attr)
        elif cls == 'EGNN':
            out = model(x=loc_0, h=node_feat, edge_index=edge_index, edge_fea=edge_attr, v=vel_0)
            loc_predict = out[0]
        else:
            raise RuntimeError('LR finder currently supports HEGNN/EGNN models')

        loss = loss_fn(loc_predict, loc_t)
        loss.backward()
        optimizer.step()

        beta = 0.98
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed = avg_loss / (1 - beta ** (step_count + 1))
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(smoothed)
        if smoothed < best_loss:
            best_loss = smoothed
        if step_count > 10 and smoothed > 4 * best_loss:
            print('Stopping LR finder early due to divergence')
            break
        for pg in optimizer.param_groups:
            pg['lr'] *= lr_mult
        step_count += 1

    suggested = None
    if len(losses) > 0:
        min_idx = int(torch.tensor(losses).argmin().item())
        suggested = lrs[min_idx] / 10.0
        print(f'LR finder complete. Suggested base LR ≈ {suggested:.3e} (10% of LR@min-loss).')
        os.makedirs(log_directory, exist_ok=True)
        csv_path = os.path.join(log_directory, f'lr_finder_{log_time_suffix}.csv')
        with open(csv_path, 'w') as f:
            f.write('step,lr,loss\n')
            for i, (lr, ls) in enumerate(zip(lrs, losses)):
                f.write(f'{i},{lr:.8e},{ls:.8f}\n')
        print(f'LR finder curve written to {csv_path}')
    return suggested if suggested is not None else start_lr

