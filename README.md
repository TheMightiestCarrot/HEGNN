# Are High-Degree Representations Really Unnecessary in Equivarinat Graph Neural Networks? (NeurIPS 2024)

Jiacheng Cen, Anyi Li, Ning Lin, Yuxiang Ren, Zihe Wang, Wenbing Huang

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/GLAD-RUC/HEGNN/blob/main/LICENSE)

[**[OpenReview]**](https://openreview.net/forum?id=M0ncNVuGYN) [**[paper]**](https://openreview.net/pdf?id=M0ncNVuGYN) [**[poster]**](https://neurips.cc/virtual/2024/poster/95552) [**[arXiv]**](https://arxiv.org/abs/2410.11443)

![](./assets/HEGNN-teaser.jpg)
Common symmetric graphs. Equivariant GNNs on symmetric graphs will degenerate to a zero function if the degree of their representations is fixed as 1.

## Key Requirements

```
dgl==1.1.3+cu118
e3nn==0.5.1
matplotlib>=3.8,<3.9
numpy==1.26.4
scipy==1.8.1
sympy==1.12
torch==2.1.0+cu118
torch_geometric==2.6.1
torch_scatter==2.1.2+pt21cu118
torch_sparse==0.6.18+pt21cu118
```

A more detailed Python environments is depicted in `requirements.txt`. 

## Expressivity on Symmetric Graphs

The `/expressivity` directory contains the notebooks with `rot-3D-test.ipynb` and `reg-poly-test.ipynb`, which respectively the expressivity evaulation on $k$-fold structure and five regular polyhedra. 

## Physical Dynamics Simulation

### $N$-body System Dataset

#### Data Preparation

To generate datasets containing multiple isolated particles, please use the following command.

```bash
python -u ./datasets/nbody/datagen/generate_dataset.py --num-train 5000 --seed 43 --n_isolated 5 --n_stick 0 --n_hinge 0 --n_workers 50
```

#### Run Experiments

```bash
python ./main_nbody.py --model HEGNN --ell 3 --data_directory <your_dir> --dataset_name "5_0_0"
```

##### Optional: Velocity Targets & Auto-Tuned Hyperparameters

- Enable velocity supervision by passing `--loss_vel_weight <λ>`; when `--integrator` predicts positions via symplectic Euler, this also supervises the implied velocity.
- Use `--use_velocity_features True` when training `HEGNN_acceleration` to feed relative velocity invariants into the force head.
- The helper `scripts/auto_tune_hparams.py` estimates a good coarse step `Δt` and loss weight from the dataset:

  ```bash
  python3 scripts/auto_tune_hparams.py \
    --data_directory datasets/nbody/data_5body_full \
    --dataset_name 5_0_0 \
    --partitions train,valid \
    --max_samples 1500 \
    --output logs/nbody/tuning.json --pretty
  ```

- To run the estimator automatically before training, add `--auto_tune_hparams`. By default this:
  - samples up to `--tune_max_samples` (default 1000) per partition listed in `--tune_partitions`,
  - applies the suggested `--coarse_dt` and `--loss_vel_weight`,
  - optionally writes the JSON report if `--tune_report_path` is provided.

Example with auto tuning:

```bash
python -u ./main_nbody.py \
  --model HEGNN_acceleration \
  --ell 1 \
  --dim_hidden 64 \
  --num_layer 4 \
  --data_directory datasets/nbody/data_5body_full \
  --dataset_name 5_0_0 \
  --device cuda \
  --batch_size 256 \
  --epochs 1000 \
  --integrator symplectic_euler \
  --auto_tune_hparams \
  --tune_partitions train,valid \
  --tune_report_path logs/nbody/tuning.json \
  --use_velocity_features True
```

#### Visualise Trajectories

Once the dataset is generated (or you use the bundled samples in `datasets/nbody/data_small`), you can inspect any simulated rollout and export an MP4 animation:

```bash
pip install -r requirements.txt  # ensures matplotlib is available

# Save simulation 0 from the validation split to an MP4 with short trails and body labels
python3 visualize_nbody.py --partition valid --simulation-index 0 \
  --trail-length 5 --annotate --fps 15

# Preview interactively without writing an MP4
python3 visualize_nbody.py --show
```

Use `--dataset-root` / `--dataset-name` to point at other generated directories, `--start-frame` / `--end-frame` to trim the playback window, and `--elev` / `--azim` to tune the 3D camera. MP4 export requires `ffmpeg` to be installed and visible on your `PATH`; if it is missing the script will warn and skip the save step.

#### Learning Rate, Schedulers, and LR Finder

The N-body runner now supports configurable learning-rate schedulers and an LR range test.

- Default optimizer: Adam with `--learning_rate` (default `5e-4`) and `--weight_decay` (default `1e-12`).
- Default scheduler: none (fixed LR). Select via `--scheduler`.

Available schedulers (via `--scheduler`):

- `none` (default): no scheduling.
- `plateau`: `ReduceLROnPlateau` stepped on validation loss.
  - `--lr_patience` (default `25`), `--lr_factor` (default `0.5`), `--lr_cooldown` (default `0`), `--lr_min` (default `1e-6`).
- `cosine`: `CosineAnnealingLR`.
  - `--lr_t_max` (default `200`), `--lr_min` (default `1e-6`).
- `cosine_restart`: `CosineAnnealingWarmRestarts`.
  - `--lr_T_0` (default `200`), `--lr_T_mult` (default `2`), `--lr_min` (default `1e-6`).
- `step`: `StepLR`.
  - `--lr_step_size` (default `50`), `--lr_gamma` (default `0.9`).
- `exponential`: `ExponentialLR`.
  - `--lr_gamma` (default `0.9`).
- `warmup_cosine`: linear warmup (`LinearLR`) then cosine.
  - `--warmup_steps` (default `0`), `--warmup_start_factor` (default `0.1`), plus cosine args above.
- `onecycle`: `OneCycleLR` (per-batch stepping; uses `--epochs` and loader length).
  - `--onecycle_pct_start` (default `0.3`), `--onecycle_div_factor` (default `25.0`), `--onecycle_final_div_factor` (default `1e4`).

Examples:

```bash
# Fixed LR (baseline)
python -u ./main_nbody.py --model HEGNN --ell 1 --data_directory <your_dir> \
  --dataset_name 5_0_0 --device cuda --scheduler none

# ReduceLROnPlateau on validation loss
python -u ./main_nbody.py --model HEGNN --ell 1 --data_directory <your_dir> \
  --dataset_name 5_0_0 --device cuda --scheduler plateau \
  --lr_patience 25 --lr_factor 0.5 --lr_min 1e-6

# Warmup + Cosine Annealing (T_max ~ total epochs - warmup)
python -u ./main_nbody.py --model HEGNN --ell 1 --data_directory <your_dir> \
  --dataset_name 5_0_0 --device cuda --scheduler warmup_cosine \
  --warmup_steps 50 --lr_t_max 950 --lr_min 1e-6 --epochs 1000

# OneCycleLR (per-batch schedule)
python -u ./main_nbody.py --model HEGNN --ell 1 --data_directory <your_dir> \
  --dataset_name 5_0_0 --device cuda --scheduler onecycle --epochs 300
```

LR Finder (range test):

```bash
python -u ./main_nbody.py --model HEGNN --ell 1 --data_directory <your_dir> \
  --dataset_name 5_0_0 --device cuda --lr_find --lr_find_steps 200 \
  --lr_find_min 1e-6 --lr_find_max 5e-2 --max_train_samples 512 --max_test_samples 512
```

The LR finder sweeps the learning rate, prints a suggested base LR, and saves a CSV curve to `./logs/nbody/lr_finder_<timestamp>.csv`.

### MD17 Dataset

#### Data Preparation

The MD17 dataset can be downloaded from [MD17](http://www.sgdml.org/#datasets).

#### Run Experiments

```bash
python -u ./main_md17.py --model HEGNN --ell 3 --batch_size 100 --epochs 500 --data_dir <your_dir> --delta_frame 3000 --mol "aspirin" --device --outf "md17-logs" 
```

### Perturbation Experiment

To run the experiment, please open `/expressivity/perturbation-test.ipynb`. 

## Acknowledgements

This project is based on the work from the [Geometric GNN Dojo](https://github.com/chaitjo/geometric-gnn-dojo) repository. We would like to express our gratitude to the original authors for their contributions to the field of geometric deep learning.

## Citation

If you find this work useful in your research, please consider citing:
```bibtex
@inproceedings{cen2024high,
  title={Are High-Degree Representations Really Unnecessary in Equivariant Graph Neural Networks?},
  author={Cen, Jiacheng and Li, Anyi and Lin, Ning and Ren, Yuxiang and Wang, Zihe and Huang, Wenbing},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=M0ncNVuGYN}
}
```
