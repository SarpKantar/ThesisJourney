# RN18/CIFAR10 LeJEPA Sweep - 2026-07-29

Purpose: test whether the `~86%` plateau is caused by under-trained linear
probes or by the current LeJEPA SSL configuration.

## Shared Changes

- Training script now supports paper-style probes:
  `LayerNorm(512) + Linear`, AdamW, configurable LR and weight decay.
- SSL config is now exposed through Slurm:
  projector dimension, hidden dimension, lambda, views, crop scale, color jitter,
  blur, and solarize.
- New post-hoc probe script:
  `scripts/probe_rn18_cifar10_lejepa_checkpoints.py`.

## Submitted Jobs

| Job ID | Type | Output | Main Config |
|---:|---|---|---|
| 1429984 | post-hoc probe | `outputs/rn18_cifar10_true_lejepa_probe_layernorm_adamw_100` | Existing checkpoints, selected SSL epochs, `LayerNorm+Linear`, AdamW, 100 epochs, wd `1e-6` |
| 1429986 | SSL sweep | `outputs/rn18_cifar10_true_lejepa_cfg_l02_v4_p16_strong` | `lambda=0.02`, `V=4`, `proj_dim=16`, strong aug, SSL lr `2e-3`, 120 epochs |
| 1429987 | SSL sweep | `outputs/rn18_cifar10_true_lejepa_cfg_l02_v4_p64_strong` | `lambda=0.02`, `V=4`, `proj_dim=64`, strong aug, SSL lr `2e-3`, 120 epochs |
| 1429988 | SSL sweep | `outputs/rn18_cifar10_true_lejepa_cfg_l02_v8_p64_moderate` | `lambda=0.02`, `V=8`, `proj_dim=64`, moderate aug, SSL lr `5e-4`, 120 epochs |
| 1429989 | SSL sweep | `outputs/rn18_cifar10_true_lejepa_cfg_l05_v4_p64_strong` | `lambda=0.05`, `V=4`, `proj_dim=64`, strong aug, SSL lr `5e-4`, 120 epochs |

All SSL sweeps use final probes with `LayerNorm+Linear`, AdamW, LR `1e-3`,
weight decay `1e-6`, 50 linear epochs, checkpoints every 10 SSL epochs, and no
early stopping.

## Monitoring

```bash
squeue -j 1429984,1429986,1429987,1429988,1429989
tail -f logs/rn18_cifar10_lejepa_probe_cudnn/rn18-probe-ln-adamw100-1429984.out
tail -f logs/rn18_cifar10_true_lejepa_cudnn/rn18-lj-l02-v4-p16-strong-1429986.out
```
