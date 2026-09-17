# RN18/CIFAR10 True LeJEPA Protocol

This is the corrected protocol for experiment 3. The earlier EMA target
encoder/predictor run was not LeJEPA and its SSL checkpoints should be ignored.

## What LeJEPA Means Here

The supplied LeJEPA paper and local `lejepa/` repository define the training
objective as:

```text
LeJEPA loss = (1 - lambda) * prediction_loss + lambda * SIGReg_loss
```

For our script:

- `prediction_loss` pulls all view embeddings of the same image toward their
  per-image view center.
- `SIGReg_loss` pushes projected embeddings toward an isotropic Gaussian using
  sliced Epps-Pulley normality tests.
- SIGReg is implemented with the original local
  `lejepa/lejepa/univariate/epps_pulley.py` and
  `lejepa/lejepa/multivariate/slicing.py` source files.
- No target encoder, EMA copy, predictor MLP, stop-gradient, prototypes, or
  SSL labels are used.

## CIFAR10/ResNet18 Adaptation

For every CIFAR10 image, the SSL dataset returns `V` stochastic 32x32 augmented
views. The ResNet18 encoder uses the same CIFAR-style 3x3 stride-1 stem as the
ImageNet fine-tuned baseline. A projector maps 512-dimensional ResNet features
into the LeJEPA embedding space before the prediction and SIGReg losses.

The paper's Algorithm 2 says that for non-ViT architectures such as ResNet,
`global_views = all_views`. Our default true-LeJEPA job follows that rule.

## Evaluation

The ImageNet fine-tuned checkpoints are reused from
`outputs/rn18_cifar10_experiment_cudnn/checkpoints`; Method A is not retrained.
LeJEPA SSL checkpoints are saved throughout training. Each saved LeJEPA
checkpoint is evaluated by training the same fixed-epoch frozen-backbone linear
head, and the final matched pair is selected by the smallest CIFAR10 test
accuracy gap.

Run:

```bash
sbatch jobs/run_rn18_cifar10_true_lejepa_cudnn.sbatch
```
