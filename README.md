# UNIBO IPCV Aircraft Image Classification

This repository contains a Jupyter notebook solution for the Module 2 image
classification assignment. The task is to classify 100 aircraft variants from
the FGVC-Aircraft dataset.

## Contents

- `assignment_module_two.ipynb` - complete experiment notebook with markdown,
  training code, ablations, plots, and result tables.
- `original.ipynb` - untouched assignment statement used as the source for the
  original markdown cells.
- `LABS/` - reference lab notebooks whose coding style is followed in the
  assignment solution.
- `notes.txt` - original planning notes for the assignment.
- `README.md` - this repository guide.

## Assignment Workflow

The notebook is organized into two main parts.

1. Custom CNN from scratch
   - Uses only PyTorch layers, not pretrained torchvision architectures.
   - Trains a best custom CNN configuration.
   - Runs ablations by removing strong augmentation, batch normalization,
     dropout, and model capacity.
   - Saves comparison tables and training curves.

2. ResNet-18 fine-tuning
   - Uses torchvision ResNet-18 with ImageNet-1K V1 weights.
   - First trains with the same training hyperparameters as the best custom CNN.
   - Then tests transfer-learning choices: frozen backbone vs full fine-tuning,
     stronger augmentation, and separate learning rates for backbone and head.

## Requirements

Use an environment with Python 3.8+ and these packages:

```bash
pip install torch torchvision matplotlib numpy pandas jupyter
```

If you use the course environment, it may already include these dependencies.

## Running

Open the notebook:

```bash
jupyter notebook assignment_module_two.ipynb
```

The runtime settings cell contains the main run switches:

```python
DOWNLOAD_DATA = True
RUN_TRAINING = True
FAST_DEV_RUN = False
SHOW_DATASET_PREVIEW = True
```

For a quick pipeline check, set `FAST_DEV_RUN = True`. For final assignment
results, keep it `False` and run all training cells. Set `RUN_TRAINING = False`
only when you want to inspect the notebook without starting the experiments.

The dataset is downloaded automatically through
`torchvision.datasets.FGVCAircraft` into `data/`.

## Generated Outputs

When the notebook is run, it writes generated files under `outputs/`:

- `outputs/ckpts/*.pt` - best model checkpoint for each experiment in the
  current lab-style notebook.
- `outputs/histories/*.csv` - per-epoch training and validation metrics.
- `outputs/*_summary.json` - final test metrics for each experiment.
- `outputs/final_model_comparison.csv` - combined result table.

The final comparison table and plots are displayed inside the notebook and can
be used directly in the written analysis cells.

## Notes

The custom CNN target is around 50 percent test accuracy. The ResNet-18 target
is around 70 percent test accuracy. Exact numbers depend on hardware, random
seed, torchvision version, and how long the experiments are allowed to train.
