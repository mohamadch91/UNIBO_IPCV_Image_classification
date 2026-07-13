# UNIBO IPCV Aircraft Image Classification

This repository contains the solution notebook for Module 2 of the UNIBO Image
Processing and Computer Vision assignment. It classifies the 100 aircraft
variants in the
[FGVC-Aircraft dataset](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
using a custom convolutional network and a pretrained ResNet-18.

## Notebook workflow

The complete implementation, experiments, plots, tables, and discussion are in
`assignment_module_two.ipynb`.

### Part 1: custom CNN

The notebook builds a compact ResNet-like CNN from standard PyTorch layers. It
uses a convolutional stem, four residual stages with channels
`[32, 64, 128, 256]`, Batch Normalization, global average pooling, dropout, and
a 100-class linear classifier. No off-the-shelf torchvision architecture is
used for this part.

The selected configuration uses light augmentation, mixup, label smoothing,
AdamW, and OneCycleLR. Its ablation study evaluates:

- no augmentation and strong augmentation;
- no mixup;
- no label smoothing;
- no dropout and excessive dropout;
- no Batch Normalization;
- a lower learning rate.

### Part 2: pretrained ResNet-18

The notebook loads torchvision's ResNet-18 with ImageNet-1K V1 weights and
replaces its final classifier for the 100 aircraft variants.

- Part 2A compares a frozen backbone with whole-model training while reusing
  the Part 1 hyperparameters.
- Part 2B directly fine-tunes the complete network and studies frozen-backbone
  training, no/strong augmentation, removal of mixup or label smoothing, and
  differential learning rates for the backbone and classifier.
- The notebook also discusses earlier frozen-backbone, learning-rate, and
  staged fine-tuning experiments.

## Reported results

These are the selected valid results preserved in the notebook:

| Model | Best validation accuracy | Test accuracy | Test loss | Trainable parameters |
| --- | ---: | ---: | ---: | ---: |
| Custom CNN (`best_model`) | 56.50% | 57.13% | 1.7122 | 2,824,580 |
| ResNet-18 (`resnet18_best`) | 71.59% | 73.03% | 1.3404 | 11,227,812 |

The custom model exceeds the assignment's approximate 50% target, and the
fine-tuned ResNet-18 exceeds the approximate 70% target. The exploratory
`resnet18_best_dropout_0.3` row visible in the saved comparison is intentionally
excluded: as explained in the notebook, changing the required ResNet-18
classifier architecture makes that run invalid for the assignment.

## Repository contents

- `assignment_module_two.ipynb` - complete assignment, implementation,
  experiments, results, and written analysis.
- `outputs/histories/` - per-epoch training and validation metrics saved as
  CSV files.
- `outputs/summaries/` - scalar metrics and configurations saved as JSON.
- `outputs/plots/` - training curves and selected confusion matrices.
- `outputs/final_model_comparison.csv` - combined table of the final notebook
  experiments.
- `outputs/analysis/` and `outputs/totals/` - preserved aggregate analyses from
  model development.
- `LABS/` - course lab notebooks used as implementation references.

Model checkpoints are written to `outputs/ckpts/`, but that directory is
ignored by Git because checkpoint files are large.

## Requirements

The notebook metadata records Python 3.10. Install the imported dependencies
with:

```bash
python -m pip install jupyter torch torchvision numpy pandas matplotlib pillow \
  scikit-learn tqdm gdown torchsummary
```

CUDA is used when available, followed by Apple MPS, with CPU as the fallback.
An internet connection is required on the first run to download FGVC-Aircraft,
the pretrained ResNet-18 weights, and report assets hosted on Google Drive.

## Running the notebook

From the repository root, start Jupyter:

```bash
jupyter notebook assignment_module_two.ipynb
```

The runtime settings cell contains these main switches:

```python
DOWNLOAD_DATA = True
RUN_TRAINING = True
SHOW_DATASET_PREVIEW = True
SHOW_CONFUSION_MATRIX = False
```

Keep `DOWNLOAD_DATA = True` for the first run; the dataset is stored under
`data/`. Set `RUN_TRAINING = False` when reading the preserved notebook without
rerunning the experiments. A complete training run executes many long
experiments, so it is best run on a CUDA- or MPS-capable machine.

Each training experiment saves:

```text
outputs/
├── ckpts/<run_name>.pt
├── histories/<run_name>.csv
├── plots/<run_name>.png
├── summaries/<run_name>_summary.json
└── final_model_comparison.csv
```

## TODO

- [ ] Review and remove unnecessary files before the final submission, while
  retaining the notebook and the artifacts needed to support its results.
