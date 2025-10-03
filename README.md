# cifar10
CNN architecture to classify CIFAR 10 images

## CIFAR-10 Training with PyTorch and Albumentations

This project trains a compact CNN (< 200k parameters) on CIFAR-10 using PyTorch. It features modern augmentations via Albumentations, depthwise separable and dilated convolutions, OneCycleLR scheduling, and automatic checkpointing of the best model.

### Features
- **Model**: C1→C2→C3→C4→C5 blocks with depthwise separable convs, dilated convs, GAP + FC head
- **Params**: < 200,000 (checked at runtime)
- **Augmentations**: HorizontalFlip, ShiftScaleRotate, CoarseDropout, Normalize (Albumentations)
- **Training**: Adam optimizer, OneCycleLR, tqdm progress bars
- **Checkpointing**: Saves `best_model.pth` when test accuracy improves
- **Target**: 85% test accuracy (script reports when reached)

### Environment and Requirements
- Python 3.8+
- PyTorch, torchvision
- Albumentations
- numpy, tqdm

Install (CPU-only example; adjust for CUDA as needed):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install albumentations==1.* numpy tqdm
```

For CUDA (example for CUDA 12.1; see PyTorch site for your setup):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install albumentations==1.* numpy tqdm
```

### Dataset
CIFAR-10 is automatically downloaded to `./data` at first run by `torchvision.datasets.CIFAR10`.

### How to Run
From the `CIFAR10` directory:
```bash
python model_train.py
```

- The script auto-selects device: CUDA if available, else CPU.
- Progress and metrics print to the console.
- Best checkpoint is saved as `best_model.pth` in the working directory.

### Model Overview
- **Blocks**
  - **C1**: Conv(3→32) → BN → ReLU → Dropout → Conv(32→32) → BN → ReLU → Dropout → Conv(32→10, 1×1) → BN → ReLU → Dropout
  - **C2**: DepthwiseSeparableConv(10→32) → DepthwiseSeparableConv(32→64)
  - **C3**: Conv(64→64, dilated) → BN → ReLU → Conv(64→64) → BN → ReLU → Dropout → Conv(64→10, 1×1) → BN → ReLU → Dropout
  - **C4**: Conv(10→64) → BN → ReLU → Dropout → Conv(64→64, stride=2) → BN → ReLU → Dropout → Conv(64→10, 1×1) → BN → ReLU → Dropout
  - **C5**: Conv(10→128, dilated) → BN → ReLU → Dropout
  - **Head**: Global Average Pooling → Linear(128→10)
- **Key properties**
  - No MaxPool (uses stride=2)
  - Receptive field ≥ 49
  - Parameter count reported at start and must be < 200k

### Data Augmentation
Training pipeline (Albumentations):
- HorizontalFlip(p=0.5)
- ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5)
- CoarseDropout(1× 16×16 patch, p=0.5) with mean-color fill
- Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616))
- ToTensorV2

Test pipeline:
- Normalize + ToTensorV2

### Training Configuration (in-code)
- `NUM_EPOCHS`: 75
- `batch_size`: 128 (in `DataLoader`)
- Optimizer: Adam(lr=1e-3, weight_decay=1e-4)
- Scheduler: OneCycleLR(max_lr=1e-2, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.2, anneal_strategy='cos')
- Dropout: 0.1 (via `DROPOUT_VALUE`)
- Target accuracy: 85% (reports when reached)
- Checkpoint: Saves `best_model.pth` whenever test accuracy improves

To tweak hyperparameters, edit them directly in `model_train.py`.

### Outputs
- Console logs include parameter count, architecture notes, live training loss/accuracy, test loss/accuracy.
- Best model weights are saved to `best_model.pth`.

### Reproducibility Tips
- Set a seed at the top of the script if you want deterministic behavior:
```python
import torch, numpy as np, random
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
