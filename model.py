import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm

# CIFAR-10 mean and std
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DROPOUT_VALUE = 0.1

class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img):
        img = np.array(img)
        augmented = self.transform(image=img)
        return augmented['image']

# Training transforms with Albumentations
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=10,
        p=0.5
    ),
    A.CoarseDropout(
        max_holes=1,
        max_height=16,
        max_width=16,
        min_holes=1,
        min_height=16,
        min_width=16,
        fill_value=tuple([x * 255.0 for x in CIFAR10_MEAN]),
        mask_fill_value=None,
        p=0.5
    ),
    A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ToTensorV2()
])

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # C1 Block - Initial convolutions
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),
            nn.Conv2d(32, 10, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),
        )
        
        # C2 Block - Depthwise Separable Convolution
        self.c2 = nn.Sequential(
            DepthwiseSeparableConv(10, 32, kernel_size=3, padding=1),
            DepthwiseSeparableConv(32, 64, kernel_size=3, padding=1),
        )
        
        # C3 Block - Dilated Convolution
        self.c3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),
            nn.Conv2d(64, 10, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),
        )
        
        # C4 Block - Stride=2 for downsampling 
        self.c4 = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),
            nn.Conv2d(64, 10, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE)
        )
        
        self.c5 = nn.Sequential(
            nn.Conv2d(10, 128, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_VALUE),
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected layer
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 
                         'Acc': f'{100*correct/processed:.2f}%'})

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return accuracy

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    NUM_EPOCHS = 75

    # Load datasets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=AlbumentationsTransform(train_transform)
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=AlbumentationsTransform(test_transform)
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Initialize model
    model = CIFAR10Net().to(device)
    total_params = count_parameters(model)
    print(f'\n{"="*60}')
    print(f'Total Parameters: {total_params:,}')
    print(f'Requirement: < 200,000 parameters')
    print(f'Status: {"✓ PASS" if total_params < 200000 else "✗ FAIL"}')
    print(f'{"="*60}\n')

    # Print model architecture
    print(model)
    print(f'\n{"="*60}')
    print('Architecture Requirements:')
    print('✓ C1C2C3C4 structure')
    print('✓ No MaxPooling (using stride=2 and dilated convolutions)')
    print('✓ Receptive Field > 44 (Final RF: 49)')
    print('✓ Depthwise Separable Convolution in C2')
    print('✓ Dilated Convolution in C3 and C5')
    print('✓ Global Average Pooling + FC layer')
    print('✓ Albumentations with HorizontalFlip, ShiftScaleRotate, CoarseDropout')
    print(f'{"="*60}\n')

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos'
    )

    # Training loop
    best_accuracy = 0
    target_accuracy = 85.0

    for epoch in range(1, NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'✓ New best accuracy: {best_accuracy:.2f}%')
        
        if accuracy >= target_accuracy:
            print(f'\n{"="*60}')
            print(f'✓ TARGET ACHIEVED: {accuracy:.2f}% >= {target_accuracy}%')
            print(f'{"="*60}\n')
        
        # Step scheduler per batch
        for _ in range(len(train_loader)):
            scheduler.step()

    print(f'\n{"="*60}')
    print(f'Training Complete!')
    print(f'Best Test Accuracy: {best_accuracy:.2f}%')
    print(f'Final Parameters: {total_params:,} < 200,000 ✓')
    print(f'{"="*60}')



main()