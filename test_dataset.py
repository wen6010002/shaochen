#!/usr/bin/env python3
"""Test script to verify ModelNet40 dataset loading"""

import sys
sys.path.insert(0, 'pointnet.pytorch-master')

from pointnet.dataset import ModelNetDataset
import torch

print("Testing ModelNet40 dataset loading...")
print("-" * 50)

try:
    # Test training dataset
    print("\n1. Loading training dataset...")
    dataset = ModelNetDataset(
        root='modelnet40_normal_resampled',
        npoints=2500,
        split='trainval',
        data_augmentation=False
    )
    print(f"   ✓ Training dataset loaded successfully!")
    print(f"   - Total samples: {len(dataset)}")
    print(f"   - Classes: {len(dataset.classes)}")

    # Test loading one sample
    print("\n2. Loading a sample from training set...")
    points, label = dataset[0]
    print(f"   ✓ Sample loaded successfully!")
    print(f"   - Points shape: {points.shape}")
    print(f"   - Label: {label.item()}")

    # Test test dataset
    print("\n3. Loading test dataset...")
    test_dataset = ModelNetDataset(
        root='modelnet40_normal_resampled',
        npoints=2500,
        split='test',
        data_augmentation=False
    )
    print(f"   ✓ Test dataset loaded successfully!")
    print(f"   - Total samples: {len(test_dataset)}")

    # Test loading test sample
    print("\n4. Loading a sample from test set...")
    points, label = test_dataset[0]
    print(f"   ✓ Sample loaded successfully!")
    print(f"   - Points shape: {points.shape}")
    print(f"   - Label: {label.item()}")

    print("\n" + "=" * 50)
    print("✓ All tests passed! Dataset is ready for training.")
    print("=" * 50)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
