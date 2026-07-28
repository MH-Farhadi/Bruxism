"""
This Script uses an MLP network with handcrafted statistical features (RMS, MAV, Variance, Waveform Length,
Zero Crossings) extracted from EMG signals instead of raw time-series data.
Well-suited for very small datasets where feature-based approaches often
outperform deep learning methods.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from bruxism_dataset import create_train_test_split
from training_improvements import (
    FeatureBasedDataset,
    SimpleFeatureClassifier,
    ReducedClassDataset,
    train_with_early_stopping
)


def main():
    """
    Trains a simple MLP classifier on handcrafted statistical features extracted
    from EMG signals. Uses the same preprocessing pipeline as wavelet training
    for fair comparison. 
    """
    print("="*70)
    print("FEATURE-BASED TRAINING PIPELINE")
    print("="*70)
    print("\nThis approach extracts statistical features from EMG signals")
    print("instead of using raw time-series data. Often better for small datasets.\n")
    
    data_root = r"C:\Users\mhfar\Desktop\Depo\Brusxism_data"
    
    print("1. Loading base datasets...")
    train_dataset, test_dataset = create_train_test_split(
        data_root=data_root,
        train_subjects=[1, 2, 3, 4],
        test_subjects=[5],
        window_size=1200,
        stride=600,
        normalize=False
    )
    
    print(f"\nRaw train dataset: {len(train_dataset)} samples")
    print(f"Raw test dataset: {len(test_dataset)} samples")
    
    # Reduce to 4 classes and removes 'rest' (poor quality)
    reduction_strategy = '4_classes'
    print(f"\nUsing reduction strategy: {reduction_strategy}")
    print("Note: REMOVING 'rest' class - it's contaminated and causing confusion.")
    print("      Focusing on active states only: movement, clenching, bruxing, chewing.\n")
    
    print("\n2. Extracting features (same preprocessing as wavelet training)...")
    # Extract handcrafted features: RMS, MAV, Variance, Waveform Length, Zero Crossings
    # 5 features per channel × 4 channels = 20 features per sample
    train_features_base = FeatureBasedDataset(
        train_dataset,
        apply_preprocessing=True,
        use_ica=False
    )
    
    # Fit scaler on training set only to prevent data leakage
    scaler = train_features_base.get_scaler()
    
    # Apply training scaler to test set for consistent feature scaling
    test_features_base = FeatureBasedDataset(
        test_dataset,
        apply_preprocessing=True,
        use_ica=False,
        scaler=scaler
    )
    
    print("\n2b. Applying class reduction...")
    train_features = ReducedClassDataset(train_features_base, reduction_strategy=reduction_strategy)
    test_features = ReducedClassDataset(test_features_base, reduction_strategy=reduction_strategy)
    
    num_classes = train_features.num_classes
    class_names = train_features.get_class_names()
    
    print(f"\nFeature dataset sizes:")
    print(f"Train: {len(train_features)} samples")
    print(f"Test: {len(test_features)} samples")
    print(f"Feature dimension: {train_features_base.features.shape[1]} features per sample")
    print(f"Number of classes: {num_classes}")
    
    # Analyze class distribution to assess imbalance
    print("\n2c. Analyzing class distribution...")
    from collections import Counter
    train_labels = [train_features[i][1] for i in range(len(train_features))]
    test_labels = [test_features[i][1] for i in range(len(test_features))]
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    
    print("Train set class distribution:")
    for class_idx in range(num_classes):
        count = train_counts.get(class_idx, 0)
        percentage = 100 * count / len(train_features) if len(train_features) > 0 else 0
        print(f"  {class_names[class_idx]:15s}: {count:4d} samples ({percentage:5.1f}%)")
    
    print("\nTest set class distribution:")
    for class_idx in range(num_classes):
        count = test_counts.get(class_idx, 0)
        percentage = 100 * count / len(test_features) if len(test_features) > 0 else 0
        print(f"  {class_names[class_idx]:15s}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Detect class imbalance to determine if weighted loss is needed
    train_class_counts = [train_counts.get(i, 0) for i in range(num_classes)]
    if len(train_class_counts) > 0:
        max_count = max(train_class_counts)
        min_count = min(train_class_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}x (max/min)")
        if imbalance_ratio > 3.0:
            print("⚠️  Significant class imbalance detected! Using class weights in loss function.")
    
    print("\n3. Creating data loaders...")
    batch_size = 64
    train_loader = DataLoader(train_features, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_features, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print("\n4. Creating simple feature classifier...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Simple MLP with strong regularization to prevent overfitting on small dataset
    input_dim = train_features_base.features.shape[1]
    model = SimpleFeatureClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=64,
        dropout=0.7
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (very small - perfect for tiny datasets)")
    
    print("\n5. Setting up training with strong regularization...")
    # Compute inverse-frequency class weights to balance training
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=np.array(train_labels)
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    # Convert class weights to tensor for loss function
    weight_tensor = torch.tensor([class_weight_dict[i] for i in range(num_classes)], dtype=torch.float32).to(device)
    
    # Use weighted loss with label smoothing for additional regularization
    try:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
        print("   Using CrossEntropyLoss with class weights and label_smoothing=0.1")
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print("   Using CrossEntropyLoss with class weights")
    
    # Conservative learning rate and high weight decay for stable training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print("\n6. Training with early stopping (monitoring accuracy)...")
    train_losses, train_accs, val_losses, val_accs = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=200,
        patience=10,
        monitor='accuracy'
    )
    
    print("\n7. Analyzing results...")
    best_val_acc = max(val_accs)
    best_val_epoch = val_accs.index(best_val_acc) + 1
    final_train_acc = train_accs[val_accs.index(best_val_acc)]
    gap = final_train_acc - best_val_acc
    
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (epoch {best_val_epoch})")
    print(f"Corresponding Train Accuracy: {final_train_acc:.2f}%")
    print(f"Train-Val Gap: {gap:.2f}%")
    print(f"Total Epochs Run: {len(train_losses)}")
    
    # Assess generalization quality based on train-val gap
    if gap < 10:
        print(f"\n✅ Excellent! Very small train-val gap indicates great generalization")
    elif gap < 20:
        print(f"\n✅ Good! Small train-val gap indicates good generalization")
    elif gap < 30:
        print(f"\n⚠️ Moderate overfitting. Still acceptable for feature-based approach.")
    else:
        print(f"\n❌ High overfitting. Consider even stronger regularization.")
    
    # Assessing performance
    if best_val_acc > 60:
        print(f"✅ Excellent validation accuracy for feature-based approach!")
    elif best_val_acc > 45:
        print(f"⚠️ Moderate performance. Consider running cross-validation.")
    else:
        print(f"❌ Low performance. May need more data or different features.")
    
    print("\n8. Plotting results...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss (Feature-Based)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(val_accs, label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy (Feature-Based)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_based_training_results.png', dpi=150, bbox_inches='tight')
    print("Saved plot to 'feature_based_training_results.png'")
    plt.show()
    
    print("\n9. Evaluating on test set...")
    # Load best model checkpoint saved during early stopping
    model.load_state_dict(torch.load('best_model_early_stopping.pth'))
    model.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features_batch, labels_batch in test_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            outputs = model(features_batch)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
    
    test_acc = 100 * correct / total
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    condition_names = class_names
    
    print("\nPer-Class Performance:")
    print(classification_report(all_labels, all_predictions, target_names=condition_names))
    
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Feature-Based MLP ({num_classes} classes)')
    plt.colorbar()
    tick_marks = np.arange(len(condition_names))
    plt.xticks(tick_marks, condition_names, rotation=45, ha='right')
    plt.yticks(tick_marks, condition_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_feature_mlp_{num_classes}classes.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved confusion matrix to 'confusion_matrix_feature_mlp_{num_classes}classes.png'")
    plt.show()
    
    print("\n" + "="*70)
    print("METHODOLOGY SUMMARY")
    print("="*70)
    print("This Feature-based MLP approach now uses the SAME methodology as wavelet training:")
    print("  ✓ FeatureBasedDataset (handcrafted features)")
    print("  ✓ ReducedClassDataset (4 classes, rest removed)")
    print("  ✓ StandardScaler normalization")
    print("  ✓ Class weights for imbalance handling")
    print("  ✓ Same preprocessing (apply_preprocessing=True, use_ica=False)")
    print("\nOnly difference: Simple MLP classifier vs WaveletFeatureCNN")
    print("This allows fair comparison between classification methods.")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("Feature-based approach completed!")
    print("\nIf results are good (>50% val acc):")
    print("  - This is likely the best approach for your 5-subject dataset")
    print("  - Consider: 1) Adding more features, 2) Running cross-validation")
    print("\nIf results are still poor (<40% val acc):")
    print("  - The dataset may be too small for reliable classification")
    print("  - Consider collecting data from more subjects (aim for 10-15)")
    print("  - Or use a simpler classification task (fewer classes)")
    
    print("\n✨ Feature-based training complete!")


if __name__ == "__main__":
    main()

