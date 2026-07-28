"""
This Script is a Random Forest classifier with the same preprocessing methodology as the wavelet training script,
for fair comparison. Random Forest is well-suited for small datasets, feature-based
classification, and handling class imbalance while providing feature importance insights.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from collections import Counter

from bruxism_dataset import create_train_test_split
from wavelet_dataset import WaveletFeatureDataset
from training_improvements import ReducedClassDataset

# Class reduction mappings: group semantically similar activities into single classes
# Strategy: Reduce 12 original classes to 4-6 classes to improve generalization
# on small datasets by combining similar activities (e.g., different clench types)
CLASS_REDUCTION_STRATEGIES = {
    '4_classes': {
        'movement': 0,  # open_close, deviation_left_right, protrusion_retrusion
        'clenching': 1,  # bite_left, bite_right, molar_clench, incisor_clench
        'bruxing': 2,    # natural_bruxing
        'chewing': 3,    # cheese, carrots, gum
    },
    '5_classes': {
        'rest': 0,
        'movement': 1,
        'clenching': 2,
        'bruxing': 3,
        'chewing': 4,
    },
    '6_classes': {
        'rest': 0,
        'movement': 1,
        'biting': 2,
        'clenching': 3,
        'bruxing': 4,
        'chewing': 5,
    },
}

ORIGINAL_CLASSES = {
    'rest': 0,
    'open_close': 1,
    'deviation_left_right': 2,
    'protrusion_retrusion': 3,
    'bite_left': 4,
    'bite_right': 5,
    'molar_clench': 6,
    'incisor_clench': 7,
    'natural_bruxing': 8,
    'cheese': 9,
    'carrots': 10,
    'gum': 11
}


def load_data_for_rf(reduction_strategy='4_classes'):
    """
    Load and prepare data for Random Forest training.
    
    Uses the same preprocessing pipeline as wavelet training for consistency:
    extracts wavelet features, applies class reduction, and converts to numpy arrays.
    Includes verification steps to ensure data integrity and prevent misalignment issues.
    
    Args:
        reduction_strategy (str): Class reduction strategy - '4_classes', '5_classes', or '6_classes'.
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test, class_names, num_classes, train_reduced, test_reduced)
            - X_train, X_test: Feature arrays (numpy)
            - y_train, y_test: Label arrays (numpy)
            - class_names: List of reduced class names
            - num_classes: Number of classes after reduction
            - train_reduced, test_reduced: Reduced dataset objects
    """
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
    
    print("\n2. Extracting wavelet features (same as wavelet training)...")
    train_wavelet_base = WaveletFeatureDataset(
        train_dataset,
        apply_preprocessing=True,
        use_ica=False,
        wavelet='db4',
        max_level=5
    )
    
    # Fit scaler on training set only to prevent data leakage
    scaler = train_wavelet_base.get_scaler()
    
    # Apply training scaler to test set for consistent feature scaling
    test_wavelet_base = WaveletFeatureDataset(
        test_dataset,
        apply_preprocessing=True,
        use_ica=False,
        wavelet='db4',
        max_level=5,
        scaler=scaler
    )
    
    print("\n2b. Applying class reduction...")
    train_reduced = ReducedClassDataset(train_wavelet_base, reduction_strategy=reduction_strategy)
    test_reduced = ReducedClassDataset(test_wavelet_base, reduction_strategy=reduction_strategy)
    
    # Extract features and labels together to maintain alignment.
    # We extract from same dataset iteration to prevent misalignment
    # that could cause model to predict only one class
    print("\nExtracting features and labels from reduced datasets...")
    
    X_train_list = []
    y_train_list = []
    for i in range(len(train_reduced)):
        features, label = train_reduced[i]
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        X_train_list.append(features)
        y_train_list.append(label)
    
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    
    X_test_list = []
    y_test_list = []
    for i in range(len(test_reduced)):
        features, label = test_reduced[i]
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        X_test_list.append(features)
        y_test_list.append(label)
    
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    
    print(f"Extracted features: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"Extracted features: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    
    class_names = train_reduced.get_class_names()
    num_classes = train_reduced.num_classes
    
    # Verify label distribution and data integrity
    train_label_counts = Counter(y_train)
    test_label_counts = Counter(y_test)
    print(f"\nTrain label distribution: {dict(train_label_counts)}")
    print(f"Test label distribution: {dict(test_label_counts)}")
    
    # Verify labels are diverse and in expected range
    print(f"\nLabel verification:")
    print(f"  Unique train labels: {sorted(np.unique(y_train))}")
    print(f"  Unique test labels: {sorted(np.unique(y_test))}")
    print(f"  Expected label range: 0 to {num_classes-1}")
    print(f"  All labels in range: {np.all((y_train >= 0) & (y_train < num_classes))}")
    print(f"  All labels are integers: {np.all(y_train == y_train.astype(int))}")
    
    # Detect critical error: all labels identical (would cause single-class prediction)
    if len(np.unique(y_train)) == 1:
        print(f"  ⚠️  CRITICAL ERROR: All training labels are the same value: {y_train[0]}")
        print(f"  This will cause the model to always predict this class!")
    else:
        print(f"  ✓ Labels are diverse (not all the same)")
    
    # Verify feature-label alignment by spot-checking samples
    print(f"\nVerifying feature-label alignment (checking first 5 samples):")
    for i in range(min(5, len(train_reduced))):
        features_check, label_check = train_reduced[i]
        if isinstance(features_check, torch.Tensor):
            features_check = features_check.numpy()
        features_match = np.allclose(features_check, X_train[i], atol=1e-6)
        label_match = (label_check == y_train[i])
        print(f"  Sample {i}: features_match={features_match}, label_match={label_match}, label={label_check}")
    
    return X_train, y_train, X_test, y_test, class_names, num_classes, train_reduced, test_reduced


def analyze_feature_separability(X_train, y_train, X_test, y_test, class_names):
    """
    Analyze feature separability between classes.
    
    Examines feature statistics per class, tests simple threshold-based separation
    (especially for 'rest' class), and verifies feature normalization to identify
    potential data quality issues.
    
    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): Test labels.
        class_names (list): List of class name strings.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    print("\n" + "="*70)
    print("FEATURE SEPARABILITY ANALYSIS")
    print("="*70)
    
    # Analyze feature statistics per class to identify class-specific patterns
    print("\n1. Feature Statistics by Class (last 5 features - rest-specific):")
    rest_idx = class_names.index('rest') if 'rest' in class_names else -1
    movement_idx = class_names.index('movement') if 'movement' in class_names else -1
    
    for class_idx in range(len(class_names)):
        mask = y_train == class_idx
        if np.sum(mask) > 0:
            class_features = X_train[mask]
            last_5_features = class_features[:, -5:].mean(axis=0)
            
            print(f"\n{class_names[class_idx]:15s} (n={np.sum(mask):4d}):")
            print(f"  Mean RMS: {last_5_features[0]:.6f}")
            print(f"  Mean Var: {last_5_features[1]:.6f}")
            print(f"  Activity Level: {last_5_features[-1]:.6f}")
            
            # Compare rest class activity level to other classes
            if rest_idx >= 0 and class_idx != rest_idx:
                rest_mask = y_train == rest_idx
                if np.sum(rest_mask) > 0:
                    rest_features = X_train[rest_mask]
                    rest_activity = rest_features[:, -1].mean()
                    class_activity = class_features[:, -1].mean()
                    ratio = rest_activity / (class_activity + 1e-8)
                    if ratio > 0.5:
                        print(f"  [WARNING] Activity level only {ratio:.2f}x lower than {class_names[class_idx]}")
    
    # Test if 'rest' class can be separated using simple activity threshold
    if rest_idx >= 0:
        rest_mask_train = y_train == rest_idx
        rest_mask_test = y_test == rest_idx
        
        if np.sum(rest_mask_train) > 0 and np.sum(rest_mask_test) > 0:
            # Use activity level feature (last feature) for threshold test
            activity_feature = X_train[:, -1]
            rest_activity = activity_feature[rest_mask_train]
            other_activity = activity_feature[~rest_mask_train]
            
            # Set threshold at 10th percentile of non-rest activity
            threshold = np.percentile(other_activity, 10)
            
            # Measure separation quality
            rest_below_threshold = np.sum(rest_activity < threshold) / len(rest_activity) * 100
            other_below_threshold = np.sum(other_activity < threshold) / len(other_activity) * 100
            
            print(f"\n2. Simple Threshold Test (Activity Level < {threshold:.6f}):")
            print(f"   Rest samples below threshold: {rest_below_threshold:.1f}%")
            print(f"   Other samples below threshold: {other_below_threshold:.1f}%")
            
            if rest_below_threshold < 50:
                print(f"   [PROBLEM] Rest is NOT primarily low activity!")
                print(f"   This suggests preprocessing or feature extraction issue.")
    
    # Verify feature normalization to detect preprocessing issues
    print(f"\n3. Feature Ranges (to detect normalization issues):")
    print(f"   Min feature value: {X_train.min():.6f}")
    print(f"   Max feature value: {X_train.max():.6f}")
    print(f"   Mean feature value: {X_train.mean():.6f}")
    print(f"   Std feature value: {X_train.std():.6f}")
    
    if np.abs(X_train.mean()) < 0.01 and np.abs(X_train.std() - 1.0) < 0.1:
        print(f"   [OK] Features appear to be standardized")
    else:
        print(f"   [WARNING] Features may not be properly standardized")


def main():
    """
    Main Random Forest training pipeline for bruxism detection.
    Handles class imbalance through aggressive class weighting
    and provides feature importance analysis.
    """
    print("="*70)
    print("RANDOM FOREST TRAINING PIPELINE")
    print("="*70)
    print("\nUsing Random Forest classifier - better for small datasets and features.\n")
    
    # Reduce to 4 classes by removing 'rest' (contaminated/poor quality)
    reduction_strategy = '4_classes'
    print(f"Using reduction strategy: {reduction_strategy}\n")
    print("Note: REMOVING 'rest' class - it's contaminated and causing confusion.")
    print("      Focusing on active states only: movement, clenching, bruxing, chewing.\n")
    
    # Load data using same methodology as wavelet training
    X_train, y_train, X_test, y_test, class_names, num_classes, train_reduced, test_reduced = load_data_for_rf(reduction_strategy)
    
    # Analyze feature separability to identify potential data quality issues
    analyze_feature_separability(X_train, y_train, X_test, y_test, class_names)
    
    print(f"\nFeature dimension: {X_train.shape[1]} features per sample (wavelet features)")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of classes: {num_classes}")
    
    # Analyze class distribution to assess imbalance
    print("\n2d. Analyzing class distribution...")
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)
    
    print("Train set class distribution (before augmentation):")
    for class_idx in range(num_classes):
        count = train_counts.get(class_idx, 0)
        percentage = 100 * count / len(y_train) if len(y_train) > 0 else 0
        print(f"  {class_names[class_idx]:15s}: {count:4d} samples ({percentage:5.1f}%)")
    
    print("\nTest set class distribution:")
    for class_idx in range(num_classes):
        count = test_counts.get(class_idx, 0)
        percentage = 100 * count / len(y_test) if len(y_test) > 0 else 0
        print(f"  {class_names[class_idx]:15s}: {count:4d} samples ({percentage:5.1f}%)")
    
    # Detect class imbalance to determine weighting strategy
    train_class_counts = [train_counts.get(i, 0) for i in range(num_classes)]
    if len(train_class_counts) > 0:
        max_count = max(train_class_counts)
        min_count = min(train_class_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}x (max/min)")
        if imbalance_ratio > 3.0:
            print("⚠️  Significant class imbalance detected! Using class weights.")
    
    # Random Forest uses class weights instead of augmentation (works with numpy arrays)
    print("\n2c. Class balancing approach...")
    print("   Note: Using class weights for imbalance (same as wavelet training).")
    print("   Augmentation is not applicable for Random Forest (works with numpy arrays directly).")
    
    # Compute class weights using two methods: standard balancing and inverse frequency with square root damping
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes = np.unique(y_train)
    
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    n_classes = len(unique_classes)
    
    # Standard sklearn balanced weights
    class_weights_balanced = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=y_train
    )
    
    # Custom aggressive weighting: inverse frequency with square root damping
    # More aggressive than 'balanced' to better handle severe imbalance
    max_count = max(class_counts.values())
    class_weights_aggressive = {}
    for class_label in unique_classes:
        count = class_counts[class_label]
        # Square root damping prevents extreme weights while still emphasizing minorities
        weight = np.sqrt(max_count / count) * (total_samples / (n_classes * count))
        class_weights_aggressive[class_label] = weight
    
    # Normalize aggressive weights to similar scale as balanced weights
    balanced_scale = np.mean(class_weights_balanced)
    aggressive_scale = np.mean(list(class_weights_aggressive.values()))
    if aggressive_scale > 0:
        scale_factor = balanced_scale / aggressive_scale
        class_weights_aggressive = {k: v * scale_factor for k, v in class_weights_aggressive.items()}
    
    # Use aggressive weights for better minority class handling
    class_weight_dict = class_weights_aggressive
    
    print(f"\nClass distribution: {dict(class_counts)}")
    print(f"Standard balanced weights: {dict(zip(unique_classes, class_weights_balanced))}")
    print(f"Aggressive custom weights: {class_weight_dict}")
    print(f"Using aggressive custom weights to better handle severe imbalance")
    
    # Verify we have all expected classes
    print(f"\nUnique classes in training data: {sorted(unique_classes)}")
    print(f"Expected classes: {list(range(num_classes))}")
    if len(unique_classes) != num_classes:
        print(f"⚠️  WARNING: Missing classes in training data! Expected {num_classes} classes, found {len(unique_classes)}")
    
    # Verify feature statistics
    print(f"\nFeature statistics:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_train mean: {X_train.mean():.6f}, std: {X_train.std():.6f}")
    print(f"  X_train min: {X_train.min():.6f}, max: {X_train.max():.6f}")
    print(f"  Any NaN values: {np.isnan(X_train).any()}")
    print(f"  Any Inf values: {np.isinf(X_train).any()}")
    
    print("\n3. Training Random Forest classifier...")
    print("   Using same methodology as wavelet training:")
    print("     - WaveletFeatureDataset (126 wavelet features)")
    print("     - ReducedClassDataset (4 classes, rest removed)")
    print("     - StandardScaler normalization")
    print("     - Explicit sample weights for severe class imbalance")
    print("\n   Random Forest Parameters (adjusted for severe class imbalance):")
    print("     - n_estimators: 1000 (increased for better minority class learning)")
    print("     - max_depth: 25 (increased to allow deeper trees for minority classes)")
    print("     - min_samples_split: 3 (decreased to allow more splits for minority classes)")
    print("     - min_samples_leaf: 1 (decreased to allow very small leaves for minority classes)")
    print("     - class_weight: custom aggressive weights (better than 'balanced')")
    print("     - max_features: 'sqrt'")
    print("     - criterion: 'gini' (default)")
    print("     - random_state: 42")
    
    # Configure Random Forest with parameters optimized for imbalanced data
    # Deeper trees and smaller leaf sizes help capture minority class patterns
    rf_model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1,
        max_features='sqrt',
        criterion='gini'
    )
    
    print("\n   Training...")
    rf_model.fit(X_train, y_train)
    print("   Training complete!\n")
    
    # Verify model predicts diverse classes (not just one class)
    y_train_pred = rf_model.predict(X_train)
    train_pred_counts = Counter(y_train_pred)
    print(f"Training predictions distribution: {dict(train_pred_counts)}")
    print(f"Training true distribution: {dict(train_counts)}")
    
    # Analyze feature importance to understand which features drive predictions
    print("6. Feature importance (top 10):")
    feature_importance = rf_model.feature_importances_
    
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    
    for idx in top_indices:
        print(f"   {feature_names[idx]:15s}: {feature_importance[idx]:.4f}")
    
    # Evaluate on test set
    print("\n7. Evaluating on test set...")
    y_pred = rf_model.predict(X_test)
    
    # Detect critical issue: single-class predictions usually indicate data quality problem
    unique_predictions = np.unique(y_pred)
    print(f"Unique predictions: {unique_predictions}")
    if len(unique_predictions) == 1:
        print(f"⚠️  CRITICAL WARNING: Model is predicting only one class: {unique_predictions[0]}")
        print(f"   This suggests a serious problem with the model or data!")
        print(f"   Checking prediction probabilities...")
        y_pred_proba = rf_model.predict_proba(X_test)
        print(f"   Prediction probability shape: {y_pred_proba.shape}")
        print(f"   Mean probabilities per class: {y_pred_proba.mean(axis=0)}")
        print(f"   Max probabilities per class: {y_pred_proba.max(axis=0)}")
    else:
        pred_counts = Counter(y_pred)
        print(f"Prediction distribution: {dict(pred_counts)}")
    
    test_acc = 100 * np.mean(y_pred == y_test)
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Classification report
    print("\n8. Per-Class Performance:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Random Forest ({num_classes} classes)')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_rf_{num_classes}classes.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved confusion matrix to 'confusion_matrix_rf_{num_classes}classes.png'")
    plt.show()
    
    # Compute per-class accuracy to identify which classes are poorly classified
    print("\n9. Per-Class Accuracy:")
    print("-" * 70)
    for class_idx in range(num_classes):
        mask = y_test == class_idx
        if np.sum(mask) > 0:
            class_acc = 100 * np.mean(y_pred[mask] == y_test[mask])
            class_count = np.sum(mask)
            correct = np.sum(y_pred[mask] == y_test[mask])
            print(f"  {class_names[class_idx]:15s}: {class_acc:6.2f}% ({correct}/{class_count})")
            
    
    # Cross-validation to assess model stability and generalization
    print("\n10. Cross-validation on training set (5-fold):")
    if rf_model is not None:
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = 100*np.mean(cv_scores)
        cv_std = 100*np.std(cv_scores)
        print(f"   CV Accuracy: {cv_mean:.2f}% ± {cv_std:.2f}%")
    else:
        print("   [Skipped - using hierarchical approach]")
        cv_mean = None
        cv_std = None
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Test Accuracy: {test_acc:.2f}%")
    if cv_mean is not None:
        print(f"CV Accuracy:   {cv_mean:.2f}% ± {cv_std:.2f}%")
    

if __name__ == "__main__":
    main()

