import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                             f1_score, accuracy_score, log_loss)
from imblearn.over_sampling import SMOTE
from sklearn.base import clone

# ----------------------
# Data Preprocessing
# ----------------------
data_folder = r"D:\MachineLearningProject\data"
preprocessed_folder = r"D:\MachineLearningProject\preprocessed_data"
preprocessed_csv = os.path.join(preprocessed_folder, "processed_data.csv")

# Create preprocessed folder 
os.makedirs(preprocessed_folder, exist_ok=True)

def preprocess_data():
    processed_data = []
    
    # Enhanced labeling function for 14 classes
    def determine_label(file_path):
        file_path_lower = file_path.lower()
        
        # Healthy motor
        if "healthy" in file_path_lower:
            return "healthy"
        
        # Broken rotor bar faults (only 100W and 300W)
        elif "broken" in file_path_lower or "rotor" in file_path_lower or "brb" in file_path_lower:
            if "100w" in file_path_lower:
                return "broken_rotor_100W"
            elif "300w" in file_path_lower:
                return "broken_rotor_300W"
            else:
                # Default if load not specified
                return "broken_rotor_unknown"
        
        # Bearing faults
        elif "bearing" in file_path_lower:
            # Extract bearing size
            size = None
            for sz in ["0.7mm", "0.9mm", "1.1mm", "1.3mm", "1.5mm", "1.7mm"]:
                if sz in file_path_lower:
                    size = sz.replace(".", "")  # 07mm, 09mm, etc.
                    break
            
            # Extract race type (inner or outer)
            race_type = "outer"  # default
            if "inner" in file_path_lower:
                race_type = "inner"
            elif "outer" in file_path_lower:
                race_type = "outer"
            
            # Extract load
            load = None
            if "100w" in file_path_lower:
                load = "100W"
            elif "200w" in file_path_lower:
                load = "200W"
            elif "300w" in file_path_lower:
                load = "300W"
            
            # Construct label
            if size and load:
                return f"bearing_{race_type}_{size}_{load}"
            elif size:
                return f"bearing_{race_type}_{size}"
            else:
                return "bearing_fault_unknown"
        
        else:
            return "unknown"
    
    # Walk through files and process CSV files
    print("Starting data preprocessing...")
    file_count = 0
    
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    # Read CSV file
                    data = pd.read_csv(file_path, on_bad_lines="skip", low_memory=False)
                    
                    # Drop timestamp column if exists
                    data = data.drop(columns=["time_stamp"], errors="ignore")
                    
                    # Select only numeric columns
                    numeric_data = data.select_dtypes(include=[np.number])
                    
                    # Get label from file path
                    label = determine_label(file_path)
                    
                    # Process data in chunks of 1000 samples
                    for i in range(0, len(numeric_data), 1000):
                        chunk = numeric_data.iloc[i:i+1000]
                        
                        # Only process if we have exactly 1000 samples
                        if len(chunk) == 1000:
                            # Flatten the chunk: each row contains all 1000 samples from all phases
                            chunk_flat = chunk.values.flatten()
                            
                            # Create a dictionary for this row
                            row_dict = {f"feature_{j}": chunk_flat[j] for j in range(len(chunk_flat))}
                            row_dict["label"] = label
                            
                            processed_data.append(row_dict)
                    
                    file_count += 1
                    print(f"Processed file {file_count}: {file}")
                    
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
    
    if processed_data:
        final_data = pd.DataFrame(processed_data)
        final_data.to_csv(preprocessed_csv, index=False)
        print(f"\nPreprocessed data saved to {preprocessed_csv}")
        print(f"Total samples created: {len(final_data)}")
        print(f"Classes found: {final_data['label'].unique()}")
        print(f"Class distribution:\n{final_data['label'].value_counts()}")
    else:
        print("No valid data processed. Please check the dataset.")
        return

# Run preprocessing
preprocess_data()

# ----------------------
# Load and Prepare Data
# ----------------------
print("\n" + "="*60)
print("LOADING AND PREPARING DATA")
print("="*60)

data = pd.read_csv(preprocessed_csv)

# Handle infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill missing values with mean
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Encode the label column
label_encoder = LabelEncoder()
data["label_encoded"] = label_encoder.fit_transform(data["label"])

print(f"\nTotal samples: {len(data)}")
print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Classes: {label_encoder.classes_}")

# Features and Labels
X = data.drop(columns=["label", "label_encoded"])
y = data["label_encoded"]

# Split the data FIRST (70% train, 30% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Apply StandardScaler AFTER splitting (FIT on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ----------------------
# Apply SMOTE for Multiclass
# ----------------------
print("\nApplying SMOTE for class balancing...")
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_multi, y_train_multi = smote.fit_resample(X_train, y_train)

print(f"Training samples after SMOTE: {len(X_train_multi)}")

# ----------------------
# Binary Classification Preparation
# ----------------------
# Define binary: "healthy" = 0 and "unhealthy" = 1
healthy_label = label_encoder.transform(["healthy"])[0]
y_train_binary = np.where(y_train == healthy_label, 0, 1)
y_test_binary = np.where(y_test == healthy_label, 0, 1)

# Apply SMOTE for Binary Classification
X_train_binary, y_train_binary = smote.fit_resample(X_train, y_train_binary)

print(f"Binary training samples after SMOTE: {len(X_train_binary)}")
print(f"Binary class distribution: Healthy={np.sum(y_train_binary==0)}, Unhealthy={np.sum(y_train_binary==1)}")

# ----------------------
# Function to Compute Specificity
# ----------------------
def compute_specificity(cm):
    """Compute specificity for binary classification"""
    TN = cm[0, 0]
    FP = cm[0, 1]
    return TN / (TN + FP) if (TN + FP) > 0 else 0

# ----------------------
# Plotting Functions
# ----------------------
def plot_learning_curves(estimator, X, y, title_prefix="", train_sizes=None):
    """Plot training and validation accuracy curves"""
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, valid_scores = learning_curve(
        estimator, X, y, cv=5, scoring="accuracy", n_jobs=-1, 
        train_sizes=train_sizes, random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color="r", label="Training Accuracy")
    plt.plot(train_sizes_abs, valid_mean, 'o-', color="g", label="Validation Accuracy")
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes_abs, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1, color="g")
    plt.title(f"{title_prefix} - Accuracy Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_loss_curves(estimator, X_train, y_train, X_val, y_val, title_prefix="", train_sizes=None):
    """Plot training and validation loss curves"""
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_losses = []
    val_losses = []
    n_samples = X_train.shape[0]
    
    for frac in train_sizes:
        size = int(n_samples * frac)
        if size < 10:  # Minimum samples
            continue
            
        estimator_clone = clone(estimator)
        estimator_clone.fit(X_train[:size], y_train[:size])
        
        # Compute log loss
        train_proba = estimator_clone.predict_proba(X_train[:size])
        val_proba = estimator_clone.predict_proba(X_val)
        
        train_loss = log_loss(y_train[:size], train_proba)
        val_loss = log_loss(y_val, val_proba)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    actual_sizes = [int(n_samples * frac) for frac in train_sizes if int(n_samples * frac) >= 10]
    
    plt.figure(figsize=(10, 6))
    plt.plot(actual_sizes, train_losses, 'o-', color="r", label="Training Loss")
    plt.plot(actual_sizes, val_losses, 'o-', color="g", label="Validation Loss")
    plt.title(f"{title_prefix} - Loss Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Log Loss")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =====================================================================
# QUESTION 1: BINARY CLASSIFICATION
# =====================================================================
print("\n" + "="*60)
print("QUESTION 1: BINARY CLASSIFICATION (Healthy vs Unhealthy)")
print("="*60)

# ----------------------
# Gaussian Naïve Bayes (Binary)
# ----------------------
print("\n--- Gaussian Naïve Bayes (Binary) ---")
gnb = GaussianNB()
gnb.fit(X_train_binary, y_train_binary)
gnb_preds = gnb.predict(X_test)
gnb_probs = gnb.predict_proba(X_test)

cm_gnb = confusion_matrix(y_test_binary, gnb_preds)
gnb_accuracy = accuracy_score(y_test_binary, gnb_preds)
gnb_precision = precision_score(y_test_binary, gnb_preds, zero_division=0)
gnb_recall = recall_score(y_test_binary, gnb_preds, zero_division=0)
gnb_f1 = f1_score(y_test_binary, gnb_preds, zero_division=0)
gnb_specificity = compute_specificity(cm_gnb)
gnb_log_loss = log_loss(y_test_binary, gnb_probs, labels=[0, 1])

print("\n=== Binary Classification - Gaussian Naïve Bayes ===")
print(f"Accuracy:    {gnb_accuracy:.4f}")
print(f"Precision:   {gnb_precision:.4f}")
print(f"Recall:      {gnb_recall:.4f}")
print(f"Sensitivity: {gnb_recall:.4f}")
print(f"Specificity: {gnb_specificity:.4f}")
print(f"F1 Score:    {gnb_f1:.4f}")
print(f"Log Loss:    {gnb_log_loss:.4f}")
print("\nConfusion Matrix:")
print(cm_gnb)

# ----------------------
# SVM (Binary)
# ----------------------
print("\n--- Support Vector Machine (Binary) ---")
svm = SVC(probability=True, random_state=42, kernel='rbf')
svm.fit(X_train_binary, y_train_binary)
svm_preds = svm.predict(X_test)
svm_probs = svm.predict_proba(X_test)

cm_svm = confusion_matrix(y_test_binary, svm_preds)
svm_accuracy = accuracy_score(y_test_binary, svm_preds)
svm_precision = precision_score(y_test_binary, svm_preds, zero_division=0)
svm_recall = recall_score(y_test_binary, svm_preds, zero_division=0)
svm_f1 = f1_score(y_test_binary, svm_preds, zero_division=0)
svm_specificity = compute_specificity(cm_svm)
svm_log_loss = log_loss(y_test_binary, svm_probs, labels=[0, 1])

print("\n=== Binary Classification - SVM ===")
print(f"Accuracy:    {svm_accuracy:.4f}")
print(f"Precision:   {svm_precision:.4f}")
print(f"Recall:      {svm_recall:.4f}")
print(f"Sensitivity: {svm_recall:.4f}")
print(f"Specificity: {svm_specificity:.4f}")
print(f"F1 Score:    {svm_f1:.4f}")
print(f"Log Loss:    {svm_log_loss:.4f}")
print("\nConfusion Matrix:")
print(cm_svm)

# =====================================================================
# QUESTION 2: MULTICLASS CLASSIFICATION (14 Classes)
# =====================================================================
print("\n" + "="*60)
print("QUESTION 2: MULTICLASS CLASSIFICATION (14 Classes)")
print("="*60)

# ----------------------
# Gaussian Naïve Bayes (Multiclass)
# ----------------------
print("\n--- Gaussian Naïve Bayes (Multiclass) ---")
gnb_multi = GaussianNB()
gnb_multi.fit(X_train_multi, y_train_multi)
gnb_multi_preds = gnb_multi.predict(X_test)
gnb_multi_probs = gnb_multi.predict_proba(X_test)

cm_gnb_multi = confusion_matrix(y_test, gnb_multi_preds)
gnb_multi_accuracy = accuracy_score(y_test, gnb_multi_preds)
gnb_multi_precision = precision_score(y_test, gnb_multi_preds, average="weighted", zero_division=0)
gnb_multi_recall = recall_score(y_test, gnb_multi_preds, average="weighted", zero_division=0)
gnb_multi_f1 = f1_score(y_test, gnb_multi_preds, average="weighted", zero_division=0)
gnb_multi_log_loss = log_loss(y_test, gnb_multi_probs)

print("\n=== Multiclass Classification - Gaussian Naïve Bayes ===")
print(f"Accuracy:    {gnb_multi_accuracy:.4f}")
print(f"Precision:   {gnb_multi_precision:.4f}")
print(f"Recall:      {gnb_multi_recall:.4f}")
print(f"Sensitivity: {gnb_multi_recall:.4f}")
print(f"F1 Score:    {gnb_multi_f1:.4f}")
print(f"Log Loss:    {gnb_multi_log_loss:.4f}")
print("\nConfusion Matrix:")
print(cm_gnb_multi)

# ----------------------
# SVM (Multiclass)
# ----------------------
print("\n--- Support Vector Machine (Multiclass) ---")
svm_multi = SVC(probability=True, random_state=42, kernel='rbf')
svm_multi.fit(X_train_multi, y_train_multi)
svm_multi_preds = svm_multi.predict(X_test)
svm_multi_probs = svm_multi.predict_proba(X_test)

cm_svm_multi = confusion_matrix(y_test, svm_multi_preds)
svm_multi_accuracy = accuracy_score(y_test, svm_multi_preds)
svm_multi_precision = precision_score(y_test, svm_multi_preds, average="weighted", zero_division=0)
svm_multi_recall = recall_score(y_test, svm_multi_preds, average="weighted", zero_division=0)
svm_multi_f1 = f1_score(y_test, svm_multi_preds, average="weighted", zero_division=0)
svm_multi_log_loss = log_loss(y_test, svm_multi_probs)

print("\n=== Multiclass Classification - SVM ===")
print(f"Accuracy:    {svm_multi_accuracy:.4f}")
print(f"Precision:   {svm_multi_precision:.4f}")
print(f"Recall:      {svm_multi_recall:.4f}")
print(f"Sensitivity: {svm_multi_recall:.4f}")
print(f"F1 Score:    {svm_multi_f1:.4f}")
print(f"Log Loss:    {svm_multi_log_loss:.4f}")
print("\nConfusion Matrix:")
print(cm_svm_multi)

# =====================================================================
# QUESTION 3: PLOT LEARNING CURVES
# =====================================================================
print("\n" + "="*60)
print("QUESTION 3: PLOTTING LEARNING CURVES")
print("="*60)

# Create validation split for loss curves
X_tr_bin, X_val_bin, y_tr_bin, y_val_bin = train_test_split(
    X_train_binary, y_train_binary, test_size=0.2, random_state=42, stratify=y_train_binary
)

X_tr_multi, X_val_multi, y_tr_multi, y_val_multi = train_test_split(
    X_train_multi, y_train_multi, test_size=0.2, random_state=42, stratify=y_train_multi
)

# ----------------------
# Binary Classification - Gaussian Naïve Bayes
# ----------------------
print("\n--- Gaussian Naïve Bayes (Binary) Learning Curves ---")
plot_learning_curves(
    GaussianNB(), 
    X_train_binary, 
    y_train_binary, 
    title_prefix="GaussianNB Binary"
)

plot_loss_curves(
    GaussianNB(), 
    X_tr_bin, 
    y_tr_bin, 
    X_val_bin, 
    y_val_bin, 
    title_prefix="GaussianNB Binary"
)

# ----------------------
# Binary Classification - SVM
# ----------------------
print("\n--- SVM (Binary) Learning Curves ---")
plot_learning_curves(
    SVC(probability=True, random_state=42), 
    X_train_binary, 
    y_train_binary, 
    title_prefix="SVM Binary",
    train_sizes=np.linspace(0.3, 1.0, 8)
)

plot_loss_curves(
    SVC(probability=True, random_state=42), 
    X_tr_bin, 
    y_tr_bin, 
    X_val_bin, 
    y_val_bin, 
    title_prefix="SVM Binary",
    train_sizes=np.linspace(0.3, 1.0, 8)
)

# ----------------------
# Multiclass Classification - Gaussian Naïve Bayes
# ----------------------
print("\n--- Gaussian Naïve Bayes (Multiclass) Learning Curves ---")
plot_learning_curves(
    GaussianNB(), 
    X_train_multi, 
    y_train_multi, 
    title_prefix="GaussianNB Multiclass"
)

plot_loss_curves(
    GaussianNB(), 
    X_tr_multi, 
    y_tr_multi, 
    X_val_multi, 
    y_val_multi, 
    title_prefix="GaussianNB Multiclass"
)

# ----------------------
# Multiclass Classification - SVM
# ----------------------
print("\n--- SVM (Multiclass) Learning Curves ---")
plot_learning_curves(
    SVC(probability=True, random_state=42), 
    X_train_multi, 
    y_train_multi, 
    title_prefix="SVM Multiclass",
    train_sizes=np.linspace(0.3, 1.0, 8)
)

plot_loss_curves(
    SVC(probability=True, random_state=42), 
    X_tr_multi, 
    y_tr_multi, 
    X_val_multi, 
    y_val_multi, 
    title_prefix="SVM Multiclass",
    train_sizes=np.linspace(0.3, 1.0, 8)
)

print("\n" + "="*60)
print("ASSIGNMENT COMPLETE")
print("="*60)
