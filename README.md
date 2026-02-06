## Motor Fault Diagnosis using Machine Learning 🏎️🛠️

## 📌 Project Overview
    This project focuses on identifying and classifying faults in industrial motors using signal data. By leveraging Machine Learning algorithms, we can predict whether a motor is healthy or unhealthy and pinpoint the specific type of fault (e.g., Broken Rotor Bar, Inner Race Fault, Outer Race Fault) with high precision. This is a critical component of Predictive Maintenance in Industry 4.0.

## Key Objectives:
## Binary Classification: 
    Distinguish between Healthy and Unhealthy motor states.
## Multiclass Classification: 
    Classify 14 different fault categories across various severity levels (0.7mm to 1.7mm).
## Performance Comparison:     
    Evaluate the effectiveness of SVM, Gaussian Naive Bayes, and Logistic Regression.

## 📊 Dataset Description:
    The dataset consists of motor vibration/signal data:
## Samples: 
    52,052
## Features: 1,000 signal points per sample.
## Classes: 14 (1 Healthy + 13 Fault types including Broken Rotor Bar and Bearing faults).

## 🛠️ Tech Stack & Methods:
    Language: Python
    Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Imbalanced-learn (SMOTE).
    Preprocessing: * StandardScaler for feature normalization.
    LabelEncoder for categorical targets.
    PCA (Principal Component Analysis) for dimensionality reduction.
    SMOTE for handling class imbalance in multiclass data.
    
## Results & Visualizations
    1. Binary Classification (Healthy vs. Unhealthy)
    Model        Accuracy  Precision  Recall  F1-Score
    SVM (Linear)  99.08%    0.99      1.00    0.99
    Naive Bayes   61.14%    1.00      0.61    0.752. 
    
    2. Multiclass Classification (14 Faults)
    The SVM model outperformed others in identifying specific fault locations, while Naive Bayes provided a faster baseline.
    
    3. Learning CurvesWe monitored the training vs. validation loss to ensure the models were not overfitting.
