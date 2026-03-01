# 🎗️ Breast Cancer Classification using CatBoost

This project builds a machine learning classification model to predict whether a breast tumor is Malignant or Benign using the CatBoost algorithm. The workflow includes data cleaning, exploratory data analysis (EDA), feature engineering, model training, evaluation, and model explainability using SHAP.

## 📌 Project Overview
The dataset was loaded and cleaned by removing unnecessary identifier columns and converting the target variable (diagnosis) into binary values (Malignant = 1, Benign = 0). The project includes descriptive statistics, correlation analysis, outlier detection, feature engineering, model training, cross-validation, ROC analysis, and SHAP interpretation.

## 🧹 Data Cleaning & Preparation
The ID column was removed as it does not contribute to prediction. The diagnosis column was mapped to binary values. The dataset was checked for missing values and duplicates to ensure data quality. No duplicated rows were found. Feature scaling was applied using StandardScaler where appropriate.

## 📊 Exploratory Data Analysis (EDA)
The distribution of diagnosis classes was visualized using count plots and percentage pie charts to confirm class balance. A full correlation matrix was generated to identify relationships between features. The top features most correlated with the target were extracted. Boxplots were used to visually inspect outliers and feature distributions.

## 🧠 Feature Engineering
New features were created to enhance predictive power, such as multiplying radius_mean and texture_mean, and dividing area_mean by perimeter_mean. These engineered features aim to capture more complex relationships between tumor measurements.

## 🤖 Model Training
The dataset was split into training and testing sets with stratification to maintain class balance. A CatBoostClassifier model was trained with the following parameters:
- iterations = 500
- learning_rate = 0.05
- depth = 6
- loss_function = Logloss
- eval_metric = Accuracy

## 📈 Model Evaluation
The model was evaluated using:
- Accuracy Score (≈ 96%)
- Precision, Recall, and F1-Score
- Confusion Matrix visualization
- 5-Fold Cross-Validation (mean accuracy calculated)
- ROC Curve and AUC score

The model achieved strong classification performance with high precision and recall for both classes.

## 🔍 Feature Importance
Feature importance was extracted from CatBoost and visualized. The most influential feature was identified and its distribution across diagnosis classes was plotted to understand its impact.

## 📊 Pairplot Visualization
A pairplot of the top correlated features was generated to visualize class separability and relationships between the most important tumor characteristics.

## 🧪 Cross-Validation
5-fold cross-validation was applied to ensure model stability and generalization performance. The average accuracy across folds was calculated.

## 📉 ROC Curve & AUC
The ROC curve was plotted to evaluate the model’s ability to distinguish between Malignant and Benign cases. The Area Under the Curve (AUC) was calculated as a measure of classification strength.

## 🔎 SHAP Model Explainability
SHAP (SHapley Additive exPlanations) was used to interpret the model’s predictions. A SHAP summary plot was generated to show which features had the strongest positive and negative impact on predictions.

## 🛠️ Technologies Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, CatBoost, SHAP.

## 🚀 How to Run
1. Install required libraries: pip install catboost shap
2. Load the breast-cancer.csv dataset.
3. Run the notebook step by step.
4. Train the model and evaluate results.

## 🎯 Skills Demonstrated
Data Cleaning, Exploratory Data Analysis, Feature Engineering, Classification Modeling, Cross-Validation, Model Evaluation Metrics, ROC Analysis, Feature Importance Analysis, and Explainable AI using SHAP.
