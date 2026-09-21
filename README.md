# Breast Cancer Wisconsin Diagnostic — Logistic Regression

A machine learning project using the **Breast Cancer Wisconsin (Diagnostic)** dataset to classify breast masses as **benign** or **malignant**.

## Project Overview

This is a **binary classification** project using **Logistic Regression**. The dataset contains 569 observations and 30 numerical features describing characteristics of cell nuclei extracted from digitized fine needle aspirate (FNA) images.

The project demonstrates an end-to-end machine learning workflow, from data exploration and preprocessing to model training and evaluation.

## Objectives

The main objectives of this project are to:

* Understand the structure and characteristics of the dataset.
* Clean and prepare the data for machine learning.
* Explore the distribution of the target variable.
* Examine relationships between numerical features.
* Split the dataset into training and testing sets.
* Apply feature standardization without data leakage.
* Train a Logistic Regression classification model.
* Evaluate the model using multiple classification metrics.
* Interpret the model coefficients.
* Document the complete workflow for reproducibility.

## Dataset

The **Breast Cancer Wisconsin (Diagnostic)** dataset contains measurements computed from digitized images of fine needle aspirates of breast masses.

Each observation contains characteristics of cell nuclei present in the image.

### Dataset characteristics

| Description                   |     Value |
| ----------------------------- | --------: |
| Total observations            |       569 |
| Numerical predictor variables |        30 |
| Benign cases                  |       357 |
| Malignant cases               |       212 |
| Target variable               | Diagnosis |

### Target Variable

The original `diagnosis` variable contains:

* `B` — Benign
* `M` — Malignant

For modeling, the target was encoded as:

* `0` — Benign
* `1` — Malignant

## Features

The dataset contains 10 measurements of cell nuclei:

1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave points
9. Symmetry
10. Fractal dimension

For each measurement, three statistical summaries are provided:

* Mean
* Standard Error (`SE`)
* Worst value

This produces a total of:

**10 measurements × 3 summaries = 30 numerical features**

Examples include:

* `radius_mean`
* `radius_se`
* `radius_worst`
* `texture_mean`
* `texture_se`
* `texture_worst`

## Machine Learning Approach

### 1. Data Loading

The dataset is loaded using Pandas.

### 2. Data Cleaning

The following columns are removed when present:

* `id` — observation identifier
* `Unnamed: 32` — empty CSV artifact

The target variable is then converted from categorical labels into binary numerical values.

### 3. Exploratory Data Analysis

The analysis includes:

* Dataset structure
* Descriptive statistics
* Missing-value inspection
* Diagnosis distribution
* Feature correlation analysis

### 4. Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

A stratified split is used so that the proportion of benign and malignant cases remains approximately consistent between the training and testing sets.

### 5. Feature Scaling

The numerical predictors are standardized using `StandardScaler`.

The scaler is fitted **only on the training data** and subsequently applied to the test data.

This prevents information from the test set from influencing the preprocessing stage.

### 6. Model

The primary model used in this project is:

**Logistic Regression**

Logistic Regression is appropriate for this task because the target contains two classes: benign and malignant.

## Model Evaluation

The model is evaluated using several metrics.

### Accuracy

Measures the proportion of all predictions that are correct.

### Precision

Measures the proportion of observations predicted as malignant that are actually malignant.

### Recall

Measures the proportion of actual malignant observations that the model correctly identifies.

### F1-Score

Combines precision and recall into a single metric.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

### Confusion Matrix

The confusion matrix provides a breakdown of:

* True Positives
* True Negatives
* False Positives
* False Negatives

For this application, **malignant-class recall is particularly important** because incorrectly classifying a malignant case as benign represents a false negative.

## Results

The notebook generates the following evaluation outputs:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Classification report
* Confusion matrix
* ROC curve
* Logistic Regression coefficients

> The numerical results should be updated here after the notebook has been executed so that the README reflects the actual model run.

## Model Interpretation

The Logistic Regression coefficients provide information about the relationship between standardized features and the model's predicted log-odds of the malignant class.

Features with larger absolute coefficients have a stronger contribution within the fitted model.

Because several measurements describe related characteristics of the same cell nuclei, correlations between predictors should be considered when interpreting individual coefficients.

## Project Structure

```text
breast-cancer-wisconsin-logistic-regression/
│
├── breast_cancer_logistic_regression.ipynb
│
├── data/
│   └── data.csv
│
├── README.md
│
└── requirements.txt
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/breast-cancer-wisconsin-logistic-regression.git
```

Navigate into the project:

```bash
cd breast-cancer-wisconsin-logistic-regression
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
breast_cancer_logistic_regression.ipynb
```

## Reproducibility

The project uses `random_state=42` for reproducible train-test splitting and model initialization.

The preprocessing workflow also prevents data leakage by fitting the scaler exclusively on the training dataset.

## Limitations

This project is an educational machine learning exercise.

The model should **not** be interpreted as a clinical diagnostic system or used for medical decision-making.

The dataset is relatively small, and performance on this dataset may not generalize to other populations or clinical environments.

Feature correlations may also affect the interpretation of individual Logistic Regression coefficients.

## Future Improvements

Possible extensions include:

* Compare Logistic Regression with Random Forest.
* Compare performance with Support Vector Machines.
* Test K-Nearest Neighbors.
* Perform cross-validation.
* Tune model hyperparameters.
* Apply feature selection.
* Explore Principal Component Analysis (PCA).
* Build a complete Scikit-learn Pipeline.
* Compare ROC-AUC with precision-recall curves.
* Perform more detailed error analysis.

## Learning Outcomes

Through this project, I practiced:

* Data cleaning with Pandas
* Exploratory data analysis
* Feature-target separation
* Train-test splitting
* Stratified sampling
* Feature standardization
* Logistic Regression
* Classification evaluation
* Confusion matrix interpretation
* ROC-AUC analysis
* Model coefficient interpretation
* Machine learning reproducibility

## Author

**Malik**

Economics Graduate | Data Analyst | Aspiring Data Scientist

---

### Dataset Source

Breast Cancer Wisconsin (Diagnostic) dataset.

The dataset was originally provided through the University of Wisconsin and is also available through the **UCI Machine Learning Repository**.
