# 🔐 Phishing Website Detection Using Machine Learning

<p align="center">
  <b>Machine Learning-Based Phishing and Legitimate Website Classification</b>
</p>

<p align="center">
  A Machine Learning project that analyzes website and URL characteristics to identify potential phishing websites and distinguish them from legitimate websites.
</p>

---

## 📌 Project Overview

This project implements a Machine Learning-based approach to detect phishing websites using website and URL-related features.

The system classifies websites into two categories:

- **Legitimate (0):** Safe or legitimate websites.
- **Phishing (1):** Websites identified as phishing.

The project focuses on analyzing website characteristics, performing data preprocessing, exploring feature relationships, and developing classification models using Logistic Regression.

Two feature selection approaches are also explored: Correlation-based Feature Selection and Random Forest Feature Importance to identify relevant features and evaluate their impact on classification performance.

## 🎯 Project Objectives

- Develop a Machine Learning model for phishing website detection.
- Analyze characteristics associated with phishing and legitimate websites.
- Perform data cleaning and preprocessing.
- Identify important features using correlation analysis.
- Apply Random Forest Feature Importance for feature selection.
- Compare classification performance using different feature configurations.
- Evaluate models using classification metrics and confusion matrices.

## ✨ Key Contributions

### 1. Exploratory Data Analysis

Analyze dataset characteristics, including:
- Dataset structure and statistical summaries.
- Missing values and duplicate records.
- Phishing and legitimate class distribution.
- Correlation between numerical features.
- Identification of important website characteristics.

### 2. Feature Selection

Explore two feature selection approaches:

**Correlation-Based Feature Selection**
- Calculate feature correlations with the target variable.
- Rank features based on absolute correlation values.
- Select the top 20 features.

**Random Forest Feature Importance**
- Train a Random Forest classifier to evaluate feature importance.
- Rank features based on their importance scores.
- Select the top 20 most influential features.

### 3. Logistic Regression Classification

Use Logistic Regression as the primary classification algorithm.

The model is evaluated using three configurations:
- Logistic Regression with all numerical features.
- Logistic Regression with the top 20 correlation-based features.
- Logistic Regression with the top 20 Random Forest-selected features.

RobustScaler is applied to scale numerical features before model training.

## 📊 Dataset Information

| Information | Description |
|---|---|
| Dataset | Phishing Website Dataset |
| File | `dataset_phishing.csv` |
| Total Samples | 11,430 |
| Total Columns | 89 |
| Numerical Features | 87 |
| Number of Classes | 2 |
| Legitimate Samples | 5,715 (50%) |
| Phishing Samples | 5,715 (50%) |
| Target Variable | `status` |
| Legitimate Label | 0 |
| Phishing Label | 1 |

### Class Distribution

The dataset contains an equal distribution of legitimate and phishing samples, with each class representing 50% of the total observations.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine Learning and evaluation |
| Logistic Regression | Classification algorithm |
| Random Forest | Feature importance |
| RobustScaler | Feature scaling |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook | Experimentation and analysis |

## 🔄 Project Workflow

The project consists of seven main stages:

**1. Data Loading**
- Load the phishing website dataset.
- Inspect dataset structure and dimensions.
- Identify numerical and categorical columns.

**2. Exploratory Data Analysis**
- Analyze class distribution.
- Check missing values and duplicate records.
- Examine descriptive statistics.
- Explore feature correlations.

**3. Data Preprocessing**
- Convert target labels into numerical values.
- Identify invalid negative values in domain-related features.
- Replace invalid values using median imputation.
- Separate features and target variables.

**4. Train-Test Splitting**
- Split data into 80% training and 20% testing.
- Apply stratified sampling to maintain class distribution.

**5. Feature Selection**
- Apply correlation-based feature selection.
- Apply Random Forest Feature Importance.
- Select the top 20 features for each approach.

**6. Model Training**
- Logistic Regression using all numerical features.
- Logistic Regression using correlation-selected features.
- Logistic Regression using Random Forest-selected features.

**7. Model Evaluation**
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 🧠 Model Architecture

### Classification Workflow

```text
       Phishing Website Dataset
                  |
                  v
       Exploratory Data Analysis
                  |
                  v
         Data Preprocessing
                  |
                  v
          Train-Test Split
                  |
                  v
         Feature Selection
          /             \
         v               v
    Correlation      Random Forest
     Analysis        Importance
         |               |
         v               v
     Top 20           Top 20
     Features         Features
         |               |
         v               v
    Logistic          Logistic
    Regression        Regression
         \               /
          \             /
           v           v
          Model Evaluation
                  |
                  v
       Legitimate / Phishing
```

## 📈 Model Performance Comparison

Three classification configurations were evaluated.

| Model Configuration | Accuracy |
|---|---:|
| Logistic Regression (All Features) | **93.35%** |
| Logistic Regression (Correlation Top 20) | 92.65% |
| Logistic Regression (Random Forest Top 20) | 91.82% |

The Logistic Regression model using all numerical features achieved the highest accuracy of 93.35% among the three evaluated configurations.

### Classification Metrics

The following results were obtained from the Logistic Regression model using all numerical features:

| Metric | Score |
|---|---:|
| Accuracy | 93.35% |
| Precision | 93% |
| Recall | 93% |
| F1-Score | 93% |

*Note: Precision, Recall, and F1-Score are rounded values from the classification report.*

## 📉 Confusion Matrix

The Logistic Regression model using all numerical features produced the following results:

| Actual / Predicted | Legitimate | Phishing |
|---|---:|---:|
| Legitimate | 1,040 | 103 |
| Phishing | 65 | 1,078 |

The model correctly classified 2,118 out of 2,286 testing samples.

For comparison, the Random Forest-based feature selection experiment produced:

| Actual / Predicted | Legitimate | Phishing |
|---|---:|---:|
| Legitimate | 1,035 | 108 |
| Phishing | 79 | 1,064 |

## 📊 Visualizations

The project includes visualizations to support the analysis and evaluation process:

1. Phishing and Legitimate Class Distribution.
2. Top 20 Features Based on Correlation.
3. Random Forest Feature Importance.
4. Feature Correlation Heatmap.
5. Logistic Regression Confusion Matrix.
6. Confusion Matrix with Selected Features.

These visualizations help illustrate dataset characteristics, feature relevance, and model classification results.

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/ilhamnurramdani/REPOSITORY-NAME.git
```

### 2. Navigate to Project Directory

```bash
cd REPOSITORY-NAME
```

### 3. Create Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### 5. Prepare Dataset

Place the dataset in the project directory:

```text
Project/
├── app.ipynb
├── dataset_phishing.csv
└── README.md
```

### 6. Run the Project

```bash
jupyter notebook app.ipynb
```

Run the notebook cells sequentially to perform data exploration, preprocessing, feature selection, model training, and evaluation.

## 🔮 Future Improvements

- Experiment with additional classification algorithms such as Random Forest, XGBoost, and LightGBM.
- Explore advanced feature selection techniques.
- Apply hyperparameter optimization.
- Evaluate model generalization using external datasets.
- Develop a web-based phishing detection application.
- Improve model interpretability using SHAP.

## 👨‍💻 Author

**Ilham Nur Ramdani**

Master of Computer Science Student  
Universitas Budi Luhur

**Areas of Interest:**
- Artificial Intelligence
- Machine Learning
- Data Mining
- Cybersecurity
- Phishing Detection
- Data Analytics

GitHub: [@ilhamnurramdani](https://github.com/ilhamnurramdani)

---

<p align="center">
  <b>Developed with Python & Machine Learning</b>
  <br>
  ⭐ Feel free to explore this project and give it a star!
</p>
