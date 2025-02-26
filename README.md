# UFC Fight Prediction Using Multiple Models

This repository contains a machine learning project aimed at predicting UFC fight outcomes using various machine learning algorithms. The goal is to compare the performance of multiple models and find the best-performing model based on fight data.

## Project Overview

This project explores several machine learning models to predict the outcome of UFC fights. The approach includes the following steps:

- **Data Preprocessing**: Cleaning, feature engineering, and preparing UFC fight data for model training.
- **Model Training**: Applying various machine learning models and training them on historical fight data.
- **Model Evaluation**: Using various evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC to compare the models.
- **Model Tuning**: Hyperparameter tuning using GridSearchCV to optimize model performance.
- **Model Comparison**: Comparing the performance of different models and selecting the best one for UFC fight predictions.

## Algorithms Used

The following machine learning models are implemented and evaluated in this project:

- Logistic Regression
- Random Forest
- XGBoost
- LightBGM
- CatBoost
- Support Vector Machine (SVM)
- Multi-Layer Perceptron
- Voting Classifier
- Ensemble Model (LightBGM, CatBoost, Logistic Regression and XGBoost)

## Features

- **Data Exploration**: Initial analysis of the fight data, including visualizations and feature importance.
- **Model Training**: Train various ML models on the UFC dataset and evaluate their performance.
- **Model Comparison**: Visualization of results and comparison between different machine learning models.
- **Prediction**: Use the best-performing model to predict the outcomes of future UFC fights.

## Requirements

To run this project locally, you need to install the required Python packages. You can do this by installing the dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

After cloning the repository and installing the dependencies, you can follow these steps to run the project:

- Preprocess the Data: Run the `UFC_Preprocessing.ipynb` and `UFC_Preprocessing2.ipynb` notebook to clean and preprocess the UFC fight dataset.
- Train and Evaluate Models: Use the `UFC_Prediction_Models.ipynb` notebook to train the machine learning models and evaluate their performance.

## What I Learned

### Fundamentals of Machine Learning & Data Analytics
- **Data Collection & Preprocessing:**  
  I learned how to gather and clean UFC fight data, ensuring consistency and quality for effective model training.

- **Exploratory Data Analysis (EDA):**  
  I developed the skills to perform EDA, uncovering trends and insights in the data that guided my feature selection and model design.

### Model Building and Evaluation
- **Multiple Machine Learning Models:**  
  I experimented with various algorithms—such as logistic regression, decision trees, random forests, and ensemble methods—to predict fight outcomes. This allowed me to compare different approaches and determine which performed best.

- **Feature Engineering:**  
  I gained experience in crafting features from raw data, such as fighter statistics and historical performance, which significantly improved model accuracy.

- **Model Evaluation Metrics:**  
  I utilized metrics like accuracy, precision, recall, and ROC-AUC to evaluate and compare the performance of the models, ensuring that the chosen models were both robust and reliable.

### Practical Implementation Skills
- **Python Programming:**  
  This project enhanced my proficiency in Python, especially using libraries like pandas for data manipulation, scikit-learn for model building, and matplotlib/seaborn for visualization.

- **Automation and Scripting:**  
  I developed scripts to automate tasks such as data preprocessing, model training, and evaluation, which streamlined the entire experimentation process.

### Experimentation and Optimization
- **Hyperparameter Tuning:**  
  I experimented with different hyperparameters to optimize the models, learning how subtle changes can greatly affect performance.

- **Cross-Validation Techniques:**  
  I implemented cross-validation to ensure that the models were generalizable and could perform well on unseen data.

### Insights into Sports Analytics
- **Understanding Fight Dynamics:**  
  By analyzing various factors influencing UFC fights, I gained deeper insights into the sport’s dynamics, which further informed my model design.

- **Real-World Application of Predictive Analytics:**  
  This project showcased how predictive models can be applied to real-world sports scenarios, potentially informing strategies and decisions in the sports analytics arena.
