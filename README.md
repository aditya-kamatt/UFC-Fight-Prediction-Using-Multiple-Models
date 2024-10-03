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

## Results
The results of the project will showcase how each machine learning model performs in predicting UFC fight outcomes. The goal is to identify the most accurate and reliable model by comparing various metrics.

