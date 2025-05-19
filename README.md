# 📈 Stock Market Movement Prediction Using Sentiment and Fundamental Analysis

## Overview

This project aims to predict stock price movements by integrating sentiment analysis with fundamental financial indicators.
Utilizing machine learning techniques, the model forecasts whether a stock's price will rise, fall, or remain stable.

## Data Source

The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/abhinavsinha845/cnbc-newsdata/data):
The dataset had news data from CNBC. Using another source, data was extracted from Guardian and Reuters.
The following data was generated:

- **Sentiment Data**: Using FINBert
- **Fundamental Data**:  Using ta library in python.

## Feature Engineering

- **Lag Features**: Created for each base feature up to 5 previous time steps.

## Dimensionality Reduction & Feature Selection

To optimize model performance and reduce overfitting:

- **Principal Component Analysis (PCA)**: Reduced feature space to 10 principal components.
- **Recursive Feature Elimination (RFE)**: Selected the top 10 features based on importance.

## Modeling Approach

Three classification models were evaluated:

1. **Logistic Regression**: With L1 regularization.
2. **Random Forest Classifier**: Ensemble method using decision trees.
3. **Gradient Boosting Classifier**: Sequential ensemble technique.

Each model underwent hyperparameter tuning using `GridSearchCV` with 5-fold cross-validation.

## Evaluation Metrics

Models were assessed using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

The best-performing model was identified based on the highest F1 Score.

## Results

The top-performing model was:

- **Model**: Gradient Boosting Classifier
- **Feature Variant**: PCA
- **F1 Score**: 0.65

The top 2 features contributing most to the model's predictions were:

1. `sentiment_score_finbert`
2. `sentiment_numeric_finbert`

## Conclusion

Integrating sentiment analysis with fundamental financial indicators, combined with advanced feature selection techniques, enhances the accuracy of stock movement predictions.

## Future Work
- Explore deep learning models for improved accuracy.
- Implement real-time prediction capabilities.
