# Lending Club Loan Default Analysis

This project analyzes Lending Club loan data to predict the likelihood of default using classification models. It addresses class imbalance using oversampling and evaluates model performance using AUC, accuracy, and kappa.

## Objectives

- Handle missing data and perform imputations
- Encode categorical variables numerically
- Deal with class imbalance through oversampling
- Train and evaluate models: Random Forest and XGBoost
- Visualize model performance and feature importance

## Methods

- Random Forest (via `ranger`)
- XGBoost (via `xgboost`)
- Performance metrics: Confusion Matrix, AUC, Kappa
- ROC and variable importance plots

## Tools

- R, tidyverse, caret, ranger, xgboost, pROC, ggplot2

## Repository Structure

- `scripts/`: Main R script with preprocessing, modeling, and evaluation
- `data/`: Raw or sample Lending Club data (schema/sample if full dataset is proprietary)
- `figures/`: Visual outputs (ROC curve, feature importance, etc.)
- `requirements.txt`: R packages used
- `README.md`: Project overview and instructions

## License

MIT License
