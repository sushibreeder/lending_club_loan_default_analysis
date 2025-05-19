
# Load necessary libraries
library(tidyverse)
library(caret)    # For data splitting and evaluation
library(ranger)   # For Random Forest
library(xgboost)  # For XGBoost
library(pROC)     # For AUC calculation
library(reshape2) # For data visualization
library(ggplot2)  # For plotting

# Step 1: Load the dataset
link <- "C:/Users/sushm/Downloads/LoanStats3a.csv"
data <- read.csv(link, header = TRUE, stringsAsFactors = FALSE)
df <- as_tibble(data)

# Step 2: Handle missing values
threshold <- 0.8
dfM <- df[, colMeans(is.na(df)) <= threshold]  # Remove columns with >80% missing values

# Imputation functions
modeImpute <- function(column) {
  tbl <- table(column, useNA = "no")
  mode_val <- names(which.max(tbl))
  column[is.na(column)] <- mode_val
  return(column)
}
meanImpute <- function(column) {
  mean_val <- mean(column, na.rm = TRUE)
  column[is.na(column)] <- mean_val
  return(column)
}

# Impute missing values
mean_cols <- c("mths_since_last_delinq", "pub_rec_bankruptcies", "chargeoff_within_12_mths", "tax_liens")
mode_cols <- c("delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "total_acc", "acc_now_delinq", "delinq_amnt")
dfM[mean_cols] <- lapply(dfM[mean_cols], meanImpute)
dfM[mode_cols] <- lapply(dfM[mode_cols], modeImpute)

# # DONT RUN IT TWICE
# # Define the rows to delete by their indices
# rows_to_delete <- c(39787, 39788, 42452, 42453, 42483, 42536)
# 
# # Remove these rows from the dataset
# dfM <- dfM[-rows_to_delete, ]

# Mode imputation for 'title'
tbl <- table(dfM$title, useNA = "no")  # Frequency table excluding NA
mode_title <- names(which.max(tbl))   # Find the mode
dfM$title[is.na(dfM$title)] <- mode_title  # Replace NA with mode

# Median imputation for 'collections_12_mths_ex_med'
median_val <- median(dfM$collections_12_mths_ex_med, na.rm = TRUE)
dfM$collections_12_mths_ex_med[is.na(dfM$collections_12_mths_ex_med)] <- median_val
# we are done with imputation
sum(is.na(dfM))

# Step 3: Encode `loan_status` as a binary variable
dfM$loan_status <- ifelse(dfM$loan_status == "Fully Paid", 1, 0)

# Step 4: Remove zero-variance columns
numeric_data <- dfM %>% select_if(is.numeric)
zero_var_cols <- sapply(numeric_data, function(col) sd(col, na.rm = TRUE) == 0)
numeric_data <- numeric_data[, !zero_var_cols]

# Add back the target variable
numeric_data$loan_status <- dfM$loan_status

# Step 5: Handle categorical variables using label encoding
categorical_features <- names(dfM)[sapply(dfM, is.character) & names(dfM) != "loan_status"]
dfM[categorical_features] <- lapply(dfM[categorical_features], as.factor)
dfM[categorical_features] <- lapply(dfM[categorical_features], as.numeric)

# Combine numeric and categorical data
df_final <- cbind(numeric_data, dfM[categorical_features])

# Step 6: Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(df_final$loan_status, p = 0.8, list = FALSE)
trainData <- df_final[trainIndex, ]
testData <- df_final[-trainIndex, ]

# Step 7: Handle class imbalance with oversampling
minority_class <- trainData %>% filter(loan_status == 0)
majority_class <- trainData %>% filter(loan_status == 1)
oversampled_minority <- minority_class[sample(1:nrow(minority_class), size = nrow(majority_class), replace = TRUE), ]
train_balanced <- rbind(majority_class, oversampled_minority)
train_balanced <- train_balanced[sample(1:nrow(train_balanced)), ]  # Shuffle rows

# Step 8: Fit Random Forest
# Fit Random Forest with probability output enabled
rf_model <- ranger(
  loan_status ~ ., 
  data = train_balanced, 
  num.trees = 100, 
  probability = TRUE,  # Enable probability predictions
  importance = "permutation"
)

# Fit Random Forest with probability output enabled
rf_model <- ranger(
  loan_status ~ ., 
  data = train_balanced, 
  num.trees = 100, 
  probability = TRUE,  # Enable probability predictions
  importance = "permutation"
)
# step8 random forest
# Predict on test data
# Load necessary libraries
library(ranger)   # For Random Forest
library(pROC)     # For AUC calculation
library(caret)    # For confusion matrix and metrics
library(ggplot2)  # For visualization
library(dplyr)    # For data manipulation

# Assuming `train_balanced` and `testData` are pre-processed and ready for modeling
# step 8: random forest
# Load necessary libraries
library(ranger)   # For Random Forest
library(pROC)     # For AUC calculation
library(caret)    # For confusion matrix and metrics
library(ggplot2)  # For visualization
library(dplyr)    # For data manipulation

# Assume train_balanced and testData are pre-processed and ready for modeling

# Step 1: Fit Random Forest with Probability Predictions Enabled
rf_model <- ranger(
  loan_status ~ ., 
  data = train_balanced, 
  num.trees = 100, 
  probability = TRUE,  # Enable probabilities for AUC calculation
  importance = "permutation"  # Enable permutation importance
)

# Step 2: Predict on Test Data
rf_predictions <- predict(rf_model, data = testData)$predictions  # Probabilities for the positive class

# Step 3: Ensure rf_predictions is a numeric vector
if (is.matrix(rf_predictions)) {
  rf_predictions <- rf_predictions[, 2]  # Extract probabilities for the positive class
}

# Step 4: Ensure testData$loan_status is Numeric
testData$loan_status <- as.numeric(as.factor(testData$loan_status)) - 1  # Convert to 0/1

# Step 5: Calculate AUC
rf_roc <- roc(response = testData$loan_status, predictor = rf_predictions)
rf_auc <- auc(rf_roc)

# Print AUC
print(paste("Random Forest AUC:", rf_auc))

# Step 6: Convert Probabilities to Binary Predictions
threshold <- 0.25  # Classification threshold
rf_predicted_classes <- ifelse(rf_predictions > threshold, 1, 0)

# Step 7: Evaluate Random Forest
rf_cm <- confusionMatrix(as.factor(rf_predicted_classes), as.factor(testData$loan_status))
rf_kappa <- rf_cm$overall["Kappa"]
rf_accuracy <- rf_cm$overall["Accuracy"]

# Print Metrics
print(paste("Random Forest Accuracy:", rf_accuracy))
print(paste("Random Forest Kappa:", rf_kappa))

# Step 8: Plot ROC Curve
plot(rf_roc, main = "ROC Curve for Random Forest")
abline(a = 0, b = 1, col = "red", lty = 2)  # Add diagonal reference line

# Step 9: Plot Variable Importance
rf_importance <- as.data.frame(rf_model$variable.importance)
rf_importance$Feature <- rownames(rf_importance)
colnames(rf_importance) <- c("Importance", "Feature")
rf_importance <- rf_importance %>% arrange(desc(Importance))

# Visualize Top 10 Variable Importance
ggplot(rf_importance[1:10, ], aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Top 10 Variable Importance (Random Forest)", x = "Feature", y = "Importance") +
  theme_minimal()


# Step 9: Fit XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train_balanced[, -ncol(train_balanced)]), label = train_balanced$loan_status)
dtest <- xgb.DMatrix(data = as.matrix(testData[, -ncol(testData)]), label = testData$loan_status)

xgb_model <- xgboost(data = dtrain, max.depth = 6, eta = 0.1, nrounds = 100, objective = "binary:logistic", verbose = 0)

# Predict on test data
xgb_predictions <- predict(xgb_model, newdata = dtest)
xgb_predicted_classes <- ifelse(xgb_predictions > 0.5, 1, 0)

# Calculate AUC for XGBoost
xgb_roc <- roc(as.numeric(testData$loan_status), xgb_predictions)
xgb_auc <- auc(xgb_roc)

# Evaluate XGBoost
xgb_cm <- confusionMatrix(as.factor(xgb_predicted_classes), as.factor(testData$loan_status))
xgb_kappa <- xgb_cm$overall["Kappa"]
xgb_accuracy <- xgb_cm$overall["Accuracy"]

# Plot Variable Importance for XGBoost
xgb_importance <- xgb.importance(model = xgb_model)
xgb_importance <- xgb_importance[1:10, ]  # Select top 10 features
xgb.plot.importance(xgb_importance)

# Step 10: Compare Models
results <- data.frame(
  Model = c("Random Forest", "XGBoost"),
  Kappa = c(rf_kappa, xgb_kappa),
  Accuracy = c(rf_accuracy, xgb_accuracy),
  AUC = c(rf_auc, xgb_auc)
)

print(results)

# Optional: Visualize Results
results_melted <- melt(results, id.vars = "Model")
ggplot(results_melted, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Comparison", y = "Metric Value", x = "Model") +
  theme_minimal()

# Calculate the optimal threshold
optimal_threshold <- coords(rf_roc, "best", ret = "threshold")
print(paste("Optimal Threshold:", optimal_threshold))

