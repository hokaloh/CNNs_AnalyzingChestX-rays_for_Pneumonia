#---------------------------------------------------------------------------------------
# Evalution Models #

## Evaluation Metrics [Accuracy, Precision, Recall, F-score] ## 
## Additional ROC(receiver operating characteristic curve) - graph performance of a classification model at all classification thresholds
model_evaluation <- model %>% evaluate_generator(
  test_data,
  steps = test_data$n / test_data$batch_size
)

# Access the metrics by index
# Usually, the first element is loss and the second element is accuracy
test_loss <- model_evaluation[1]
test_accuracy <- model_evaluation[2]

# Predict on test data for precision, recall, F1 score, and ROC
predictions <- model %>% predict_generator(test_data, steps = ceiling(test_data$n / test_data$batch_size))
predictions <- as.vector(predictions)  # Ensure predictions is a numeric vector
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
true_classes <- test_data$classes

# Ensure true_classes is a numeric vector of the same length as predictions
true_classes <- as.vector(true_classes[1:length(predictions)])

# Calculate PRECISION and RECALL
precision_value <- precision(true_classes, predicted_classes)
recall_value <- recall(true_classes, predicted_classes)

# Manually calculate F1 score
f1_score_value <- if (precision_value + recall_value == 0) {
  0  # Handle the case where the denominator is zero
} else {
  2 * (precision_value * recall_value) / (precision_value + recall_value)
}

# ROC and AUC
roc_analysis <- roc(true_classes, predictions)
auc_value <- auc(roc_analysis)
# Print additional metrics
cat("Precision:", precision_value, "\n")
cat("Recall:", recall_value, "\n")
cat("F1 Score:", f1_score_value, "\n")
cat("AUC:", auc_value, "\n")
cat("Test Loss:", test_loss, "\n")
cat("Test Accuracy:", test_accuracy, "\n")

# Plot ROC curve
plot(roc_analysis, main = "ROC Curve")