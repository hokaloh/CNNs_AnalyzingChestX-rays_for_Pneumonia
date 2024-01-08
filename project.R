#clear the environment
rm(list = ls())

#CNN 2 package
# Load necessary libraries
install.packages("Metrics") 
install.packages("pROC")
library(tidyverse)
library(keras)
library(tensorflow)
library(Metrics)
library(pROC)

# Dataset - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download

# path directory of MyDataset
path_to_data <- 'C:/Users/syahi/Documents/Degree UTHM/Data Mining/Project/project/Datasets/chest_xray'

#---------------------------------------------------------------------------------------
## Feature Extraction ##

# Preparing Training Data 
train_data_generator <- image_data_generator(rescale = 1/255) 
# pixel_values of the images will be rescaled by dividing them by 255
train_data <- flow_images_from_directory(file.path(path_to_data, "train"),
                                         generator = train_data_generator,
                                         target_size = c(64, 64),
                                         batch_size = 32,
                                         class_mode = 'binary')

# Preparing Test Data
test_data_generator <- image_data_generator(rescale = 1/255)

test_data <- flow_images_from_directory(file.path(path_to_data, "test"),
                                        generator = test_data_generator,
                                        target_size = c(64, 64),
                                        batch_size = 32,
                                        class_mode = 'binary',
                                        shuffle = FALSE)

#-------------------------------------------------------------------------

## Modelling ## 

# Building the CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(64, 64, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')
# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)


#----------------------------------------------------------------------

## Train and Testing The Model ## 

# Train the model
history <- model %>% fit_generator(
  train_data,
  steps_per_epoch = 163, # Train_Images/Batch_Size [Number batch process per_epoch during training]
  epochs = 10,
  validation_data = test_data,
  validation_steps = 19  # Test_Images/Batch_Size
)


# Evaluate model on test data for loss and accuracy
model_evaluation <- model %>% evaluate_generator(
  test_data,
  steps = test_data$n / test_data$batch_size
)


#-------------------------------------------------------------------------------

## Evaluation Metrics [Accuracy, Precision, Recall, F-score] ## 
## Additional ROC(receiver operating characteristic curve) - graph performance of a classification model at all classification thresholds

# Metrics of loss accuracy in index 
view(model_evaluation)
test_loss <- model_evaluation[1]
test_accuracy <- model_evaluation[2]
cat("Test Loss:", test_loss, "\n")
cat("Test Accuracy:", test_accuracy, "\n")

# Predict on test data for precision, recall, F1 score, and ROC
predictions <- model %>% predict_generator(test_data, steps = ceiling(test_data$n / test_data$batch_size))
predictions <- as.vector(predictions)  # Ensure predictions is a numeric vector
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
true_classes <- test_data$classes

# Ensure true_classes is a numeric vector of the same length as predictions
true_classes <- as.vector(true_classes[1:length(predictions)])

# Calculate precision and recall
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

# Plot ROC curve
plot(roc_analysis, main = "ROC Curve")

