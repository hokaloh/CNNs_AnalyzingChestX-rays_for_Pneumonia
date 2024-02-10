# Clear the environment
rm(list = ls())

# Load necessary libraries
library(keras)
library(tensorflow)
library(Metrics)
library(pROC)

path_to_data <- 'C:/Users/syahi/Documents/Deep-Chest-Diagnostics/Datasets/chest_xray'

# Preprocess and load the dataset
train_data_generator <- image_data_generator(rescale = 1/255)
test_data_generator <- image_data_generator(rescale = 1/255)

train_data <- flow_images_from_directory(file.path(path_to_data, "train_Set"),
                                         generator = train_data_generator,
                                         target_size = c(32, 32),  # Adjust target_size based on your model requirements
                                         batch_size = 32,
                                         class_mode = 'binary')

test_data <- flow_images_from_directory(file.path(path_to_data, "test_set"),
                                        generator = test_data_generator,
                                        target_size = c(32, 32),  # Adjust target_size based on your model requirements
                                        batch_size = 32,
                                        class_mode = 'binary',
                                        shuffle = FALSE)
