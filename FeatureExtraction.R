# Clear the environment
rm(list = ls())

# Load necessary libraries
library(keras)
library(tensorflow)
library(Metrics)
library(pROC)
library(tidyverse)

#---------------------------------------------------------------------------------------
## Feature Extraction ##

# Set the path to the dataset
path_to_data <- '' # Replace with the actual path

# Load and preprocess train and test data
train_data_generator <- image_data_generator(rescale = 1/255)
test_data_generator <- image_data_generator(rescale = 1/255)

train_data <- flow_images_from_directory(file.path(path_to_data, "training_set"),
                                        generator = train_data_generator,
                                        target_size = c(32, 32),
                                        batch_size = 32,
                                        class_mode = 'binary')

test_data <- flow_images_from_directory(file.path(path_to_data, "test_set"),
                                        generator = test_data_generator,
                                        target_size = c(32, 32),
                                        batch_size = 32,
                                        class_mode = 'binary',
                                        shuffle = FALSE)