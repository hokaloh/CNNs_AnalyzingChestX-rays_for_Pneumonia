#-------------------------------------------------------------------------
## Modelling ## 

# Function to create VGG-style model
create_vgg_model <- function() {
  modell <- keras_model_sequential()
  
  # Block 1
  modell %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu', padding = 'same', input_shape = c(224, 224, 3)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  
  # Block 2
  modell %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  
  # Block 3
  modell %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  
  # Block 4
  modell %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  
  # Block 5
  modell %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  
  # Flatten the output and add dense layers for classification
  modell %>%
    layer_flatten() %>%
    layer_dense(units = 4096, activation = 'relu') %>%
    layer_dense(units = 4096, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'sigmoid')  # Assuming 1000 classes for ImageNet
  
  return(modell)
}

# Create the VGG model
model <- create_vgg_model()

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

# Train the model
history <- model %>% fit(
  train_data,
  steps_per_epoch = 163,  # Adjust based on the number of training samples
  epochs = 5,  # Choose the number of epochs
  validation_data = test_data,
  validation_steps = 19  # Adjust based on the number of test samples
)

