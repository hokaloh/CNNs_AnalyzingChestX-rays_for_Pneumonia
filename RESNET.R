#-------------------------------------------------------------------------
## Modelling ## 

# Function to create a basic residual block
create_residual_block <- function(input_layer, filters, kernel_size, strides = c(1, 1)) {
  x <- input_layer
  
  # First convolutional layer
  x <- layer_conv_2d(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(x)
  x <- layer_batch_normalization()(x)
  x <- layer_activation_relu()(x)
  
  # Second convolutional layer
  x <- layer_conv_2d(filters = filters, kernel_size = kernel_size, padding = 'same')(x)
  x <- layer_batch_normalization()(x)
  
  # Skip connection
  if (strides[1] > 1) {
    input_layer <- layer_conv_2d(filters = filters, kernel_size = c(1, 1), strides = strides, padding = 'same')(input_layer)
  }
  
  x <- layer_add()(list(x, input_layer))
  x <- layer_activation_relu()(x)
  
  return(x)
}

# Function to create ResNet model
create_resnet_model <- function(input_shape = c(224, 224, 3), num_classes = 1000) {
  input_layer <- layer_input(shape = input_shape)
  
  x <- layer_conv_2d(filters = 64, kernel_size = c(7, 7), strides = c(2, 2), padding = 'same')(input_layer)
  x <- layer_batch_normalization()(x)
  x <- layer_activation_relu()(x)
  x <- layer_max_pooling_2d(pool_size = c(3, 3), strides = c(2, 2), padding = 'same')(x)
  
  # Adding residual blocks (adjust the number as needed)
  num_blocks <- 3
  for (i in 1:num_blocks) {
    x <- create_residual_block(x, filters = 64, kernel_size = c(3, 3))
  }
  
  x <- layer_global_average_pooling_2d()(x)
  output <- layer_dense(units = 1, activation = 'sigmoid')(x)
  
  modell <- keras_model(inputs = input_layer, outputs = output)
  return(modell)
}

# Create ResNet model
model <- create_resnet_model(input_shape = c(224, 224, 3), num_classes = 1000)

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
  epochs = 5, 
  validation_data = test_data,
  validation_steps = 19  # Adjust based on the number of test samples
)



