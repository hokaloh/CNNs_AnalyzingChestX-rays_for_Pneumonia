#-------------------------------------------------------------------------
## Modelling ## 

# Build the LeNet-5 model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 6, kernel_size = c(5, 5), activation = 'relu', input_shape = c(32, 32, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 16, kernel_size = c(5, 5), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 120, activation = 'relu') %>%
  layer_dense(units = 84, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

# Train the model
history <- model %>% fit_generator(
  train_data,
  steps_per_epoch = 163,  # Adjust based on the size of dataset [train_set/batch_size]
  epochs = 10,
  validation_data = test_data,
  validation_steps = 19  # Adjust based on validation dataset [test_set/batch_size]
)
