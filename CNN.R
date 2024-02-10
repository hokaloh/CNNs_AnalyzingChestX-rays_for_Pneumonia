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

# Train the model
history <- model %>% fit_generator(
  train_data,
  steps_per_epoch = 163, # Adjust based on the size of train dataset
  epochs = 10,
  validation_data = test_data,
  validation_steps = 19 # Adjust based on validation dataset
)
