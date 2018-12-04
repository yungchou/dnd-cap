library(keras)


# Binary Classification
model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(num_input_features)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy"
)

# Single Label Categorical Classification
model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(num_input_features)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy"
)

# Multilabel Categorical Classification
model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(num_input_features)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy"
)

# Regressionmodel <keras_
model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(num_input_features)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = num_values)
model %>% compile(
  optimizer = "rmsprop",
  loss = "mse"
)

# Image Classification
model <- keras_model_sequential() %>%
  layer_separable_conv_2d(filters = 32, kernel_size = 3,
                          activation = "relu",
                          input_shape = c(height, width, channels)) %>%
  layer_separable_conv_2d(filters = 64, kernel_size = 3,
                          activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_separable_conv_2d(filters = 64, kernel_size = 3,
                          activation = "relu") %>%
  layer_separable_conv_2d(filters = 128, kernel_size = 3,
                          activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_separable_conv_2d(filters = 64, kernel_size = 3,
                          activation = "relu") %>%
  layer_separable_conv_2d(filters = 128, kernel_size = 3,
                          activation = "relu") %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")
model %>% compile(optimizer = "rmsprop",
                  loss = "categorical_crossentropy"
)

# Single Layer RNN for Binary Classifiction
model <- keras_model_sequential() %>%
  layer_lstm(units = 32, input_shape = c(num_timestamps, num_features)) %>%
  layer_dense(units = num_classes, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy"
)

# Stack RNN for Binary Classification
model <- keras_model_sequential() %>%
  layer_lstm(units = 32, return_sequences = TRUE,
             input_shape = c(num_timestamps, num_features)) %>%
  layer_lstm(units = 32, return_sequences = TRUE) %>%
  layer_lstm(units = 32) %>%
  layer_dense(units = num_classes, activation = "sigmoid")
model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy"
)
