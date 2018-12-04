# Data Set
# Diabetes 130-US hospitals for years 1999-2008 Data Set
# https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

data <- read.csv('dataset_diabetes/diabetic_data.csv', header=TRUE)

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