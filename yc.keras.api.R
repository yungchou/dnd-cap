library(keras)

# The functional API can be used to build models that
# have multiple inputs.

seq_model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu", input_shape = c(64)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

input_tensor <- layer_input(shape = c(64))

output_tensor <- input_tensor %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model <- keras_model(input_tensor, output_tensor)

summary(model)

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy"
)

x_train <- array(
  runif(1000 * 64), dim = c(1000, 64))
y_train <- array(
  runif(1000 * 10), dim = c(1000, 10))

model %>% fit(x_train, y_train, epochs = 10, batch_size = 128)

model %>% evaluate(x_train, y_train)


# Shared Convolutional Base
library(keras)
xception_base <- application_xception(weights = NULL,
                                      include_top = FALSE)
left_input <- layer_input(shape = c(250, 250, 3))
right_input <- layer_input(shape = c(250, 250, 3))
left_features = left_input %>% xception_base()
right_features <- right_input %>% xception_base()
merged_features <- layer_concatenate(
  list(left_features, right_features)
)

# Using callbacks to act on a model and stop training
# when you measure that the validation loss in no longer
# improving. This can be achieved using a Keras callback.
library(keras)

callbacks_list <- list(
  callback_early_stopping(
    monitor = "acc",
    patience = 1
    # call back, if accuracy has stopped imrpoving more than 1 epoch.
    ),
  callback_model_checkpoint(
    filepath = "my_model.h5",
    monitor = "val_loss",
    save_best_only = TRUE
    )
  )

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
  )

model %>% fit(
  x, y,
  epochs = 10,
  batch_size = 32,
  callbacks = callbacks_list,
  validation_data = list(x_val, y_val)
)

callbacks_list <list(
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.1,
    patience = 10
  )
)

model %>% fit(
  x, y,
  epochs = 10,
  batch_size = 32,
  callbacks = callbacks_list,
  validation_data = list(x_val, y_val)
)

# Save a list of losses over each batch during training
library(keras)
library(R6)
LossHistory  <- R6Class(
  "LossHistory",
  inherit = KerasCallback,
  public = list(
    losses = NULL,
    on_batch_end = function(batch, logs = list()) {
      self$losses <c(self$losses, logs[["loss"]])
      }
    )
  )

history <LossHistory$new()

model %>% fit(
  x, y,
  batch_size = 128,
  epochs = 20,
  callbacks = list(history)
)

str(history$losses)
