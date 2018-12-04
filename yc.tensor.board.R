library(keras)

max_features <2000
max_len <500

imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
x_train <- pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 128,
                  input_length = max_len, name = "embed") %>%
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 5) %>%
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 1)

summary(model)

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

# Before you start using TensorBoard, you need to create
# a directory where you’ll store the log files it generates.
dir.create("my_log_dir")

tensorboard("my_log_dir")
callbacks = list(
  callback_tensorboard(
    log_dir = "my_log_dir",
    histogram_freq = 1,
    embeddings_freq = 1
  )
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2,
  callbacks = callbacks
)

# depthwise separable convolution layer
library(keras)
height <- 64
width <- 64
channels <- 3
num_classes <- 10
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
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy"
)

# Ensemble model
# Ensembling consists of pooling together the predictions
# of a set of different models, to produce better predictions.
# This will work only if the classifiers are more or less
# equally good. If one of them is significantly worse than
# the others, the final predictions may not be as good as
# the best classifier of the group.

preds_a <- model_a %>% predict(x_val) 1
preds_b <- model_b %>% predict(x_val)
preds_c <- model_c %>% predict(x_val)
preds_d <- model_d %>% predict(x_val)
final_preds <- 0.25* (preds_a + preds_b + preds_c + preds_d)

# A smarter way to ensemble classifiers is to do a weighted
# average, where the weights are learned on the validation
# data—typically, the better classifiers are given a higher
# weight, and the worse classifiers are given a lower weight.

preds_a <model_a %>% predict(x_val)
preds_b <model_b %>% predict(x_val)
preds_c <model_c %>% predict(x_val)
preds_d <model_d %>% predict(x_val)
final_preds < 0.5*preds_a + 0.25*preds_b + 0.1*preds_c + 0.15*preds_d
# These weights (0.5, 0.25, 0.1, 0.15) are assumed to be
# learned empirically.


