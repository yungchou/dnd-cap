# Ref: https://github.com/ecpolley/SuperLearner
# For maximum accuracy one might try at least the following models:
# glmnet, randomForest, XGBoost, SVM, and bartMachine.
# These should ideally be tested with multiple hyperparameter
# settings for each algorithm.
sl_baseline = c(
  'SL.glmnet', 'SL.randomForest', 'SL.xgboost',
  'SL.SVM', 'SL.bartMachine'
  )

library(SuperLearner)
listWrappers()   # Available models in SuperLearner

data(Boston, package = "MASS")

colSums(is.na(Boston))

set.seed(1-1)

sl_lib = c("SL.xgboost", "SL.randomForest", "SL.glmnet",
           "SL.nnet", "SL.ksvm","SL.bartMachine",
           "SL.kernelKnn", "SL.rpartPrune", "SL.lm", "SL.mean")

# Fit XGBoost, RF, Lasso, Neural Net, SVM, BART, K-nearest neighbors, Decision Tree,
# OLS, and simple mean; create automatic ensemble.
boxplot(Boston[,-14])
dn <- data.frame(scale(Boston[,-14]))
boxplot(dn)
result = SuperLearner(Y=Boston$medv, X=dn, SL.library=sl_lib)

# Review performance of each algorithm and ensemble weights.
result

# Use external (aka nested) cross-validation to estimate ensemble accuracy.
# This will take a while to run.
# Default to 10-fold
result2 = CV.SuperLearner(Y=Boston$medv, dn, SL.library=sl_lib)

# Plot performance of individual algorithms and compare to the ensemble.
plot(result2) + theme_minimal()

# Hyperparameter optimization --
# Fit elastic net with 5 different alphas:
# 0 (Ridege), 0.2, 0.4, 0.6, 0.8, 1.0 (Lasso).
enet = create.Learner("SL.glmnet", detailed_names=TRUE,
                      tune=list( alpha=seq(0,1,length.out=5) ))

sl_lib2 = c("SL.mean", "SL.lm", enet$names)

enet_sl = SuperLearner(Y=Boston$medv, X=Boston[, -14], SL.library=sl_lib2)

# Identify the best-performing alpha value or use the automatic ensemble.
enet_sl
