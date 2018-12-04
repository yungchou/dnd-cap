# Ref: https://cran.r-project.org/web/packages/SuperLearner/vignettes/Guide-to-SuperLearner.html
# Required packages:
# 'caret', 'glmnet', 'randomForest', 'ggplot2', 'RhpcBLASctl',
# 'SuperLearner'

data(Boston, package = "MASS")
str(Boston)
colSums(is.na(Boston))
# If factor variables, use model.matrix() to convert to numerics.

# Extract our outcome variable from the dataframe.
outcome = Boston$medv

# Create a dataframe to contain our explanatory variables.
data = subset(Boston, select = -medv)

dim(data)

#----------------
#  PREPPING DATA
#----------------
set.seed(1)

# Reduce to 150 observations to speed up model fitting.
train_obs = sample(nrow(data), 150)

# X is our training sample.
X_train = data[train_obs, ]

# Create a holdout set for evaluating model performance.
# Note: cross-validation is even better than a single holdout sample.
X_holdout = data[-train_obs, ]

# Create a binary outcome variable:
# towns in which median home value is > 22,000.
outcome_bin = as.numeric(outcome > 22)

Y_train = outcome_bin[train_obs]
Y_holdout = outcome_bin[-train_obs]

# Review the outcome variable distribution.
table(Y_train, useNA = "ifany")

# Availabel models
library(SuperLearner)
listWrappers()   # Screening is for feature selection.

# Basline set of models
# SL.glmnet default to Lasso, i.e. alpha =1
sl_baseline = c( 'SL.mean',  # use SL.mean as a benchmark
  'SL.bartMachine','SL.glmnet','SL.randomForest','SL.xgboost','SL.svm'
)

#----------------
# FITTING MODELS
#----------------
set.seed(1)

# SuperLearner defaults to 10-fold cross-validation.
sl = SuperLearner(Y = Y_train, X = X_train,
                  family = binomial(),
                  # binomial:classification, regression: Gaussian
                  SL.library = sl_baseline)

sl$times$everything; sl

#------------
# PREDICTION
#------------
# onlySL is set to TRUE so we don't fit algorithms
# that had weight = 0, saving computation.
pred = predict(sl, X_holdout, onlySL = TRUE)

# Examine the structure of this prediction object.
str(pred)

# Review the columns of $library.predict.
summary(pred$library.predict)

library(ggplot2)
qplot(pred$pred[, 1]) + theme_minimal()

# Scatterplot of original values (0, 1) and predicted values.
# Ideally we would use jitter or slight transparency to deal with overlap.
qplot(Y_holdout, pred$pred[, 1]) + theme_minimal()

# Review AUC - Area Under Curve
pred_rocr = ROCR::prediction(pred$pred, Y_holdout)
auc = ROCR::performance(pred_rocr, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc

#----------------------------
# EXTERNAAL CROSS-VALIDATION
#----------------------------
set.seed(1)

# We use V = 3 to save computation time;
# for a real analysis use V = 10 or 20.
system.time({
  cv_sl = CV.SuperLearner(Y = Y_train, X = X_train,
                          family = binomial(), V = 3,
                          SL.library = c("SL.mean", "SL.glmnet", "SL.randomForest"))
})
summary(cv_sl)
table(simplify2array(cv_sl$whichDiscreteSL))

# Plot the performance with 95% CIs (use a better ggplot theme).
plot(cv_sl) + theme_bw()


#------------------
# FEATURE SELETION
#------------------
listWrappers()

# Review code for corP, which is based on univariate correlation.
screen.corP

set.seed(1)

# Fit the SuperLearner.
# We need to use list() instead of c().
# We use V = 3 to save computation time; for a real analysis use V = 10 or 20.
cv_sl = CV.SuperLearner(Y = Y_train, X = X_train, family = binomial(), V = 3,
                        parallel = "multicore",
                        SL.library = list("SL.mean", "SL.glmnet", c("SL.glmnet", "screen.corP")))
summary(cv_sl)
