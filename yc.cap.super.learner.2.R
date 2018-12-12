if (!require('xgboost')) install.packages('xgboost'); library(xgboost)
if (!require('SuperLearner')) install.packages('SuperLearner'); library(SuperLearner)
if (!require('ranger')) install.packages('ranger'); library(ranger)
if (!require('caret')) install.packages('caret'); library(caret)
if (!require('DMwR'   )) install.packages('DMwR'   ); library(DMwR   )

df <- read.csv('demo/capstone.dataimp.keeper.csv')[-1]

set.seed(0-0)

df$readmitted <- as.factor(df$readmitted)
table(df$readmitted)
df.smote <- SMOTE(readmitted~., df, perc.over=100, perc.under=220)
table(df.smote$readmitted)

df.smote$readmitted <- as.numeric(factor(df.smote$readmitted))-1

part <- sample(2, nrow(df.smote), replace=TRUE, prob=c(0.7,0.3))

train <- df.smote[part==1,]
test  <- df.smote[part==2,]

names(train)

x.train <- train[,-27]
y.train <- train[, 27]

x.test  <- test[,-27]
y.test  <- test[, 27]

create.SL.xgboost(
  tune = list(ntrees = c(300, 500), max_depth = c(4),
              shrinkage =c(0.1), minobspernode = c(10)),
  detailed_names = F, env = .GlobalEnv,name_prefix = "SL.xgb")

SL.ranger.tune <- function(...){SL.ranger(..., num.trees=1000, mtry=5)}

set.seed(1-1)
ensem <-  SuperLearner( Y=y.train, X=x.train,
                        family='binomial', verbose=TRUE,
                        SL.library=c( #'SL.kernelKnn','SL.nnet',
                          #'SL.glmnet',
                          #'SL.ipredbagg','SL.ksvm', 'SL.gbm',
                          #'SL.ranger',
                          'SL.ranger.tune'
                          #'SL.xgboost',
                          #'create.SL.xgboost',
                          #'SL.mean' # baseline algorithm
                        ))

ensem; saveRDS('ensem','capstone.SL.ranger.rds')

ensem.cv <-  CV.SuperLearner( Y=y.train, X=x.train,
                              family='binomial', verbose=TRUE, V=10,
                              SL.library=c( #'SL.kernelKnn','SL.nnet',
                                #'SL.glmnet',
                                #'SL.ipredbagg','SL.ksvm', 'SL.gbm',
                                #'SL.ranger',
                                'SL.ranger.tune'
                                #'SL.xgboost',
                                #'create.SL.xgboost',
                                #'SL.mean' # baseline algorithm
                              ))


summary(ensem.cv); saveRDS('ensem','capstone.ranger.cv.rds')

plot(ensem.cv) + theme_minimal()

#------------
# PREDICTION
#------------
predictions <- predict.SuperLearner(ensem, x.test, onlySL=TRUE)

summary(predictions$library.predict)

head(predictions$pred)

head(predictions$library.predict)

# Converting probabilities into classification
predictiions.converted <- ifelse(predictions$pred>=0.5,1,0)

# Construct a confusion matrix
cm <- confusionMatrix(factor(predictiions.converted), factor(y.test)) 
cm

#mean((y.train-ensem.cv$SL.predict)^2) #CV risk for SuperLearner


#ensem.cv$cvRisk[which.min(ensem.cv$cvRisk)]

#SL.rf.better <- function(...) {SL.randomForest(..., ntree = 3000)}
#SL.ipredbagg.tune <- function(...){SL.ipredbagg(..., nbagg=250)}
#SL.xgboost.tune <- function(...){SL.xgboost(..., ntree=250, max_depth=4, shrinkage=0.01)}