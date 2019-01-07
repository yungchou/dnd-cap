if (!require('SuperLearner')) install.packages('SuperLearner'); library(SuperLearner)
if (!require('parallel')) install.packages('parallel'); library(parallel)

if (!require('h2o')) install.packages('h2o'); library(h2o)

if (!require('snow')) install.packages('snow'); library(snow)
if (!require('Rmpi')) install.packages('Rmpi'); library(Rmpi)

if (!require('MASS')) install.packages('MASS'); library(MASS)
if (!require('cvAUC')) install.packages('cvAUC'); library(cvAUC)
if (!require('caret'  )) install.packages('caret'  ); library(caret  )
#if (!require('mlbench')) install.packages('mlbench'); library(mlbench)
if (!require('ranger' )) install.packages('ranger' ); library(ranger )
if (!require('ggplot2')) install.packages('ggplot2'); library(ggplot2)
if (!require('kernlab')) install.packages('kernlab'); library(kernlab)
if (!require('arm')) install.packages('arm'); library(arm)
if (!require('ipred')) install.packages('ipred'); library(ipred)
if (!require('gbm')) install.packages('gbm'); library(gbm)
if (!require('xgboost')) install.packages('xgboost'); library(xgboost)
if (!require('KernelKnn')) install.packages('KernelKnn'); library(KernelKnn)
if (!require('RcppArmadillo')) install.packages('RcppArmadillo'); library(RcppArmadillo)

if(TRUE){ # skip the following

}

#listWrappers()   # Available models in SuperLearner
set.seed(0-0)

#------------------
# DATA PREPARATION
#------------------
df <- read.csv('data/capstone.dataimp.csv') # data set with Boruta selected fetures
df <- df[-1]

# use 500 observations for developing the model
#df <- df[1:100,]

do.smote <- FALSE

if(do.smote) {

  df$readmitted <- as.factor(df$readmitted)
  table(df$readmitted)

  if (!require('DMwR')) install.packages('DMwR'); library(DMwR)
  df.smote <- SMOTE(readmitted~., df, perc.over=100, perc.under=220)
  table(df.smote$readmitted)

  par(mfrow=c(1,2))
  plot(df['readmitted'],las=1,col='lightblue',xlab='df$readmitted',main='Original')
  plot(df.smote['readmitted'],las=1,col='lightgreen',xlab='df.smote$readmitted',main='SMOTE')
  par(mfrow=c(1,1))

  # When converted from factor to numberic, '0' and '1' become '1' and '2'.
  df.smote$readmitted <- as.numeric(factor(df.smote$readmitted))-1
  #tail(df.smote$readmitted)

  part <- sample(2, nrow(df.smote), replace=TRUE, prob=c(0.7,0.3))

  train <- df.smote[part==1,]
  test  <- df.smote[part==2,]

} else {

  part  <- sample(2, nrow(df), replace=TRUE, prob=c(0.7,0.3))
  train <- df[part==1,]
  test  <- df[part==2,]
}

#names(train)

# Check the index of 'readmitted'
x.train <- train[,-27]
y.train <- train[, 27]

#names(x.train)

x.test  <- test[,-27]
y.test  <- test[, 27]

#names(x.test)

#-----------------------------
# ENSEMBLE & CROSS-VALIDATION
#-----------------------------

# For diabetes data set, the following algorithms commented out if
# low or zero coefficients or compatibility issues in previous test runs.

#---------
# Tuning
#---------
xgboost.custom <- create.Learner('SL.xgboost'
  ,tune=list(
    ntrees=c(500,1000,2000), max_depth=1:4
   ,shrinkage=c(0.01,0.1,0.3), minobspernode=c(10,30,50)
   )
  ,detailed_names = TRUE, name_prefix = "xgb"
)
#environment(xgboost.custom) <-asNamespace("SuperLearner")

ranger.custom <- create.Learner('SL.ranger'
 ,tune = list(
    num.trees = c(500,1000,2000)
   ,mtry = floor(sqrt(ncol(x.train))*c(0.5,1,2))
   #,nodesize = c(1,3,5)
   )
 ,detailed_names = TRUE, name_prefix = "ran"
)
#environment(ranger.custom) <-asNamespace("SuperLearner")

glmnet.custom <-  create.Learner("SL.glmnet"
 ,tune = list(
   alpha  = seq(0, 100, length.out=100)
  ,nlambda = seq(0, 100, length.out=100)
   )
 ,detailed_names = TRUE, name_prefix = "enet"
)
#'
#ranger.custom <- function(...) SL.ranger(...,num.trees=1000, mtry=5)
kernelKnn.custom <- function(...) SL.kernelKnn(...,transf_categ_cols=TRUE)

#-----------------------
# SuperLearner Settings
#-----------------------
family   <-  'binomial' #'gaussian'
nnls     <- 'method.NNLS'   # NNLS-default
auc      <- 'method.AUC'   # NNLS-default
nnloglik <- 'method.NNloglik'
SL.algorithm <- c(
  #'SL.ranger'
  ranger.custom$names  #,c('ranger.custom$names','screen.corP')
 #,c('screen.randomForest','screen.randomForest')
 #,'SL.xgboost'
 ,xgboost.custom$names  #,c('xgboost.custom$names','screen.corP')
 #,'SL.glm'
 #,'SL.bayesglm'
#,c('glmnet.custome$names','screen.glmnet')
 #,c('kernelKnn.custom','screen.corP')
 #,c('SL.nnet','screen.corP')
 #,c('SL.gbm','screen.corP')
 #,SL.treebag'
 #,'SL.svmRadial'
)

#-------------------------------
# Multicore/Parallel Processing
#-------------------------------

nfold <- 10 # Use 10-20 for production

cl <- makeCluster(detectCores()-1)

#listWrappers()
clusterExport(cl, c( listWrappers()

  ,'SuperLearner','CV.SuperLearner','predict.SuperLearner'
  ,'nfold','y.train','x.train','x.test'
  ,'family','nnls','auc','nnloglik'

  ,'SL.algorithm'
  ,ranger.custom$names,xgboost.custom$names
  ,'kernelKnn.custom'
  ,'glmnet.custom$names'

  ))

clusterSetRNGStream(cl, iseed=135)

system.time({

  clusterEvalQ(cl,{

    # NNLS
    ensem.nnls <- SuperLearner(Y=y.train, X=x.train, verbose=TRUE
      ,family=family,method=nnls
      ,SL.library=SL.algorithm,cvControl=list(V=nfold)
        )
    saveRDS(ensem.nnls, 'ensem.nnls')
    pred.nnls.train <- predict.SuperLearner(ensem.nnls, x.train)

    # AUC
    ensem.auc <- SuperLearner( Y=y.train, X=x.train, verbose=TRUE
      ,family=family,method=auc
      ,SL.library=SL.algorithm,cvControl=list(V=nfold)
      )
    saveRDS(ensem.auc, 'ensem.auc')
    pred.auc.train <- predict.SuperLearner(ensem.auc, x.train)

    # NNLogLik
    ensem.nnloglik <- SuperLearner( Y=y.train, X=x.train, verbose=TRUE
      ,family=family,method=nnloglik
      ,SL.library=SL.algorithm,cvControl=list(V=nfold)
      )
    saveRDS(ensem.nnloglik, 'ensem.nnloglik')
    pred.nnloglik.train <- predict.SuperLearner(ensem.nnloglik, x.train)

  })

  stopCluster(cl)

})

#------------------------------------------
# Read in results form papallel processing
#------------------------------------------
ensem.nnls <- readRDS('ensem.nnls');ensem.nnls$times;ensem.nnls
ensem.auc  <- readRDS('ensem.auc');ensem.auc$times;ensem.auc
ensem.nnloglik <- readRDS('ensem.nnloglik');ensem.nnloglik$times;ensem.nnloglik

#------------
# PREDICTION
#------------
pred.nnls <- predict.SuperLearner(ensem.nnls, x.test, onlySL=TRUE)
summary(pred.nnls$library.predict);summary(pred.nnls$pred)

pred.auc <- predict.SuperLearner(ensem.auc, x.test, onlySL=TRUE)
summary(pred.auc$library.predict);summary(pred.auc$pred)

pred.nnloglik <- predict.SuperLearner(ensem.nnloglik, x.test, onlySL=TRUE)
summary(pred.nnloglik$library.predict);summary(pred.nnloglik$pred)

#------------------
# CONSUSION MATRIX
#------------------
# Converting probabilities into classification
pred.nnls.converted     <- ifelse(pred.nnls$pred>=0.5,1,0)
pred.auc.converted      <- ifelse(pred.auc$pred>=0.5,1,0)
pred.nnloglik.converted <- ifelse(pred.nnloglik$pred>=0.5,1,0)

# Confusion Matrix
cm.nnls <- confusionMatrix(factor(pred.nnls.converted), factor(y.test));cm.nnls
cat('Mean Square Error (NNLS) = ',mse.nnls <- mean((y.test-pred.nnls$pred)^2))
cm.auc <- confusionMatrix(factor(pred.auc.converted), factor(y.test));cm.auc
cat('Mean Square Error (AUC) = ',mse.auc <- mean((y.test-pred.auc$pred)^2))
cm.nnloglik <- confusionMatrix(factor(pred.nnloglik.converted), factor(y.test));cm.nnloglik
cat('Mean Square Error (NNLogLik) = ',mse.nnloglik <- mean((y.test-pred.nnloglik$pred)^2))

noquote(cbind(
  Method=c('nnls','auc','nnloglik')
 ,MSE=c(mse.nnls,mse.auc,mse.nnloglik)
 ,Accuracy=c(
    cm.nnls$overall['Accuracy']
   ,cm.auc$overall['Accuracy']
   ,cm.nnloglik$overall['Accuracy']
  )
))

#-----------------------------
# Comparing the three methods
#-----------------------------
library(dplyr);library(plotly)
err <- as.data.frame( cbind(
   pred.nnls$pred
  ,pred.auc$pred
  ,pred.nnloglik$pred
));colnames(err) <- c('nnls','auc','nnloglik'
);err$label <- as.factor(ifelse(y.test == 0,'Not readmitted','Readmitted')
);colnames(err) <- c('nnls','auc','nnloglik','label')

p3d.method <- plot_ly(err
  ,x = ~err$nnls, y = ~err$auc, z = ~err$nnloglik, color = err$label
  ,hoverinfo = 'text'
  ,text = ~paste(
     'NNLS:\t',round(err$nnls,7)
    ,'\nAUC:\t', round(err$auc,7)
    ,'\nNNLogLik: ', round(err$nnloglik,7)
    ,'\nLabel:\t', err$label)
  ,colors = c('blue', 'yellow', 'red')
  ,marker = list( size = 10, opacity = 1
    ,line = list( color = 'black', width = 1))
) %>% add_markers() %>%
  layout(title='Comparison of SuperLearner Methods',scene = list(
    xaxis = list(title = 'NNLS')
   ,yaxis = list(title = 'AUC')
   ,zaxis = list(title = 'NNLogLik'))
   );p3d.method

#---------------------------------------------
# External Cross-Validation for the Ensembles
#---------------------------------------------
system.time({
  ensem.cv <-  CV.SuperLearner( Y=y.train, X=x.train, verbose=TRUE, V=3
    ,family=SL.family, method=SL.method ,SL.library=SL.algorithm)
}); ensem.cv
})

summary(ensem.cv); saveRDS('ensem','capstone.ensem.cv.rds')

plot(ensem.cv) + theme_minimal()


#-----
# AUC
#-----
auc <- cvAUC(p1,labels=y.test)$cvAUC;auc

#Plot fold AUCs
plot(auc$perf, col="grey82", lty=3, main="10-fold CV AUC")
#Plot CV AUC
plot(auc$perf, col="red", avg="vertical", add=TRUE)