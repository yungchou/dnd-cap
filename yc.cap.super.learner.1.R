if (!require('SuperLearner')) install.packages('SuperLearner'); library(SuperLearner)

if(TRUE){ # skip the following

if (!require('h2o')) install.packages('h2o'); library(h2o)

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

}

#listWrappers()   # Available models in SuperLearner
set.seed(0-0)

#------------------
# DATA PREPARATION
#------------------
df <- read.csv('data/capstone.dataimp.csv') # data set with Boruta selected fetures
df <- df[-1]

# use 500 observations for developing the model
df <- df[1:10000,]

do.smote <- TRUE

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

#-------------
# Tuning Grid
#-------------
xgboost.custom <- create.Learner('SL.xgboost'
  ,tune=list(
    ntrees=c(500,1000), max_depth=3:4
   ,shrinkage=c(0.01,0.1,0.3), minobspernode=c(10,30)
   )
  ,detailed_names = TRUE, name_prefix = 'xgboost'
)
#environment(xgboost.custom) <-asNamespace("SuperLearner")

ranger.custom <- create.Learner('SL.ranger'
 ,tune = list(
    num.trees = c(500,1000)
   ,mtry = floor(sqrt(ncol(x.train))*c(0.5,1,2))
   #,nodesize = c(1,3,5)
   )
 ,detailed_names = TRUE, name_prefix = 'ranger'
)

#environment(ranger.custom) <-asNamespace("SuperLearner")

glmnet.custom <-  create.Learner('SL.glmnet'
 ,tune = list(
   alpha  = seq(0, 1, length.out=10)  # (0,1)=>(ridge, lasso)
  ,nlambda = seq(0, 10, length.out=10)
   )
 ,detailed_names = TRUE, name_prefix = 'glmnet'
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
  ranger.custom$names  #,c('ranger.custom$names','screen.corP')
 ,xgboost.custom$names  #,c('xgboost.custom$names','screen.corP')
)

#SL.algorithm <- list(c('SL.ranger','screen.corP'),c('SL.xgboost','screen.corP'))
  #'SL.ranger','SL.xgboost'
 #glmnet.custom$names#,'screen.glmnet'
 #,c('screen.randomForest','screen.randomForest')
 #,'SL.xgboost'
 #,'SL.glm'
 #,'SL.bayesglm'
 #,c('kernelKnn.custom','screen.corP')
 #,c('SL.nnet','screen.corP')
 #,c('SL.gbm','screen.corP')
 #,SL.treebag'
 #,'SL.svmRadial'

#-------------------------------
# Multicore/Parallel Processing
#-------------------------------

nfold <- 3 # Use 10-20 for production

if (!require('parallel')) install.packages('parallel'); library(parallel)
#if (!require('snow')) install.packages('snow'); library(snow)
#if (!require('Rmpi')) install.packages('Rmpi'); library(Rmpi)

cl <- makeCluster(detectCores()-1)

#listWrappers()
clusterExport(cl, c( listWrappers()

  ,'SuperLearner','CV.SuperLearner','predict.SuperLearner'
  ,'nfold','y.train','x.train','x.test'
  ,'family','nnls','auc','nnloglik'

  ,'SL.algorithm'
  ,ranger.custom$names,xgboost.custom$names
  ,glmnet.custom$names

  ))

clusterSetRNGStream(cl, iseed=135)
#----------------------
# Parallel Proccessing
#----------------------
system.time({

  clusterEvalQ(cl,{

    # NNLS
    ensem.nnls <- SuperLearner(Y=y.train, X=x.train, verbose=TRUE
      ,family=family,method=nnls
      ,SL.library=SL.algorithm,cvControl=list(V=nfold)
        )
    saveRDS(ensem.nnls, 'ensem.nnls')

    # AUC
    ensem.auc <- SuperLearner( Y=y.train, X=x.train, verbose=TRUE
      ,family=family,method=auc
      ,SL.library=SL.algorithm,cvControl=list(V=nfold)
      )
    saveRDS(ensem.auc, 'ensem.auc')

    # NNLogLik
    ensem.nnloglik <- SuperLearner( Y=y.train, X=x.train, verbose=TRUE
      ,family=family,method=nnloglik
      ,SL.library=SL.algorithm,cvControl=list(V=nfold)
      )
    saveRDS(ensem.nnloglik, 'ensem.nnloglik')

  })

})

system.time({

  clusterEvalQ(cl,{

    # NNLS
    ensem.nnls.cv <- CV.SuperLearner(Y=y.train, X=x.train, verbose=TRUE
      ,family=family,method=nnls
      ,SL.library=SL.algorithm,cvControl=list(V=nfold)
    )
    saveRDS(ensem.nnls.cv, 'ensem.nnls.cv')

    # AUC
    ensem.auc.cv <- CV.SuperLearner( Y=y.train, X=x.train, verbose=TRUE
      ,family=family,method=auc
      ,SL.library=SL.algorithm,cvControl=list(V=nfold)
    )
    saveRDS(ensem.auc.cv, 'ensem.auc.cv')

    # NNLogLik
    ensem.nnloglik.cv <- CV.SuperLearner( Y=y.train, X=x.train, verbose=TRUE
      ,family=family,method=nnloglik
      ,SL.library=SL.algorithm,cvControl=list(V=nfold)
    )
    saveRDS(ensem.nnloglik, 'ensem.nnloglik.cv')

  })

})

stopCluster(cl)

#------------------------------------------
# Read in results form papallel processing
#------------------------------------------
ensem.nnls <- readRDS('ensem.nnls');ensem.nnls$times;ensem.nnls
ensem.auc  <- readRDS('ensem.auc');ensem.auc$times;ensem.auc
ensem.nnloglik <- readRDS('ensem.nnloglik');ensem.nnloglik$times;ensem.nnloglik

compare.risk <- cbind(ensem.nnls$cvRisk,ensem.auc$cvRisk,ensem.nnloglik$cvRisk)
colnames(compare.risk) <- c('NNLS', 'AUC','NNLogLik');compare.risk

plot(ensem.nnls$cvRisk,las=1,pch=1)
points(ensem.auc$cvRisk,col='red',pch=2)
points(ensem.nnloglik$cvRisk,col='blue',pch=3)

#------------
# PREDICTION
#------------
pred.nnls<- predict.SuperLearner(ensem.nnls, x.test, onlySL=TRUE)
summary(pred.nnls$pred)

pred.auc <- predict.SuperLearner(ensem.auc, x.test, onlySL=TRUE)
summary(pred.auc$pred)

pred.nnloglik <- predict.SuperLearner(ensem.nnloglik, x.test, onlySL=TRUE)
summary(pred.nnloglik$pred)

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

#------------------------------------------------
# COMPARING PREDICTONS MADE BY THE THREE METHODS
#------------------------------------------------
pred3 <- as.data.frame( cbind(
   pred.nnls$pred
  ,pred.auc$pred
  ,pred.nnloglik$pred
));pred3$label <- as.factor(ifelse(y.test == 0,'Not readmitted','Readmitted')
);pred3$y.test <- y.test
colnames(pred3) <- c('nnls','auc','nnloglik','label','y.test');pred3

#-----------------
# 2D Scatter Plot
#-----------------
plot(pred3$nnls,las=1,pch=1, main='Comparison of Predictions')
points(pred3$auc,col='red',pch=2)
points(pred3$nnloglik,col='blue',pch=3)

if (!require('dplyr')) install.packages('dplyr'); library(dplyr)
if (!require('plotly')) install.packages('plotly'); library(plotly)

p2d.pred.method <- plot_ly( pred3, type='scatter',mode = 'markers'
                            ,width=1280 ,height=700 #,margin=5
  ,x = ~(1:length(pred3$label)), y=~pred3$nnls, name='NNLS' ) %>%
  add_trace(y = ~pred3$auc, name='AUC') %>%
  add_trace(y = ~pred3$nnloglik, name='NNLokLik') %>%
  add_trace(y = ~pred3$y.test, name='Label') %>%
  layout( title='Comparing Predictions Made by NNLS/AUC/NNLogLik'
         ,xaxis=list(title='')
         ,yaxis=list(title='Prediction')
         ,plot_bgcolor="rgb(230,230,230)"
  );p2d.pred.method

#-----------------
# 3D Scatter Plot
#-----------------
p3d.pred.method <- plot_ly( pred3
  ,x = ~pred3$auc, y = ~pred3$nnls, z = ~pred3$nnloglik, color = pred3$label
  ,hoverinfo = 'text'
  ,text = ~paste(
    'AUC:\t',round(pred3$auc,7)
    ,'\nNNLS:\t', round(pred3$nnls,7)
    ,'\nNNLogLik:\t', round(pred3$nnloglik,7)
    ,'\nLabel:\t', pred3$label)
  ,colors = c('blue', 'yellow', 'red')
  ,marker = list(
    size = 10, opacity = 0.5
    ,line = list( color = 'black', width = 1))
) %>% add_markers() %>%
  layout( title='Comparing Predictions Made by NNLS/AUC/NNLogLik',scene = list(
     xaxis = list(
      title = 'AUC'
      ,backgroundcolor="rgb(204,204,204)"
      ,gridcolor="rgb(255, 255, 255)"
      ,zerolinecolor="rgb(0,0,0)"
      ,showbackground=TRUE)
    ,yaxis = list(
      title = 'NNLS'
      ,backgroundcolor="rgb(217,217,217)"
      ,gridcolor="rgb(255, 255, 255)"
      ,zerolinecolor="rgb(0,0,0)"
      ,showbackground=TRUE)
    ,zaxis = list(
      title = 'NNLogLik'
      ,backgroundcolor="rgb(230,230,230)"
      ,gridcolor="rgb(255, 255, 255)"
      ,zerolinecolor="rgb(0,0,0)"
      ,showbackground=TRUE)
    ,camera = list(
      up=list(x=0, y=0, z=1)
      ,center=list(x=0, y=0, z=0)
      ,eye=list(x=2, y=0.4, z=0.25)
    )
  )
  );p3d.pred.method

#---------------------------------------------
# External Cross-Validation for the Ensembles
#---------------------------------------------
system.time({
  ensem.cv <-  CV.SuperLearner( Y=y.train, X=x.train, verbose=TRUE, V=3
    ,family=SL.family, method=SL.method ,SL.library=SL.algorithm)
}); ensem.cv

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
