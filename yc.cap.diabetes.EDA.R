if (!require('knitr')) install.packages('knitr'); library(knitr)
if (!require('psych')) install.packages('psych'); library(psych)
if (!require('dplyr')) install.packages('dplyr'); library(dplyr)
if (!require('corrplot')) install.packages('corrplot'); library(corrplot)
if (!require('caret')) install.packages('caret'); library(caret)
if (!require('mice')) install.packages('mice'); library(mice)
if (!require('stats')) install.packages('stats'); library(stats)

#----------
# DATA SET
#----------
# Dropping encounter_id upon importing
data<- read.csv('data/diabetic_data.csv', header=TRUE)
#ref  <- read.csv('dataset_diabetes/IDs_mapping.csv'  , header=TRUE)

cat('Diabetes data set imported (',nrow(data),
    ' observations with ', ncol(data),' variables )')

str(data) # 101766 obs. of  50 variables

#-------------
# MISSINGNESS
#-------------
# missing values were stored as '?'
cat('Missing values indicated by "?" = ',sum(data=='?'),'\n')
cat('Replacing ', sum(data=='?'), ' values stored as "?" with NA.')
data[data=='?'] <- NA
cat('Total NA count = ',sum(is.na(data)))

if (!require('naniar')) install.packages('naniar'); library('naniar')
gg_miss_upset(data);

#------------------------------
# Percentage of values missing
#------------------------------
q <- sapply(colnames(data), function(x){
  round(sum(is.na(data[x]))*100/nrow(data[x]),2)
  })
# Variables with percentage of missing values > 0
q[q>0]

barplot( log(q[q>0]+1),
         las=2, col=grey.colors(10), cex.names = 0.8,
         main='Missingness', ylab='log(% missing)', xlab='')
abline(h=log(30+1), col='red',lwd=2)

#-----------------------------------------------------
# Keeping variables with less than 30% values missing
#-----------------------------------------------------
cat('Removing the three variables: ', names(q[q>30]))
data <- data[ names(data) %in% names(q[q<30]) ] # here 45 variables

cat('At this time, the data set has ',nrow(data),' observations with ', ncol(data),' variables.')
# Remaining variables with missing values: race, diag_1, diag_2, diag_3

#------------------------------
# NEAR_SERO VARIANCE VARIABLES
#------------------------------
# The following two factor variables have only one level with no missing value.
# Therefore these two variables have zero-variance, are not influential on
# predicting readmission and removed.

nzv <- names(data[ nearZeroVar(data,saveMetrics=FALSE) ])
cat('caret reports ',length(nzv),'near-nero variables as the following:\n\n');nzv

if(length(nzv)>0){data <- data[, !names(data)%in%nzv]}

cat('The listed, ',length(nzv),' near-zero variables have been removed.\n\n', 'At this time, the data set has ',nrow(data),' observations with ', ncol(data),' variables remaining.')

#-----------------------------
# MULRIPLE-ENCOUNTER PATIENTS
#-----------------------------
cat('patient_nbr with multiple encounters = ',
    length(data$patient_nbr[duplicated(data$patient_nbr)]),'\n' )

cat('Before eliminating multiple encounters of a patient, total ',
    nrow(data), ' observations\n')

data <- data %>% arrange(desc(time_in_hospital))
data <- data[!duplicated(data$patient_nbr),]

cat('After eliminating multiple encounters of a patient, total ',
    nrow(data), ' observations\n')

# Now there should be no more multiple encounters of a patient.
cat('Multiple encounters of a patient now = ',
    length(data$patient_nbr[duplicated(data$patient_nbr)]),'\n' )

#--------------------
# DROPPING ID FIELDS
#--------------------
cat('Dropping the ID fields, ', names(data[1]),' and ', names(data[2]))
data <- data[-c(1,2)]

cat('At this time, the data set has ',nrow(data),' observations with ', ncol(data),' variables.')
# 27 variables

#-------------------------------------------
# CONSOLIDATING LEVELS OF A FACTOR VARIABLE
#-------------------------------------------
# Gender
data$gender <- factor(ifelse(data$gender=='Female','Female','Male'))
cat('"gender" is now ', class(data$gender) ,' with the levels: ',levels(data$gender))

# Age
data$age <- as.character(data$age)
data$age[data$age %in% c('[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)')]  <- '1'
data$age[data$age %in% c('[60-70)','[70-80)')] <- '1.5'
data$age[data$age %in% c('[80-90)','[90-100)')] <- '2'
data$age <- as.numeric(as.character(data$age))

cat('Considering those older than 60 are twice more likely to be readmitted.\n')
cat('"age" is now ', class(data$age) ,' with the unique values: ',unique(data$age),
    '\n\twhere age<60 is assigned as 1, 60<=age<80 1.5, and age>80 as 2')

# Admission Types
#unique(data$admission_type_id)
data$admission_type_id <- factor(ifelse(data$admission_type_id%in%c(5,6,8),'u','k'))
cat('"admission_type_id" is now ',class(data$admission_type_id),
    ' with levels: ', levels(data$admission_type_id),
    '\n\twhere u: unknow, k: known\n')

# Admission Sources
#unique(data$admission_source_id)
data$admission_source_id <- factor(ifelse(data$admission_source_id%in%c(1,2,3,19),'r',
  (ifelse(data$admission_source_id%in%c(4,5,6,10,18,22,25,26),'t',
    (ifelse(data$admission_source_id%in%c(9,15,17,20,21),'u',
      (ifelse(data$admission_source_id%in%c(7,8),'o',
        (ifelse(data$admission_source_id%in%c(11,12,13,14,23,24),'b',NA))))))))))
cat('"admission_source_id" is now ',class(data$admission_source_id),
    ' with levels: ', levels(data$admission_source_id),
    '\n\twhere r: referral, t: transfer, u: unknown, o: other, b: birth')

# Disposition Ids
# Removing disposition code as 'Expired', since not relevant to readmission
cat('Before removing "discharge_disposition_id" of 11,19,20,21, \nthere were ',nrow(data),
    ' observations with unique ids: \n',unique(data$discharge_disposition_id))

data <- data[!(data$discharge_disposition_id %in% c(11,19,20,21)),]

cat('After removing "discharge_disposition_id" of 11,19,20,21,\nthere were ',nrow(data),
    ' observations with unique ids: \n',unique(data$discharge_disposition_id))

#unique(data$discharge_disposition_id)
data$discharge_disposition_id <- factor(
  ifelse(data$discharge_disposition_id%in%c(1,2,3,4,5,6,7,8,22,23,24),'d',
    (ifelse(data$discharge_disposition_id%in%c(9,10,12,10),'o',
        (ifelse(data$discharge_disposition_id%in%c(13,14),'h',
            (ifelse(data$discharge_disposition_id%in%c(18,25),'u',NA))))))) )
cat('"discharge_disposition_id)" is now ',class(data$discharge_disposition_id),
    ' with levels: ', levels(data$discharge_disposition_id),
    '\n\twhere d: discharge, h: hospice, u: unknown, o: other')

# DIAGNOSTIC INFORMATION
# Converting diag_1, diag_2, and diag_3 into 9 Categories, per Table 2
# Ref: https://www.hindawi.com/journals/bmri/2014/781670/tab2/

### diag_1

cat('*** diag_1 with ', length(levels(data$diag_1)), ' levels\n')

data$diag_1 <- as.character(data$diag_1)
data$diag_1[grep(c('V'),data$diag_1)] <- '290'
data$diag_1[grep(c('E'),data$diag_1)] <- '290'

data$diag_1 <- as.numeric(data$diag_1)

data$diag_1 <- factor(
  ifelse( data$diag_1 %in% c(390:459,785),'Circulatory',
  (ifelse(data$diag_1 %in% c(460:519,786),'Respiratory',
  (ifelse(data$diag_1 %in% c(520:579,787),'Digestive',
  (ifelse(floor(data$diag_1/250)==1,'Diabetes',
  (ifelse(data$diag_1 %in% c(800:999),'Injury',
  (ifelse(data$diag_1 %in% c(710:739),'Musculature',
  (ifelse(data$diag_1 %in% c(580:629,788),'Genitourinary',
  (ifelse(data$diag_1 %in%  c(140:239,780,781,784,790:799,240:249,251:279,680:709,782,001:139),'Neoplasms',
  (ifelse(data$diag_1 %in% c(290:319,280:289,320:359,630:679,360:389,740:759),'Other',NA
  ))))))))))))))))))

l1 <- levels(data$diag_1)
cat('*** diag_1 is now converted to ', length(l1), ' levels as the following: \n', l1)

### diag_2

cat('*** diag_2 with ', length(levels(data$diag_2)), ' levels\n')

data$diag_2 <- as.character(data$diag_2)
data$diag_2[grep(c('V'),data$diag_2)] <- '290'
data$diag_2[grep(c('E'),data$diag_2)] <- '290'

data$diag_2 <- as.numeric(data$diag_2)

data$diag_2 <- factor(
  ifelse( data$diag_2 %in% c(390:459,785),'Circulatory',
  (ifelse(data$diag_2 %in% c(460:519,786),'Respiratory',
  (ifelse(data$diag_2 %in% c(520:579,787),'Digestive',
  (ifelse(floor(data$diag_2/250)==1,'Diabetes',
  (ifelse(data$diag_2 %in% c(800:999),'Injury',
  (ifelse(data$diag_2 %in% c(710:739),'Musculature',
  (ifelse(data$diag_2 %in% c(580:629,788),'Genitourinary',
  (ifelse(data$diag_2 %in%  c(140:239,780,781,784,790:799,240:249,251:279,680:709,782,001:139),'Neoplasms',
  (ifelse(data$diag_2 %in% c(290:319,280:289,320:359,630:679,360:389,740:759),'Other',NA
  ))))))))))))))))))

l2 <- levels(data$diag_2)
cat('*** diag_2 is now converted to ', length(l2), ' levels as the following:\n', l2)

### diag_3

cat('*** diag_3 with ', length(levels(data$diag_3)), ' levels')

data$diag_3 <- as.character(data$diag_3)
data$diag_3[grep(c('V'),data$diag_3)] <- '290'
data$diag_3[grep(c('E'),data$diag_3)] <- '290'

data$diag_3 <- as.numeric(data$diag_3)

data$diag_3 <- factor(
  ifelse( data$diag_3 %in% c(390:459,785),'Circulatory',
  (ifelse(data$diag_3 %in% c(460:519,786),'Respiratory',
  (ifelse(data$diag_3 %in% c(520:579,787),'Digestive',
  (ifelse(floor(data$diag_3/250)==1,'Diabetes',
  (ifelse(data$diag_3 %in% c(800:999),'Injury',
  (ifelse(data$diag_3 %in% c(710:739),'Musculature',
  (ifelse(data$diag_3 %in% c(580:629,788),'Genitourinary',
  (ifelse(data$diag_3 %in%  c(140:239,780,781,784,790:799,240:249,251:279,680:709,782,001:139),'Neoplasms',
  (ifelse(data$diag_3 %in% c(290:319,280:289,320:359,630:679,360:389,740:759),'Other',NA
  ))))))))))))))))))

l3 <- levels(data$diag_3)
cat('*** diag_3 is now converted to ', length(l3), ' levels as the following: \n', l3)

### Response Variable

# Reduced to 2 levels for classification and converted to numeric 0 and 1
#data$readmitted[15000:15005]
cat('"readmitted" levels: ', levels(data$readmitted))
data$readmitted <- as.numeric(factor(ifelse(data$readmitted=='<30','yes','no')))-1
#data$readmitted[15000:15005]
cat('"readmitted" is now a ',class(data$readmitted),
    ' with unique values: ', unique(data$readmitted))

cat('At this time, the data set has ',nrow(data),' observations with ', ncol(data),' variables.')

#------------------------------------
# Show variabels based on data types
#------------------------------------
var <- sapply(names(data), function(x){ class(data[[x]]) })
data.type <- lapply(unique(var), function(x){names(var[var==x])})
names(data.type) <- unique(var)

summary(data.type)[,1]
cat('Total ',(total <- sum(as.integer(summary(data.type)[,1]))),' variables')

summary(data[data.type$integer])

#-------------------
# NUMERIC VARIABLES
#-------------------

notUsed <- function(){
# Number of outliers
outliers <- function(x,n){
  row <- sum( x > n*IQR(x) )
  percentage <- round( row*100/length(x) , 2)
  rbind(row,percentage)
}

cat('Outliers as > ', (numIQR <- 3) ,
    'x IQR with rows and percentage of a variable')
sapply(data.type$integer, function(x){outliers(data[,x],numIQR)})

cat('Outliers as > ', (numIQR <- 4) ,
    'x IQR with rows and percentage of a variable')
sapply(data.type$integer, function(x){outliers(data[,x],numIQR)})

cat('Removing outliers with values > ',3,' x IQR')
for (i in data.type$integer){
  data <- data[ data[,i] <= 3*IQR( data[,i] ), ]
}

cat('Removing outliers in number_emergency and number_outpatient',
    '\nfor those with values > 40 and > 29, respectively')
data <- data[ data[,'number_emergency'] < 40, ]
data <- data[ data[,'number_outpatient'] < 29, ]
}
## Centering & Normalization
cat('Centering and normalizing the ',length(data.type$integer),'integer variables')
data[data.type$integer] <- scale(data[data.type$integer])
head(data[data.type$integer])
summary(data[data.type$integer])

#---------------
# VISUALIZATION
#---------------
par(mfrow=c(ceiling(length(data.type$integer)/3),3),
    mar=c(3,2,3,2))
sapply(data.type$integer,function(x){
  hist(data[,x],las=1,col='lightblue',main=x,xlab='',ylab='')})
par(mfrow=c(1,1))

# For demonstratoin/documentation, load instead of generating plots
if (!require('EBImage')) install.packages("BiocManager");BiocManager::install("EBImage");library(EBImage)
img = readImage('capstone.pairs.panels.70k.keeper.jpeg')
display(img, method = "raster")
img = readImage('capstone.corrplot.70k.keeper.jpeg')
display(img, method = "raster")

# If to plot in real-time, pairs will take 30+ minutes
notRun <- function(){
pairs.panels(data[data.type$integer],las=3, pch=23, hist.col='lightblue',
             cex.cor=4,stars=TRUE,lm=TRUE)

corrplot.mixed(cor(data[data.type$integer]),outline=FALSE)
}

#------------
# IMPUTATION
#------------
# The missing values were in age, diag_1, diag_2 and diag_2
# ref: https://www.r-bloggers.com/imputing-missing-data-with-r-mice-package/
#library(mice)

cat(sum(is.na(data[,data.type$integer])),
    ' missing values of all numeric variables\n')
cat(sum(is.na(data[,data.type$factor])),
    ' missing valuesof all factor variables')

imp <- mice(data,m=1,method='cart',maxit=3,seed=111)

dataimp <- complete(imp,1);write.csv(dataimp,'data/capstone.dataimp.csv')

img = readImage('capstone.imp.stripplot.keeper.jpeg')
display(img, method = "raster")

#stripplot(imp, las=1, cex=0.5,pch=23, col='lightblue', main='Imputation')

#---------------
# SPLITING DATA
#---------------
dataimp <- read.csv('data/capstone.dataimp.csv')
part <- sample(2,nrow(dataimp),replace=TRUE,prob=c(0.7,0.3))

train <- dataimp[part==1,]; write.csv(train, 'data/capstone.train.csv')
test  <- dataimp[part==2,]; write.csv(test,  'data/capstone.test.csv' )

#-------------------
# FEATURE SELECTION
#-------------------

if (!require('Boruta')) install.packages('Boruta'); library(Boruta)

boruta.input <- read.csv('data/capstone.train.csv', header=TRUE)[-c(1,2)]
# Removed the id field generated by previous imports.

set.seed(5-1)
boruta.output <- Boruta(readmitted~., data=boruta.input, doTrace=2, maxRuns=100)

saveRDS(boruta.output, 'capstone.boruta100.ouput.rds')
print(boruta.output)

boruta.fix <- TentativeRoughFix(boruta.output)

saveRDS(boruta.output, 'capstone.boruta100.fix.rds')
str(boruta.fix)

#attStats(boruta.fix)

print(
  boruta.selected <- getSelectedAttributes(boruta.fix, withTentative = FALSE)
);
write.file(boruta.selected,'data/capstone.boruta100.txt')


# For demonstratoin/documentation, load instead of generating plots
if (!require('EBImage')) install.packages("BiocManager");BiocManager::install("EBImage");library(EBImage)
img = readImage('capstone.boruta.run.100.keeper.jpeg')
display(img, method = "raster")
#plot(boruta.output, las=2, cex.axis=0.7, xlab='', main='Feature Importance')
#

dataimp.boruta <- read.csv('data/capstone.dataimp.csv')[c(boruta.selected,'readmitted')]
write.csv(dataimp.boruta,'data/dataimp.boruta.csv') # This is the prepared data for modeling.

# Finalized data set
dib <-read.csv('data/dataimp.boruta.csv')[-1]
str(dib)
summary(dib)
