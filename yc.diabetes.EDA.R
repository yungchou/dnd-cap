library(psych); library(dplyr)

#----------
# DATA SET
#----------

# Diabetes 130-US hospitals for years 1999-2008 Data Set
# https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
# Ref: Feature description
# https://www.hindawi.com/journals/bmri/2014/781670/

data <- read.csv('dataset_diabetes/diabetic_data.csv', header=TRUE)
#ref  <- read.csv('dataset_diabetes/IDs_mapping.csv'  , header=TRUE)

str(data) # 101766 obs. of  50 variables

# Dropping ID fields
data <-data[-1] # here 49 variables

summary(data)

#-------------
# MISSINGNESS
#-------------

#Checking NA
sapply(colnames(data), function(x){ sum(is.na(data[x])) })

# Checking those having a question mark as the value
sapply(colnames(data), function(x){ sum(data[x]=='?') })

# Considering '?' as missing values
sum(data=='?')
data[data=='?'] <- NA
sum(is.na(data))

library(naniar);gg_miss_upset(data)

#colSums(is.na(data))
# Shown as percentage missing
q <- sapply(colnames(data), function(x){ round(sum(is.na(data[x]))*100/nrow(data[x]),2) })

# Variables with percentage of missing values > 0
q[q>0]

barplot( log(q[q>0]+1),
      las=2, col=grey.colors(10), cex.names = 0.8,
      main='Missingness', ylab='log(% missing)', xlab='' )
abline(h=log(30+1), col='red',lwd=2)

# Taking variables with less than 30% values missing
data <- data[ names(data) %in% names(q[q<30]) ] # here 45 variables

# Patient with multiple encounters
cat('patient_nbr with multiple encounters = ',
    length(data$patient_nbr[duplicated(data$patient_nbr)]) )

data <- data %>% arrange(desc(time_in_hospital))
data <- data[!duplicated(data$patient_nbr),]

# Eliminated suplicated and left with one na donly the one
# with the longest time_in_hospital
cat('patient_nbr duplicates now = ',
    length(data$patient_nbr[duplicated(data$patient_nbr)]) )

cat('Total observations now = ', nrow(data),
    '\nVariables list = ')
names( q[ q<30 ] )

#---------
# FACTORS
#---------

order(unique(data$admission_type_id))         # all valid per IDs_mappings.csv
order(unique(data$discharge_disposition_id))  # all valid per IDs_mappings.csv
order(unique(data$admission_source_id))       # all valid per IDs_mappings.csv

# Removing disposition code as 'Expired'
data <- data[data$discharge_disposition_id[
  !(data$discharge_disposition_id %in% c(11,19,20,21))
  ],]

data$admission_type_id        <- factor(data$admission_type_id)
data$admission_source_id      <- factor(data$admission_source_id)
data$discharge_disposition_id <- factor(data$discharge_disposition_id)

# Group variabels based on data types
var <- sapply(names(data), function(x){ class(data[[x]]) } )
data.type <- lapply(unique(var), function(x){names(var[var==x])})
names(data.type) <- unique(var)

# List out unique values of factor variables
f <- sapply(data.type$factor, function(x){unique(data[x])})

# Each of these three variables had 700+ levels. This would
# generate 2100+ dummy variables.
cat('*** diag_1 got ', length(levels(data$diag_1)), ' levels')
cat('*** diag_2 got ', length(levels(data$diag_2)), ' levels')
cat('*** diag_3 got ', length(levels(data$diag_3)), ' levels')

# Diagnosis 1	Nominal	The primary diagnosis coded as first three
# digits of ICD9 with 848 distinct values
# Diagnosis 2	Nominal	Secondary diagnosis coded as first three digits
# of ICD9 with 923 distinct values
# Diagnosis 3	Nominal	Additional secondary diagnosis coded as first
# three digits of ICD9 with 954 distinct values	with 1% missing values


# The following two factors has only one level with no missing
# value. Therefore these two variables have zero-variance,
# are not influential on predicting readmission and removed.
f[27:28]

sum(is.na(data$examide))
sum(is.na(data$citoglipton))

rm <- c('examide', 'citoglipton')
data <- data[ !(names(data) %in% rm) ]
names(data)

#-------------------
# NUMERIC VARIABLES
#-------------------

# logarithm, followed by standardization
data[data.type$integer] <- scale(log(data[data.type$integer]+1))

# Plot histograms of all 8 numberic variables
par(mfrow=c(ceiling(length(data.type$integer)/3),3))
for (i in data.type$integer){
  thisOne <- eval(parse(text=paste0('data$',i)))
  hist(
    thisOne,
    las=1, col='lightblue', freq=FALSE,
    main=i, xlab='', ylab=''
  )
  lines(density(thisOne),col="red",lwd=2)
}
par(mfrow=c(1,1))

# NA recheck
sapply(data.type$integer, function(x){sum(is.na(data[x]))})

#-------------------
# MULTICOLLINEARITY
#-------------------

# Total obsrvations are more then 100K and will take
# a long while to plot. Here for demonstration, plot
# only the first 10K or 20K observations.
these <- data[1:10000,data.type$integer]
pairs.panels( these, hist.col = "lightblue",
              las=1, cex.cor=1, lm=TRUE, stars=TRUE)

# From a plot with all 100k observations, most correlation
# coefficients were low and raised little concern.
# There were two pairs with coefficients higher the 0.3:
# time_in_hospital and num_medications (0.46)
#  num_procedures and num_medications (0.35)

#-----------------------
# CATEGORICAL VARIABLES
#-----------------------

head(data$age)
# Converting to numberic with 4 levels
data$age <- as.character(data$age)
data$age[data$age %in% c('[0-10)','[10-20)')]             <- 'youth'
data$age[data$age %in% c('[20-30)','[30-40)','[40-50)')]  <- 'adult'
data$age[data$age %in% c('[50-60)','[60-70)')]            <- 'senior'
data$age[data$age %in% c('[70-80)','[80-90)','[90-100)')] <- 'senior2'
data$age <- as.numeric(factor(data$age,levels=c('youth','adult','senior','senior2')))
head(data$age)

# Reduced to 2 levels
#data$readmitted[15000:15005]
data$readmitted <- as.character(data$readmitted)
data$readmitted <- ifelse(data$readmitted=='<30','yes','no')
data$readmitted <- factor(data$readmitted)
#data$readmitted[15000:15005]

#------------------
# Forming criteria
#------------------
str(data)

n <- scale(data[var.type$integer])
summary(n)
str(n)
library(psych)
# For demonstration, plotting only the first 10,000 obs
pairs.panels(n[1:10000], las=1, cex.cor=0.7, stars=TRUE, lm=TRUE)
