#feature engineering - R Code   < author: Ajay Gautam M >

#FEATURE ENGINEERING

#Feature Binning - Age Variable
train$age <- cut(train$age, c(0,19,35,60,100), labels = c("Teens","Young Adults", "Adults", "Senior Citizens"))
test$age <- cut(test$age, c(0,19,35,60,100), labels = c("Teens","Young Adults", "Adults", "Senior Citizens"))

#Feature Selection

#1
#Checking the predictor variables that are highly correlated with each other
#Two metrics are used - Correlation factor and VIF

#Correlation Factor
bank_cor <- subset(train, select=-c(y))
for(i in 1:ncol(bank_cor)){bank_cor[,i]<- as.integer(bank_cor[,i])} #Changing the variables into integer
correlationMatrix <- cor(bank_cor) #Correlation matrix
#Finding attributes that are highly correlated (>0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75, names=TRUE, verbose = TRUE)
print(highlyCorrelated) #Result -> 'euribor3m' and 'emp.var.rate'

#VIF Factor
#Defining a custom VIF Function
vif_func<-function(in_frame,thresh=10,trace=T,...){
  
  library(fmsb)
  
  if(any(!'data.frame' %in% class(in_frame))) in_frame<-data.frame(in_frame)
  
  #get initial vif value for all comparisons of variables
  vif_init<-NULL
  var_names <- names(in_frame)
  for(val in var_names){
    regressors <- var_names[-which(var_names == val)]
    form <- paste(regressors, collapse = '+')
    form_in <- formula(paste(val, '~', form))
    vif_init<-rbind(vif_init, c(val, VIF(lm(form_in, data = in_frame, ...))))
  }
  vif_max<-max(as.numeric(vif_init[,2]), na.rm = TRUE)
  
  if(vif_max < thresh){
    if(trace==T){ #print output of each iteration
      prmatrix(vif_init,collab=c('var','vif'),rowlab=rep('',nrow(vif_init)),quote=F)
      cat('\n')
      cat(paste('All variables have VIF < ', thresh,', max VIF ',round(vif_max,2), sep=''),'\n\n')
    }
    return(var_names)
  }
  else{
    
    in_dat<-in_frame
    
    #backwards selection of explanatory variables, stops when all VIF values are below 'thresh'
    while(vif_max >= thresh){
      
      vif_vals<-NULL
      var_names <- names(in_dat)
      
      for(val in var_names){
        regressors <- var_names[-which(var_names == val)]
        form <- paste(regressors, collapse = '+')
        form_in <- formula(paste(val, '~', form))
        vif_add<-VIF(lm(form_in, data = in_dat, ...))
        vif_vals<-rbind(vif_vals,c(val,vif_add))
      }
      max_row<-which(vif_vals[,2] == max(as.numeric(vif_vals[,2]), na.rm = TRUE))[1]
      
      vif_max<-as.numeric(vif_vals[max_row,2])
      
      if(vif_max<thresh) break
      
      if(trace==T){ #print output of each iteration
        prmatrix(vif_vals,collab=c('var','vif'),rowlab=rep('',nrow(vif_vals)),quote=F)
        cat('\n')
        cat('removed: ',vif_vals[max_row,1],vif_max,'\n\n')
        flush.console()
      }
      
      in_dat<-in_dat[,!names(in_dat) %in% vif_vals[max_row,1]]
      
    }
    
    return(names(in_dat))
    
  }
  
}

#Checking the variables that have VIF value greater than 10
vif_func(in_frame=bank_cor,thresh=10,trace=T) #Result - 'euribor3m'

#From the above 2 metrics- it is decided to remove the 'euribor3m' variable
train <- subset(train, select = -c(euribor3m))
test <- subset(test, select = -c(euribor3m))

#2

#Dataset Author's note on the 'duration' variable
#Note : this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

#So based on the above note, 'duration' variable is removed from the model
#Since, it won't be help in building a predictive model

train <- subset(train, select = -c(duration))
test <- subset(test, select = -c(duration))


#3
#pdays variable is also removed because it is not available for test data.
train <- subset(train, select = -c(pdays))

#############################################################################################

#PRE-PROCESSING:

#Splitting the training data into train and validation set
#CreateDataPartition from the 'caret'package is used. Because it will maintain same class distribution in the resulting datasets as same as in the original data while splitting
set.seed(123)
i <- createDataPartition(train$y, p = 3/4,list = FALSE)
new_train_pre <- train[i,]
new_test <- train[-i,]

#Dealing with Imbalanced Data
#Synthetic Data generation method is used to balance the classes
#Specifically, SMOTE Technique is used
new_train <-SMOTE(y~.,new_train_pre,perc.over = 400, perc.under = 150,k=5)
train_smote <- SMOTE(y~.,train, perc.over = 400, perc.under = 150, k=5)

#checking the proportion of classes before and after SMOTE
prop.table(table(new_train_pre$y)) # No-89%, Yes-11%
prop.table(table(new_train$y)) #No-55%, Yes-45%

prop.table(table(train$y)) # No-89%, Yes-11%
prop.table(table(train_smote$y)) #No-55%, Yes-45%

