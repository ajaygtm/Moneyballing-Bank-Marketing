#Complete R Scirpt < author: Ajay Gautam M>

#Setting the working directory
getwd()
setwd('.../data')

#Required libraries
library(mice)
library(ggplot2)
library(ggthemes)
library(Hmisc)
library(moments)
library(car)
library(fmsb)
library(corrplot)
library(psych)
library(DMwR)
library(caret)
library(ROCR)
library(randomForest)
library(e1071)
library(data.table)
library(mlr)
library(xgboost)
library(parallel)
library(parallelMap)
library(VIM)


#Importing the dataset
train = read.csv("bank.csv", na.strings = "")
test = read.csv("bank-additional.csv", na.strings = "")

#Initial look at the data
names(train)
summary(train)
str(train)

##############################################################################################

#DATA CLEANING
#Checking for the missing values
sum(is.na(train)) 
sum(is.na(test))
sapply(train, FUN=function(x) sum(is.na(x)))  
sapply(test, FUN=function(x) sum(is.na(x)))   
#Train data - lots of missing values
#Test data - no missing value

#Imputing the missing values
#Method - MICE (Multivariate Imputation by Chained Equation) - 'mice' package
init = mice(train, maxit = 0)
predM = init$predictorMatrix #Predictor Matrix
meth = init$method #Setting the methods for each type of variable

imp = mice(train, m=5, predictorMatrix = predM, method = meth)
train = complete(imp) #Combining the imputed data with the original

##############################################################################################

#EDA.
#Generating plots to visually get some insigths about the variables

#Theme Setting
theme_set(theme_fivethirtyeight())
barfill <- "#4271AE"
barlines <- "#1F3552"    

#Missing Vales Plot
aggr_plot <- aggr(train, col=c('slateblue','red'), numbers=TRUE, prop=FALSE,
                  sortVars=TRUE, labels=names(train), cex.axis=0.7, gap=1,
                  varheight = FALSE,combined = FALSE,cex.numbers =0.5, 
                  ylab=c("Histogram of missing data","Pattern"))

#Age Variable
p1 <- ggplot(train, aes(x = age)) +
  geom_histogram(aes(fill = ..count..), binwidth = 1,
                 colour = barlines, fill = barfill)+
  scale_x_continuous(name = "Age",
                     breaks = seq(15,100,10),
                     limits = c(15,100)) +
  xlab("Age")+ylab("Count")+
  ggtitle("Frequency histogram of Age") +
  theme(plot.title = element_text(hjust = 0.5))
p1

p2 <- ggplot(train, aes(x = y, y =age)) +
  geom_boxplot(fill = "coral2", colour="firebrick4")+
  scale_y_continuous(name = "Age") +
  scale_x_discrete(name="Subscribed") +
  ggtitle("Boxplot of Age by Subscription") +
  theme(plot.title = element_text(hjust = 0.5))

p2

#Marital Vairable
p3 <- ggplot() + geom_bar(aes(y = (..count..), x = marital, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Marital Status Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p3

#Job Vairable
p4 <- ggplot() + geom_bar(aes(y = (..count..), x = job, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Job Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text.x =
          element_text(size  = 10,
                       angle = 45,
                       hjust = 1,
                       vjust = 1))
p4

#Education Vairable
p5 <- ggplot() + geom_bar(aes(y = (..count..), x = education, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Education Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text.x =
          element_text(size  = 10,
                       angle = 45,
                       hjust = 1,
                       vjust = 1))
p5

#Contact Variable
p6 <- ggplot() + geom_bar(aes(y = (..count..), x = contact, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Contact Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p6

#Default Variable
p7 <- ggplot() + geom_bar(aes(y = (..count..), x = default, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Default Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p7

#Housing Variable
p8 <- ggplot() + geom_bar(aes(y = (..count..), x = housing, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Housing Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p8

#Loan Variable
p9 <- ggplot() + geom_bar(aes(y = (..count..), x = loan, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Loan Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p9

#Campaign
p11 <- ggplot() + geom_histogram(aes(y = (..count..), x = campaign, fill = y ), data = train,
                                 binwidth=5, stat="count") +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  scale_x_continuous(breaks = seq(1,15,1),limits=c(0,15))+
  ggtitle("No.of Contacts Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))

p11

#Month Vairable
p12 <- ggplot() + geom_bar(aes(y = (..count..), x = month, fill = y), data = train,
                           stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Month Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p12


#Days of Week Vairable
p13 <- ggplot() + geom_bar(aes(y = (..count..), x = day_of_week, fill = y), data = train,
                           stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Day Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p13


#Duration Vairable
p14 <- ggplot() + geom_bar(aes(y = (..count..), x = duration, fill = y), data = train,
                           stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Duration Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p14

#ln_duration <- log(train$duration)
ln_duration <- log(1+(train$duration))
p24 <- ggplot() + geom_histogram(aes(x = ln_duration, fill = y), data = train,
                                 binwidth = 0.1 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Log Duration Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p24

#pdays Vairable
p16 <- ggplot() + geom_histogram(aes( x = pdays, fill = y), data = train,
                                 binwidth = 50 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("PCampaign Days Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p16
table(train$pdays)


#Previous
p17 <- ggplot() + geom_histogram(aes( x = previous, fill = y), data = train,
                                 binwidth = 1 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("PContact Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p17

#POutcome
p18 <- ggplot() + geom_bar(aes(y = (..count..), x = poutcome, fill = y), data = train,
                           stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("POutcome Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p18

#Employment variation Rate
p19 <- ggplot() + geom_histogram(aes( x = emp.var.rate, fill = y), data = train,
                                 binwidth = 1 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Emp.Var.Rate Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p19

#Consumer Price Index
p20 <- ggplot() + geom_histogram(aes( x = cons.price.idx, fill = y), data = train,
                                 binwidth = 1 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Cons.Price.Index Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p20

#Consumer Confidence Index
p21 <- ggplot() + geom_histogram(aes( x = cons.conf.idx, fill = y), data = train,
                                 binwidth =5 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Cons.Conf.Index Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p21

#Euro Interbank Offered Rates
p22 <- ggplot() + geom_histogram(aes( x = euribor3m, fill = y), data = train,
                                 binwidth =0.01 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Euro.Interbank.Rate Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p22

#No. of Employees
p23 <- ggplot() + geom_histogram(aes( x = nr.employed, fill = y), data = train,
                                 binwidth =100 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("No.of Employees Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p23

###############################################################################################

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

################################################################################################

#MODELLING - VARIOUS ALGORITHMS are USED BELOW
set.seed(123)

################################################################################################

#1 - Logistic Regression

#Model Execution
logit_model <- glm(y ~.,family=binomial(link = "logit"),data = new_train)

#Summary Statistics
summary(logit_model)
anova(logit_model, test="Chisq")

#Prediction
pred_log <- predict(logit_model,newdata=new_test,type='response')

#Creating prediction and performance parameter for tuning
pred <- prediction(pred_log, new_test$y)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)

#Choosing the optimum cutoff value 
#1-Based on Cost function
cost.perf = performance(pred, "cost", cost.fp = 1 , cost.fn = 3.5 ) #Type II error has given more cost because it is not affordable in our case
pred@cutoffs[[1]][which.min(cost.perf@y.values[[1]])]
#2-Based on ROC Curve
plot(perf)

#Optimum Cutoff Value
pred_log_res <- ifelse(pred_log > 0.571182,2,1)

#Confusion Matrix
confusionMatrix(pred_log_res, new_test$y, positive = "2", mode="prec_recall")

#ROC - Curve
pred_res <- prediction(pred_log_res, new_test$y)
perf_res <- performance(pred_res, measure = "tpr", x.measure = "fpr")
plot(perf_res)

#AUC - Metric
auc <- performance(pred_res, measure = "auc")
auc <- auc@y.values[[1]]
auc  

###################################################################################################

#2 - Random Forest

#Model Execution
rf_model<-randomForest(y ~.,data = new_train, importance=TRUE, ntree=2000)

#Summary Statistics
summary(rf_model)

#Variable Importance
varImpPlot(rf_model)

#Prediction
pred_rf<- predict(rf_model,new_test, type="prob")
pred_rf <- pred_rf[,2]

#Creating prediction and performance parameter for tuning
pred <- prediction(pred_rf, new_test$y)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)

#Choosing the optimum cutoff value 
#1-Based on Cost function
cost.perf = performance(pred, "cost", cost.fp = 1 , cost.fn = 2.7 ) #Type II error has given more cost because it is not affordable in our case
pred@cutoffs[[1]][which.min(cost.perf@y.values[[1]])]
#2-Based on ROC Curve
plot(perf)

#Optimum Cutoff Value
pred_rf_res <- ifelse(pred_rf > 0.3605,2,1)

#Confusion Matrix
confusionMatrix(pred_rf_res, new_test$y, positive = "2")

#ROC - Curve
pred_res <- prediction(pred_rf_res, new_test$y)
perf_res <- performance(pred_res, measure = "tpr", x.measure = "fpr")
plot(perf_res)

#AUC - Metric
auc <- performance(pred_res, measure = "auc")
auc <- auc@y.values[[1]]
auc  

#################################################################################################

#3 - Support Vector Machines (SVM)

#Model Execution
svm_model <- svm(y~., data=new_train, kernel="radial",  probability=TRUE)

#Tuning 
svm_tune <- tune.svm(y~., data = new_train, gamma = 10^(-2:2), cost= 1)

#Model with tuned parameters
svm_model <- svm(y~., data=new_train, kernel="radial",  probability=TRUE, cost= 1,gamma=0.02 )

#Summary Statistics
summary(svm_model)

#Prediction
pred_svm<- predict(svm_model,new_test, probability = TRUE)
pred_svm <- pred_svm$probabilities
pred_svm_prob <- attr(pred_svm, "probabilities")
pred_svm_prob <- pred_svm_prob[,2]

#Creating prediction and performance parameter for tuning
pred <- prediction(pred_svm_prob, new_test$y)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)

#Choosing the optimum cutoff value 
#1-Based on Cost function
cost.perf = performance(pred, "cost", cost.fp = 1 , cost.fn = 4 ) #Type II error has given more cost because it is not affordable in our case
pred@cutoffs[[1]][which.min(cost.perf@y.values[[1]])]
#2-Based on ROC Curve
plot(perf)

#Optimum Cutoff Value
pred_svm_res <- ifelse(pred_svm_prob > 0.4355,2,1)

#Confusion Matrix
confusionMatrix(pred_svm_res, new_test$y, positive = "2")

#ROC - Curve
pred_res <- prediction(pred_svm_res, new_test$y)
perf_res <- performance(pred_res, measure = "tpr", x.measure = "fpr")
plot(perf_res)

#AUC - Metric
auc <- performance(pred_res, measure = "auc")
auc <- auc@y.values[[1]]
auc  

#################################################################################################

#4 - eXtreme Gradient Boosting (XGBoost)

#Data Preprocessing

#Using one hot encoding 
labels <- new_train$y 
ts_label <- new_test$y
new_tr <- model.matrix(~.+0,data = new_train[,-17]) #Converting to matrix without target variable
new_ts <- model.matrix(~.+0,data = new_test[,-17])

#Converting factor to numeric 
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1

#Preparing Matrix for the model
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

#Default Parameter setting
params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0,
               max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

#Finding best iteration using Cross Validation
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 200, nfold = 5, 
                 showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)
#Best iteration - 63

#Model Execution with default parameters
xgb1 <- xgb.train (params = params, data = dtrain, watchlist = list(val=dtest,train=dtrain), 
                   nrounds = 63,maximize = F , eval_metric = "error")


#Using MLR Package to find Optimum Parameters - the following steps are done

#Converting characters to factors
fact_col <- colnames(new_train)[sapply(new_train,is.character)]

for(i in fact_col) set(new_train,j=i,value = factor(new_train[[i]]))
for (i in fact_col) set(new_test,j=i,value = factor(new_test[[i]]))

#Creating tasks
traintask <- makeClassifTask (data = new_train,target ="y")
testtask <- makeClassifTask (data = new_test,target = "y")

#One hot encoding
traintask <- createDummyFeatures (obj = traintask) 
testtask <- createDummyFeatures (obj = testtask)

#Creating learner
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

#Setting parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")),
                        makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                        makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.5,upper = 1),
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))
#Setting resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

#Search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)

#Setting parallel backend
parallelStartSocket(cpus = detectCores())

#Parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc,
                     measures = acc, par.set = params, control = ctrl, show.info = T)
mytune


#Now modelling is done with the obtained parameters

#Setting the Best Optimum Parameters obtained from the tuning
best_params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.1, gamma=0,
                    max_depth=9, min_child_weight=5.991561, subsample=0.5144212, colsample_bytree=0.80843439)

#Model Execution
xgb_model <- xgb.train (params = best_params, data = dtrain, watchlist = list(val=dtest,train=dtrain), 
                        nrounds = 63,maximize = F , eval_metric = "error")

#Prediction
pred_xgb <- predict(xgb_model,newdata=dtest)

#Creating prediction and performance parameter for tuning
pred <- prediction(pred_xgb, ts_label)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)

#Choosing the optimum cutoff value 
#1-Based on Cost function
cost.perf = performance(pred, "cost", cost.fp = 1 , cost.fn = 2.3 ) #Type II error has given more cost because it is not affordable in our case
pred@cutoffs[[1]][which.min(cost.perf@y.values[[1]])]
#2-Based on ROC Curve
plot(perf)

#Optimum Cutoff Value
pred_xgb_res <- ifelse(pred_xgb> 0.35,1,0)

#Confusion Matrix
confusionMatrix(pred_xgb_res, ts_label, positive ="1")

#ROC - Curve
pred_res <- prediction(pred_xgb_res, ts_label)
perf_res <- performance(pred_res, measure = "tpr", x.measure = "fpr")
plot(perf_res)

#AUC - Metric
auc <- performance(pred_res, measure = "auc")
auc <- auc@y.values[[1]]
auc

################################################################################################

#MODEL SELECTION 

#Even though XGBoost and Random Forest had same performance metrics. We are choosing RF.
#RF has high F1 and then Test Accuracy than XGBoost


#Prediction on the test data
final_pred_prob <- predict(rf_model, test, type ="prob")
final_pred_prob <- final_pred_prob[,2]

#Setting the optimum cutoff value
pred_rf_res <- ifelse(final_pred_prob > 0.3605,2,1)

#Storing result in a column
test$prediction <- pred_rf_res

#Checking the proportion of subscribed and not subscirbed in the training and the test data
prop.table(table(train$prediction))
prop.table(table(test$prediction))

#Writing it in a csv file
write.csv(test, file="result.csv", row.names=FALSE)

#################################################################################################


