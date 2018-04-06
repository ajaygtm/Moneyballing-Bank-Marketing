#Prediction - R Code < author: Ajay Gautam M >

#XGBOOST

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
