#Prediction - R Code < author: Ajay Gautam M >

#RANDOM FOREST 

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





