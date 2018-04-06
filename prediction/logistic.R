#Prediction - R Code < author: Ajay Gautam M >

#LOGISTIC REGRESSION

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




