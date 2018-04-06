#Prediction - R Code < author: Ajay Gautam M >

#SVM

set.seed(123)

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

