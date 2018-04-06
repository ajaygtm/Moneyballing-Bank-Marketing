#Prediction - R Code < author: Ajay Gautam M >

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
