
# Algorithms Used

Four different classification algorithms are used in this project to predict the term deposit subscription.
1.	**Linear algorithm:** Logistic Regression (GLM)
2. 	**Non-Linear algorithm:** Support Vector Machine (SVM)
3. 	**Bagging algorithm:** Random Forest (RF)
4. 	**Boosting algorithm:** eXtreme Gradient Boosting (XGBoost)


# Model Evalutaion and Selection:

## Performance Metrics:

The ***test accuracy is not a very good indicator*** of model performance in the case of imbalanced dataset. The following metrics were considered in choosing the model.
1.	**Confusion Matrix:** A breakdown of predictions into a table showing correct predictions (the diagonal) and the types of incorrect predictions made (what classes incorrect predictions were assigned).

2.	**Precision:** A measure of  a classifiers exactness.

3.	**Recall:** A measure of a classifiers completeness

4.	**F1 Score (or F-score):** A weighted average of precision and recall.

5.	**ROC Curves:** Like precision and recall, accuracy is divided into sensitivity and specificity and models are chosen based on the balance thresholds of these values.


## Comparison Table:

The following table shows the performance of the chosen models across various metrics

Table             | Logistic | RF | SVM | XGBoost
-----------       | -----    |----| ----| ------- 
**AUC**           | 0.7031   | 0.7108| 0.6978| **0.7115**
**F1-Score**      | 0.4430 | **0.4723** | 0.4532 | 0.4720
**Test Accuracy** | 0.8571 | **0.8741** | 0.8669 | 0.8734

It can be seen that **Random Forest** and **XGBoost**  are the top performing algorithms. Even though the differences in the metric values are infinitesimal, we will choose **Random Forest** as the ***best model*** for our prediction because it ***ranks first in 2 out of the 3 considered metrics .***


### Note:
Complete **R Code** for each model is provided separately in this folder. 



















