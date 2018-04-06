
## Feature Binning

 **Age** variable is binned into 4 buckets.
 1. Teens
 2. Teen Adults
 3. Adults
 4. Senior Citizens
 
 ## Feature Selection
 
#### 1. euribor3m’ ( Euribor 3 month rate):
This variable is **removed** since it was highly correlated with other variables leading to severe multi-colinearity. Metrics used to test the multi-colinearity were **correlation factor** and **Variance Inflation Factor (VIF).**

#### 2. duration:
This variable is **removed** even though it was a strong predictor based on the following dataset author’s notes. ***Note**: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.*

#### 3. pdays:
This variable is **removed** because it has constant value and also the given test dataset doesn’t have that attribute. So, it won’t be helpful in predicting.



The remaining attributes are properly transformed and then taken into account for modelling.

## Imbalanced Dataset:
A dataset is **imbalanced** if the distribution of output classes is not uniform. Here, we can see that class 1 (‘not subscribed’) has 89% of instances but class 2 (‘subscribed’) has only 11% of instances.   

### Problems due to Imbalanced Dataset:
1. The algorithms will be biased towards majority class. So it will predict the majority class irrespective of the predictors.
2. The ‘Test Accuracy” of the model will be more than 80% but it won’t serve the objective of our prediction. Since, the test data will also have more than 80% of the data pertaining to majority class. So, here, the accuracy won’t be a good representation of the model performance.
3. The algorithm assumes that errors obtained from both classes have same cost. But in this case, the Type II error (False Negative) error is very serious than the Type I error (False Positive). Because we should not miss the chance or opportunity of identifying a potential customer. But we can afford to get a few false positive, because comparatively it won’t do much harm.

### Methods used to deal with Imbalanced Dataset:
#### Sampling: 
This method aims to modify an imbalanced data into balanced distribution using some mechanism. This modification occurs by altering the size of original data set and provides the same proportion of balance. For this project, *Synthetic Data Generation method* is used. In regards to synthetic data generation, **Synthetic Minority Oversampling TEchnique” (SMOTE)** - a powerful and widely used method is applied. SMOTE algorithm creates artificial data based on feature space (rather than data space) similarities from minority samples. It generates a random set of minority class observations to shift the classifier learning bias towards minority class. To generate artificial data, it uses *bootstrapping* and *k-nearest neighbours.*
 
#### Performance Metric:
Since, the test accuracy is not a very good indicator of model performance in the case of imbalanced dataset. The following metrics were considered in choosing the model.
1.	**Confusion Matrix:** A breakdown of predictions into a table showing correct predictions (the diagonal) and the types of incorrect predictions made (what classes incorrect predictions were assigned).

2.	**Precision:** A measure of  a classifiers exactness.

3.	**Recall:** A measure of a classifiers completeness

4.	**F1 Score (or F-score):** A weighted average of precision and recall.

5.	**ROC Curves:** Like precision and recall, accuracy is divided into sensitivity and specificity and models are chosen based on the balance thresholds of these values.

#### Cost Sensitive Learning:

This deals with the cost associated with misclassifying observations. We are more concerned about **false positives** and **false negatives.** There is no cost penalty associated with True Positive and True Negatives as they are correctly identified. In our case, we are giving **more penalty to the False Negative than the False Positive.** The reason is, the bank can’t afford to lose a customer who is willing to subscribe to term deposit. Losing a *sure customer* will be a severe problem to the marketing campaign. But it can afford some false positives, since the money and work associated with those false positives is not a serious problem comparitively.



### Note: 
Complete **R code** used for the above feature engineering is provided in this folder.

















































