
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


### Note: 
Complete **R code** used for the above feature engineering is provided in this folder.

















































