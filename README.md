# Moneyballing-Bank-Marketing

A European Banking Institution has provided data related to direct marketing campaigns. The campaigns were based on phone calls to their customers in order to offer **term deposit subscriptions**. In this project, we will build classification models to predict *whether the client will subscribe to a term deposit or not* and also to identify *the main factors that affect the clients’ decision.*

## Data Source

The data used for this project is publicly available on UCI Machine Learning repository - [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing).
Further details about the dataset is provided in the README file in the data folder.

## Analysis

**Data Cleaning** and an extensive **Exploratory Data Analysis** is done on the dataset. The plots and the inferences can be found in the analysis folder.

## Feature Engineering

**Feature Selection** is done to select the best variables for prediction. 
A major problem associated with the given dataset is that it is *imbalanced.* So, **SMOTE** - a Synthetic Data Generation sampling method is used to overcome this problem. 

## Algorithms Used

Four different algorithms are used to predict the term deposit subscription.
1.	**Linear algorithm:** Logistic Regression (GLM)
2. 	**Non-Linear algorithm:** Support Vector Machine (SVM)
3. 	**Bagging algorithm:** Random Forest (RF)
4. 	**Boosting algorithm:** eXtreme Gradient Boosting (XGBoost)

The algorithms were assessed using test accuracy, AUC and F1 Score. 
**Random forest** performed the best among the four algorithms considered with **87.41% test accuracy, 71.08% AUC and 0.4723 F1 Score**

## Result

The prediction result on the test data using the **Random Forest algorithm** is stored in the **result.csv** file. It is avaiable in the data folder.




