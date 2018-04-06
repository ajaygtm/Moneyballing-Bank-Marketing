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

Four different algorithms are to predict the term deposit subscription.
1.	**Linear algorithm:** Logistic Regression (GLM)
2. 	**Non-Linear algorithm:** Support Vector Machine (SVM)
3. 	**Bagging algorithm:** Random Forest (RF)
4. 	**Boosting algorithm:** eXtreme Gradient Boosting (XGBoost)

## Model Comparision 

Since, the test accuracy cannot be used as a sole indicator of the model performance in case of imbalanced dataset, metrics like **AUC**, **F1-Score** and **Confusion Matrix** are used for comparing the models prediction efficiency.

Table             | Logistic | RF | SVM | XGBoost
-----------       | -----    |----| ----| ------- 
**AUC**           | 0.7031   | 0.7108| 0.6978| 0.7115
**F1-Score**      | 0.4430 | 0.4723 | 0.4532 | 0.4720
**Test Accuracy** | 0.8571 | 0.8741 | 0.8669 | 0.8734

It can be seen that **Random Forest** and **XGBoost**  are the best performing algorithms. Even though the difference in the metrics value is infinitesimal, we will choose **Random Forest** as the best model for our prediction because it ranks first in 2 out of 3 considered metrics .

The predictions for the test data set are stored in the **result.csv** file which is available in the data folder. 








