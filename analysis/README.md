
# Data Cleaning

## Missing Values

Due to the data being collected through **phone call interviews,** many clients refused to provide some information. This missing data is not at random so it should be modelled. The missing values are marked as “unknown” in the data set.

![missing_values_plot](https://user-images.githubusercontent.com/16735822/38424599-33d3f3d0-39cf-11e8-908b-036db73fa978.png)


In the training dataset, there are 19.5% missing values in default, 4% missing values in education, and so on. We can also look at the histogram from the above figure which clearly depicts the influence of missing values in the attributes. There are no missing values in the test data.

## Imputation

There are a number of different methods to handle missing data. Since the attributes with missing values are dichotomous and also categorical with more than two factors, we will use **Multivariate Imputation by Chained Equations (MICE)** with *logistic and polytomous logistic regression methods* for dichotomous and multiclass categorical variables respectively.


# Exploratory Data Analysis

Following plots are generated between different pairs of variables for an extensive EDA.

![plot1](https://user-images.githubusercontent.com/16735822/38424887-11f487d8-39d0-11e8-8174-be52296c0292.png)
![plot2](https://user-images.githubusercontent.com/16735822/38424890-12460f68-39d0-11e8-9c27-2c8fe84886ae.png)
![plot2a](https://user-images.githubusercontent.com/16735822/38424891-129afe9c-39d0-11e8-8889-91e3a3b805c9.png)
![plot3](https://user-images.githubusercontent.com/16735822/38424892-12f3f812-39d0-11e8-9de1-a2262dbd600d.png)
![plot4](https://user-images.githubusercontent.com/16735822/38424893-134d5722-39d0-11e8-9b0e-27f04b523f95.png)
![plot5](https://user-images.githubusercontent.com/16735822/38424895-139f9ed8-39d0-11e8-871b-c4c977fc947c.png)

## Inferences:

* 	75% of the clients are **younger** than 47 years old.

* 	2% of the clients are **older** than 60 years old.

*   The average age for subscribed and non-subscribed clients is quite similar (41 and 40 years old, respectably).

* 	75% of clients were contacted less than three times during this campaign.

* 	Of the clients contacted once in this campaign, 82% had not been contacted previously.

* 	The **employment variation rate** median of subscribed clients is almost 3 points lower than the median employment variation rate of non-subscribed clients.

* 	The **average consumer price index** is similar for groups: 93.41 for subscribed clients and 93.59 for non-subscribe clients.

* 	The **consumer confidence index** by non-subscribed and subscribed clients does not show an important difference: -40.58 non-subscribed clients and -39.78 subscribed clients.

* 	The subscribed group in **euribor 3 month rate** shows a lower median and is more variable than non-subscribed clients. The bimodal shape suggests that there are two distinct groups: the low interest rate clients and the high interest rate clients.

* 	There is an important difference in the **number of employees** of the bank by groups of clients. The median of non-subscribed clients (5196 employees) is higher than the median of subscribed clients (5076 employees).

* 	There is significant difference in the **duration of the call** between subscribed and non-subscribed clients. The subscribed-clients show a median call of 7.63 minutes, instead, 50% of non-subscribed clients were on the phone less than 2.75 minutes. Subscribed clients spent more time on the phone during the call.

* 	64% of the clients were contacted via **telephone.** 5% of these resulted in a subscription.

* 	36% of the clients were contacted via **cellular,** 14% of these resulted in a subscription.

* 	Cellular is the type of contact with the **highest subscription rate.**

* 	Telephone calls that finished with a subscription were **longer** than cellular calls that ended with a subscription.

* 	Clients for who the previous campaign was successful are **more likely** to subscribe for a term deposit.







