#Importing - R Code < author: Ajay Gautam M>

#Setting the working directory
getwd()
setwd('.../data')


#Required libraries
library(mice)
library(ggplot2)
library(ggthemes)
library(Hmisc)
library(moments)
library(car)
library(fmsb)
library(corrplot)
library(psych)
library(DMwR)
library(caret)
library(ROCR)
library(randomForest)
library(e1071)
library(data.table)
library(mlr)
library(xgboost)
library(parallel)
library(parallelMap)
library(VIM)


#Importing the dataset
train = read.csv("bank.csv", na.strings = "")
test = read.csv("bank-additional.csv", na.strings = "")

#Initial look at the data
names(train)
summary(train)
str(train)
