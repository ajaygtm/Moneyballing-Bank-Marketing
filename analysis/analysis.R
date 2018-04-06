#analysis - R Script  <author : Ajay Gautam M>

#Initial look at the data
names(train)
summary(train)
str(train)

#DATA CLEANING
#Checking for the missing values
sum(is.na(train)) 
sum(is.na(test))
sapply(train, FUN=function(x) sum(is.na(x)))  
sapply(test, FUN=function(x) sum(is.na(x)))   
#Train data - lots of missing values
#Test data - no missing value

#Imputing the missing values
#Method - MICE (Multivariate Imputation by Chained Equation) - 'mice' package
init = mice(train, maxit = 0)
predM = init$predictorMatrix #Predictor Matrix
meth = init$method #Setting the methods for each type of variable

imp = mice(train, m=5, predictorMatrix = predM, method = meth)
train = complete(imp) #Combining the imputed data with the original

##############################################################################################

#EDA.
#Generating plots to visually get some insigths about the variables

#Theme Setting
theme_set(theme_fivethirtyeight())
barfill <- "#4271AE"
barlines <- "#1F3552"    

#Missing Vales Plot
aggr_plot <- aggr(train, col=c('slateblue','red'), numbers=TRUE, prop=FALSE,
                  sortVars=TRUE, labels=names(train), cex.axis=0.7, gap=1,
                  varheight = FALSE,combined = FALSE,cex.numbers =0.5, 
                  ylab=c("Histogram of missing data","Pattern"))

#Age Variable
p1 <- ggplot(train, aes(x = age)) +
  geom_histogram(aes(fill = ..count..), binwidth = 1,
                 colour = barlines, fill = barfill)+
  scale_x_continuous(name = "Age",
                     breaks = seq(15,100,10),
                     limits = c(15,100)) +
  xlab("Age")+ylab("Count")+
  ggtitle("Frequency histogram of Age") +
  theme(plot.title = element_text(hjust = 0.5))
p1

p2 <- ggplot(train, aes(x = y, y =age)) +
  geom_boxplot(fill = "coral2", colour="firebrick4")+
  scale_y_continuous(name = "Age") +
  scale_x_discrete(name="Subscribed") +
  ggtitle("Boxplot of Age by Subscription") +
  theme(plot.title = element_text(hjust = 0.5))

p2

#Marital Vairable
p3 <- ggplot() + geom_bar(aes(y = (..count..), x = marital, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Marital Status Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p3

#Job Vairable
p4 <- ggplot() + geom_bar(aes(y = (..count..), x = job, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Job Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text.x =
          element_text(size  = 10,
                       angle = 45,
                       hjust = 1,
                       vjust = 1))
p4

#Education Vairable
p5 <- ggplot() + geom_bar(aes(y = (..count..), x = education, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Education Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text.x =
          element_text(size  = 10,
                       angle = 45,
                       hjust = 1,
                       vjust = 1))
p5

#Contact Variable
p6 <- ggplot() + geom_bar(aes(y = (..count..), x = contact, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Contact Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p6

#Default Variable
p7 <- ggplot() + geom_bar(aes(y = (..count..), x = default, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Default Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p7

#Housing Variable
p8 <- ggplot() + geom_bar(aes(y = (..count..), x = housing, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Housing Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p8

#Loan Variable
p9 <- ggplot() + geom_bar(aes(y = (..count..), x = loan, fill = y), data = train,
                          stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Loan Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p9

#Campaign
p11 <- ggplot() + geom_histogram(aes(y = (..count..), x = campaign, fill = y ), data = train,
                                 binwidth=5, stat="count") +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  scale_x_continuous(breaks = seq(1,15,1),limits=c(0,15))+
  ggtitle("No.of Contacts Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))

p11

#Month Vairable
p12 <- ggplot() + geom_bar(aes(y = (..count..), x = month, fill = y), data = train,
                           stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Month Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p12


#Days of Week Vairable
p13 <- ggplot() + geom_bar(aes(y = (..count..), x = day_of_week, fill = y), data = train,
                           stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Day Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p13


#Duration Vairable
p14 <- ggplot() + geom_bar(aes(y = (..count..), x = duration, fill = y), data = train,
                           stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Duration Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p14

#ln_duration <- log(train$duration)
ln_duration <- log(1+(train$duration))
p24 <- ggplot() + geom_histogram(aes(x = ln_duration, fill = y), data = train,
                                 binwidth = 0.1 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Log Duration Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p24

#pdays Vairable
p16 <- ggplot() + geom_histogram(aes( x = pdays, fill = y), data = train,
                                 binwidth = 50 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("PCampaign Days Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p16
table(train$pdays)


#Previous
p17 <- ggplot() + geom_histogram(aes( x = previous, fill = y), data = train,
                                 binwidth = 1 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("PContact Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p17

#POutcome
p18 <- ggplot() + geom_bar(aes(y = (..count..), x = poutcome, fill = y), data = train,
                           stat="count" ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("POutcome Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p18

#Employment variation Rate
p19 <- ggplot() + geom_histogram(aes( x = emp.var.rate, fill = y), data = train,
                                 binwidth = 1 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Emp.Var.Rate Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p19

#Consumer Price Index
p20 <- ggplot() + geom_histogram(aes( x = cons.price.idx, fill = y), data = train,
                                 binwidth = 1 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Cons.Price.Index Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p20

#Consumer Confidence Index
p21 <- ggplot() + geom_histogram(aes( x = cons.conf.idx, fill = y), data = train,
                                 binwidth =5 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Cons.Conf.Index Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p21

#Euro Interbank Offered Rates
p22 <- ggplot() + geom_histogram(aes( x = euribor3m, fill = y), data = train,
                                 binwidth =0.01 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("Euro.Interbank.Rate Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p22

#No. of Employees
p23 <- ggplot() + geom_histogram(aes( x = nr.employed, fill = y), data = train,
                                 binwidth =100 ) +
  labs(fill="Subscription") + 
  scale_y_continuous(name = "Count")+
  ggtitle("No.of Employees Vs Subscription") +
  theme(plot.title = element_text(hjust = 0.5))
p23

