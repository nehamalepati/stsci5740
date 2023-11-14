# install necessary libraries
library(ISLR)
library(class)
library(janitor)

# read data
data <- read.csv('wine-quality-white-and-red.csv', sep=',')
red <- read.csv('winequality-red.csv', sep=';')
white <- read.csv('winequality-white.csv', sep=';')
data <- clean_names(data)

# summary statistics
quant_vals <- sapply(data, is.numeric)
summary(data[, quant_vals])
cor(data[, quant_vals])

# scatterplots
pairs(data[, quant_vals])

# split into test and train
set.seed(7)
train_idx <- createDataPartition(data$quality, p = 0.7, list = FALSE, times = 1)
data_train <- data[train_idx,]
data_test  <- data[-train_idx,]

# linear regression
linear <- lm(quality ~ volatile_acidity + chlorides + total_sulfur_dioxide + p_h + sulphates + alcohol, data = data_train)
summary(linear)
pred_linear <- predict(linear, data_test)
linear_error <- mean(pred_linear != data_test$quality)
print("Test Error for Linear: ")
print(linear_error)

# logistic regression
data$rating = if_else(data$quality >= 6, 1, 0 )
set.seed(7)
train_idx <- createDataPartition(data$quality, p = 0.7, list = FALSE, times = 1)
data_train <- data[train_idx,]
data_test  <- data[-train_idx,]
logistic <- glm(rating ~ type + volatile_acidity + chlorides + total_sulfur_dioxide + p_h + sulphates + alcohol, family=binomial, data=data_train)
summary(logistic)
pred_logistic = predict(logistic, data_test) > 0.5
logistic_error <- mean(pred_logistic != data_test$rating)
print("Test Error for Logistic: ")
print(logistic_error)

# knn model
features = c("volatile_acidity", "chlorides", "total_sulfur_dioxide", "p_h", "sulphates", "alcohol")
kset<-c(1:9,seq(10,60,5))
test_error<-c()
for(num_k in kset)
{
  predknn_k1 = knn(data_train[, features], data_test[, features], data_train$quality, k=num_k)
  test_error<-c(test_error,mean(predknn_k1 != data_test$quality))
}
test_error
print("Test Error for KNN: ")
print(test_error)
print(kset[which.min(test_error)])

# regression trees

