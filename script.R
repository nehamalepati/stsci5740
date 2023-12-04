# install necessary libraries
library(ISLR)
library(class)
library(janitor)
library(Metrics)
library(rpart)
library(caret)
library(dplyr)
library(MASS)

# read data
data <- read.csv("wine-quality-white-and-red.csv", sep = ",")
red <- read.csv("winequality-red.csv", sep = ";")
white <- read.csv("winequality-white.csv", sep = ";")
data <- clean_names(data)
red <- clean_names(red)
white <- clean_names(white)

# summary statistics
quant_vals <- sapply(data, is.numeric)
summary(data[, quant_vals])
cor(data[, quant_vals])
summary(red)
summary(white)

# scatterplots
pairs(data[, quant_vals])

# boxplots
par(mfrow = c(3, 4))
boxplot(fixed_acidity ~ quality, data = data, main = "Fixed Acidity vs Quality")
boxplot(volatile_acidity ~ quality, data = data, main = "Volatile Acidity vs Quality")
boxplot(citric_acid ~ quality, data = data, main = "Citric Acid vs Quality")
boxplot(residual_sugar ~ quality, data = data, main = "Residual Sugar vs Quality")
boxplot(chlorides ~ quality, data = data, main = "Chlorides vs Quality")
boxplot(free_sulfur_dioxide ~ quality, data = data, main = "Free Sulfur Dioxide vs Quality")
boxplot(total_sulfur_dioxide ~ quality, data = data, main = "Total Sulfur Dioxide vs Quality")
boxplot(density ~ quality, data = data, main = "Density vs Quality")
boxplot(p_h ~ quality, data = data, main = "pH vs Quality")
boxplot(sulphates ~ quality, data = data, main = "Sulphates vs Quality")
boxplot(alcohol ~ quality, data = data, main = "Alcohol vs Quality")

# split into test and train
set.seed(7)
train_idx <- createDataPartition(data$quality, p = 0.7, list = FALSE, times = 1)
data_train <- data[train_idx, ]
data_test <- data[-train_idx, ]

set.seed(7)
red_idx <- createDataPartition(red$quality, p = 0.7, list = FALSE, times = 1)
red_train <- red[red_idx, ]
red_test <- red[-red_idx, ]

set.seed(7)
white_idx <- createDataPartition(white$quality, p = 0.7, list = FALSE, times = 1)
white_train <- white[white_idx, ]
white_test <- white[-white_idx, ]

# linear regression
linear <- lm(quality ~ volatile_acidity + chlorides + total_sulfur_dioxide + p_h + sulphates + alcohol, data = data_train)
summary(linear)
pred_linear <- predict(linear, data_test)
mae_linear <- mae(data_test$quality, pred_linear)
print("Mean Absolute Error for Linear: ")
print(mae_linear)

linear_red <- lm(quality ~ volatile_acidity + chlorides + total_sulfur_dioxide + p_h + sulphates + alcohol, data = red_train)
summary(linear_red)
pred_linear_red <- predict(linear_red, red_test)
mae_linear_red <- mae(red_test$quality, pred_linear_red)
print("Mean Absolute Error for Linear (Red): ")
print(mae_linear_red)

linear_white <- lm(quality ~ volatile_acidity + chlorides + total_sulfur_dioxide + p_h + sulphates + alcohol, data = white_train)
summary(linear_white)
pred_linear_white <- predict(linear_white, white_test)
mae_linear_white <- mae(white_test$quality, pred_linear_white)
print("Mean Absolute Error for Linear (White): ")
print(mae_linear_white)

# logistic regression
data$rating <- if_else(data$quality >= 6, 1, 0)
set.seed(7)
train_idx <- createDataPartition(data$quality, p = 0.7, list = FALSE, times = 1)
data_train <- data[train_idx, ]
data_test <- data[-train_idx, ]
logistic <- glm(rating ~ type + volatile_acidity + chlorides + total_sulfur_dioxide + p_h + sulphates + alcohol, family = binomial, data = data_train)
summary(logistic)
pred_logistic <- predict(logistic, data_test) > 0.5
logistic_error <- mean(pred_logistic != data_test$rating)
print("Test Error for Logistic: ")
print(logistic_error)

red$rating <- if_else(red$quality >= 6, 1, 0)
set.seed(7)
red_idx <- createDataPartition(white$quality, p = 0.7, list = FALSE, times = 1)
red_train <- red[red_idx, ]
red_test <- red[-red_idx, ]
logistic_red <- glm(rating ~ volatile_acidity + chlorides + total_sulfur_dioxide + p_h + sulphates + alcohol, family = binomial, data = red_train)
summary(logistic_red)
pred_logistic_red <- predict(logistic_red, red_test) > 0.5
logistic_error_red <- mean(pred_logistic_red != red_test$rating)
print("Test Error for Logistic (Red): ")
print(logistic_error_red)

white$rating <- if_else(white$quality >= 6, 1, 0)
set.seed(7)
white_idx <- createDataPartition(white$quality, p = 0.7, list = FALSE, times = 1)
white_train <- white[white_idx, ]
white_test <- white[-white_idx, ]
logistic_white <- glm(rating ~ volatile_acidity + chlorides + total_sulfur_dioxide + p_h + sulphates + alcohol, family = binomial, data = white_train)
summary(logistic_white)
pred_logistic_white <- predict(logistic_white, white_test) > 0.5
logistic_error_white <- mean(pred_logistic_white != white_test$rating)
print("Test Error for Logistic (White): ")
print(logistic_error_white)

# knn model
features <- c("volatile_acidity", "chlorides", "total_sulfur_dioxide", "p_h", "sulphates", "alcohol")
kset <- c(1:9, seq(10, 60, 5))
test_error <- c()
for (num_k in kset)
{
  predknn_k1 <- knn(data_train[, features], data_test[, features], data_train$quality, k = num_k)
  test_error <- c(test_error, mean(predknn_k1 != data_test$quality))
}
test_error
print("Test Error for KNN: ")
print(test_error)
print(kset[which.min(test_error)])

red_train <- na.omit(red_train)
kset <- c(1:9, seq(10, 60, 5))
test_error_red <- c()
for (num_k in kset)
{
  predknn_k1 <- knn(red_train[, features], red_test[, features], red_train$quality, k = num_k)
  test_error_red <- c(test_error_red, mean(predknn_k1 != red_test$quality))
}
test_error_red
print("Test Error for KNN (Red): ")
print(test_error_red)
print(kset[which.min(test_error_red)])

white_train <- na.omit(white_train)
kset <- c(1:9, seq(10, 60, 5))
test_error_white <- c()
for (num_k in kset)
{
  predknn_k1 <- knn(white_train[, features], white_test[, features], white_train$quality, k = num_k)
  test_error_white <- c(test_error_white, mean(predknn_k1 != white_test$quality))
}
test_error_white
print("Test Error for KNN (White): ")
print(test_error_white)
print(kset[which.min(test_error_white)])

# lda model
lda <- lda(quality ~ volatile_acidity + chlorides + total_sulfur_dioxide + p_h + sulphates + alcohol, data = data_train)
summary(lda)
pred_lda <- predict(lda, data_test)
lda_error <- mean(pred_lda$class != data_test$quality)
print("Test Error for LDA: ")
print(lda_error)

lda_red <- lda(quality ~ volatile_acidity + chlorides + total_sulfur_dioxide + p_h + sulphates + alcohol, data = red_train)
summary(lda_red)
pred_lda_red <- predict(lda_red, red_test)
lda_error_red <- mean(pred_lda_red$class != red_test$quality)
print("Test Error for LDA (Red): ")
print(lda_error_red)

lda_white <- lda(quality ~ volatile_acidity + chlorides + total_sulfur_dioxide + p_h + sulphates + alcohol, data = white_train)
summary(lda_white)
pred_lda_white <- predict(lda, white_test)
lda_error_white <- mean(pred_lda_white$class != white_test$quality)
print("Test Error for LDA (White): ")
print(lda_error_white)