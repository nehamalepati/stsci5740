# install necessary libraries
library(plyr)
library(readr)
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(Hmisc)
library(zoo)
library(lubridate)
library(caret)
library(kknn)
library(gbm)
library(mlr)
library(parallel)
library(parallelMap)
library(ISLR)

# read comb_data
data <- read.csv('wine-quality-white-and-red.csv', sep=',')
red <- read.csv('winequality-red.csv', sep=';')
white <- read.csv('winequality-white.csv', sep=';')

# summary statistics
quant_vals <- sapply(data, is.numeric)
# quant_vals <- c("fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality")
summary(data[, quant_vals])

# comb_data pre processing (remove outliers)

# boxplots and scatterplots
# pairs(red)
# pairs(white)
pairs(data[, quant_vals])

# linear regression
pred <- lm(quality ~ volatile.acidity + chlorides + total.sulfur.dioxide + pH + sulphates + alcohol, data = data)

summary(pred)

confint(pred, conf.level=0.95)

plot(pred)

effect_plot(model = pred, pred = alcohol, interval = TRUE, plot.points = TRUE)

d = comb_data.frame(pred$model, pred$residuals)

with(d, sd(pred.residuals))

with(subset(d, alcohol > 8 & alcohol < 15), sd(pred.residuals))

d$resid <- as.numeric(d$pred.residuals)

ggplot2(aes(y = resid, x = round(alcohol, 2)), comb_data = d) + geom_line(stat = "summary", fun.y = sd)

# Logistic regression
# KNN model 
# Random forest 


# 
