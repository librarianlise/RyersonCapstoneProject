# -------------------- INSTALL AND LOAD PACKAGES -------------------- #

#import and load libraries
install.packages("class")
install.packages("gmodels")
install.packages("e1071")
install.packages("randomForest")
install.packages("plyr")
install.packages("descr")
install.packages("ggplot2")
install.packages("ROCR")
install.packages("caret")
install.packages("inTrees")
install.packages("rpart")
install.packages("rpart.plot")

library(class)
library(gmodels)
library(e1071) 
library(randomForest)
library(plyr)
library(descr)
library(ggplot2)
library(ROCR)
library(caret)
library(inTrees)
library(rpart)
library(rpart.plot)


# -------------------- READ IN DATA AND CREATE CROSS TABS -------------------- #

#set working directory
setwd("C:/Users/lised/Desktop/CapstoneR")

# read in cleaned data
fludataraw<-read.csv("CapstoneDataCleaned.csv", header=TRUE, sep=",")

# look at data
head(fludataraw)
str(fludataraw)

# change 0/1 binary variables to logical
fludataraw[, c(2,4,7,8,12)] <- sapply(fludataraw[, c(2,4,7,8,12)], as.logical)

# remove extra variableto create machine learning data; move class variable to the end
fludataml = fludataraw[, c(3:12, 2)]

# create cross-tabs for each independent variable crossed with the class variable
for (i in 3:12) print(crosstab(fludataraw[,i], fludataraw$flu_past_year, prop.r=TRUE))


# -------------------- DETERMINE DECISION RULES -------------------- #


dtree <- rpart(flu_past_year ~., data=fludataml, method="class", parms = list(split="information"))
printcp(dtree)
summary(dtree)

plot(dtree, uniform=TRUE, main="Classification Tree for Flu Shots")
text(dtree, use.n=TRUE, all=TRUE, cex=0.65)

rpart.rules(dtree, nn=TRUE)


# -------------------- CREATE TRAINING AND TEST SETS -------------------- #

# create test and training sets

set.seed(123)
index <- sample(1:nrow(fludataml), 0.65 *nrow(fludataml))
flutrain <- fludataml[index,]
flutest <- fludataml[-index,]
flulabels <- flutest[,11]

# -------------------- CLASSIFIER 1: LOGISTIC REGRESSION -------------------- #

#create model using the training data
modelLR<- (glm(flu_past_year ~., family=binomial(link="logit"), data = flutrain))

modelLR

# calculate flu test results using test set
predLR <- predict(modelLR, newdata=flutest[, c(1:10)],type='response')

# because results are expressed as probabilities, change to TRUE if P>0.5 and FALSE otherwise
predLRbinary <- ifelse(predLR > 0.5, TRUE, FALSE)

xtabLR <- table(predLRbinary, flulabels)
resultsLR <- confusionMatrix(xtabLR)

resultsLR
resultsLR$byClass
resultsLR$overall
resultsLR$table
resultsLR$positive

# -------------------- CLASSIFIER 2: NAIVE BAYES -------------------- #

modelNB=naiveBayes(flu_past_year ~., data=flutrain)

modelNB

# calculate flu test results using test set
predNB=predict(modelNB,newdata=flutest[, c(1:10)])

xtabNB <- table(predNB, flulabels)
resultsNB <- confusionMatrix(xtabNB)

resultsNB

# -------------------- CLASSIFIER 3: RANDOM FOREST -------------------- #

#create model using the training data (estimated 35 minutes to run)
modelRF <- randomForest(flutrain[,c(1:10)], y=flutrain[,11], importance = TRUE, ntree=500)

modelRF

# calculate flu test results using test set
predRF <- predict(modelRF, newdata=flutest[, c(1:10)],type='response')

# because results are expressed as probabilities, change to TRUE if P>0.5 and FALSE otherwise
predRFbinary <- ifelse(predRF > 0.5, TRUE, FALSE)

xtabRF <- table(predRFbinary, flulabels)
resultsRF <- confusionMatrix(xtabRF)

resultsRF