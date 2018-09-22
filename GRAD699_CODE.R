
# load libraries

library(data.table)

library(ggplot2)

library(plyr)

library(e1071)

library(dplyr)

library(corrplot)

library(pROC)

library(glmnet)

library(caret)

library(xgboost)

library(readr)

library(randomForest)



data <- read.csv("C:/Users/Aruna/Desktop/GRAD695/data.csv")

head(data)

apply(data, 2, function(x) sum(is.na(x)))

common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))
p <- ggplot(data, aes(x = Class)) + geom_bar() + ggtitle("Number of class labels") + common_theme
print(p)

summary(data)



data %>% group_by(Class) %>% summarise(mean(Amount), median(Amount))


data$Class <- as.numeric(data$Class)
corr_plot <- corrplot(cor(data[,-data$Time]), method = "circle", type = "upper")

normalize <- function(x){
  return((x - mean(x, na.rm = TRUE))/sd(x, na.rm = TRUE))}
data$Amount <- normalize(data$Amount)

#Model development


#separate train and validation set
nrow(data)
nrow(data) * .75
sample_rows <- sample(284807,213605)
train <- data[sample_rows,]
test <- data[-sample_rows,]

# Logistic Regression Model development


log_mod <- glm(Class ~ ., family = "binomial", data = train)
summary(log_mod)

p <- predict(log_mod,test,type = "response")
summary(p)

p_class <- ifelse(p > 0.5,"1" , "0" )
table(p_class)

ConfMat_table <- table(p_class,test$Class)
(Accuracy <- (ConfMat_table[1]+ConfMat_table[4])/sum(ConfMat_table)*100)

fourfoldplot(ConfMat_table)

confusionMatrix(ConfMat_table)


ROC <- roc(test$Class,p)

plot(ROC, main = paste0("ROC Curve of Logistic Regression Model AUC: ", round(pROC::auc(ROC), 3)))


## Random Forests Model
rfmodel <- randomForest(Class ~., ntree = 100, data = train)
summary(rfmodel)

#RF Model Prediction 
rfPredict <- predict(rfmodel, test, type = "class")
summary(rfPredict)


RFp_class <- ifelse(p > 0.5,"1" , "0" )
table(RFp_class)

#Accuracy

rfConfMat_table <- table(RFp_class,test$Class)
(Accuracy <- (rfConfMat_table[1]+ rfConfMat_table[4])/sum(rfConfMat_table)*100)


#Confusion Matrix
fourfoldplot(rfConfMat_table)
confusionMatrix(ConfMat_table)

# ROC and AUC
ROCrf <- roc(test$Class,rfPredict)
plot(ROC, main = paste0("ROC Curve of Random Forest Model AUC: ", round(pROC::auc(ROC), 3)))

#variable Importance
options(repr.plot.width=5, repr.plot.height=4)
varImpPlot(rfmodel,
           sort = T,
           n.var=10,
           main="Top 10 Most Important Variables")
