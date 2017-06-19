library(ggplot2)
library(MASS)
library(caret)
library(class)
library(randomForest)
library(nnet)
library(kernlab)
library(ROCR)
library(glmnet)
library(coefplot)
library(e1071)
library(neuralnet)
detach("package:neuralnet", unload=TRUE)



## read in data and remove na's
data <- read.csv("C:/Users/dayif/Desktop/lets do it/marketing/bank-full.csv", sep = ";") 
data = na.omit(data)
summary(data)

## dropping contact, duration(from the data desription, we dont know this data until a decision is made, therefore it cant be considerd as a influential factor), and pdays(where -1 doesnt make any sence)
myvars <- names(data) %in% c("contact", "duration", "pdays") 
data = data[!myvars]
str(data)

## convert day to factor 
data$day = factor(data$day)

## standardize age, balance, campaign, and previous variables 
stand.data = data
stand.data$age =  scale(stand.data$age)
stand.data$balance = scale(stand.data$balance)
stand.data$campaign = scale(stand.data$campaign)
stand.data$previous = scale(stand.data$previous)

str(stand.data)

## take a look at our major predictors. This dataset is clean without too many missing values, we dont need to do much cleanning work

ggplot(stand.data, aes(x=age)) + 
  geom_bar(fill= 'purple', color='black') +
  labs(x= 'age', y= 'number') +
  ggtitle('client age Frequency Diagram')

ggplot(stand.data, aes(x=job)) + 
  geom_bar(fill= 'blue', color='black') +
  labs(x= 'job type', y= 'number') +
  ggtitle('Job Type Frequency Diagram')

summary(stand.data$marital) # knowing the number is ok 

# a few unknown's, it is ok to leave them there
ggplot(stand.data, aes(x=education)) + 
  geom_bar(fill= 'darkgreen', color='black') +
  labs(x= 'education level', y= 'number') +
  ggtitle('education level Frequency Diagram')

summary(stand.data$default)
summary(stand.data$balance) # outliner detected, could omit 99% level
summary(stand.data$housing)
summary(stand.data$loan)
summary(stand.data$campaign)# outliner detected, could omit 99% level
summary(stand.data$previous)# outliner detected, could omit 99% level
summary(stand.data$y)

## could be removed
ggplot(stand.data, aes(x=poutcome)) + 
  geom_bar(fill= 'red', color='black') +
  labs(x= 'previous outcome type', y= 'number') +
  ggtitle('previous outcome type Frequency Diagram')

## modeling with one trainning set and one validation set 
set.seed(604) # randomly set 70% training data set and 30% testing data set
subdata <- sample(nrow(stand.data), floor(nrow(stand.data)*0.7))
training <- stand.data[subdata,]
validation <- stand.data[-subdata,]

## lda method(not so good for involving categorical predictors)
lda = lda(y ~ ., data = training) 
lda.p = predict(lda,newdata=validation,type="response")
tablelda = table(lda.p$class,validation$y)
tablelda
summary(lda.p$class)
confusionMatrix(lda.p$class,validation$y,positive = levels(validation$y)[2])

## roc plot, auc, and threshold 
rocr.ldapred <- prediction(lda.p$posterior[,2], validation$y)
rocr.ldaperf <- performance(rocr.ldapred,"tpr","fpr")
plot(rocr.ldaperf)

rocr.ldaauc <- performance(rocr.ldapred,"auc")
ldaauc <- as.numeric(rocr.ldaauc@y.values) # 0.76

rocr.ldaacc = performance(rocr.ldapred, "acc")
ldaac.val = max(unlist(rocr.ldaacc@y.values))
ldath = unlist(rocr.ldaacc@x.values)[unlist(rocr.ldaacc@y.values) == ldaac.val]

plot(rocr.ldaacc)
abline(v=ldath, col='red', lty=2)

## manually set the threshold
lda.pred = rep("no", 13564)
lda.pred[lda.p$posterior[,2] > 0.857] = "yes"  ## according to the threshold value for the highest overall accuracy from the last part
confusionMatrix(lda.pred,validation$y,positive = levels(validation$y)[2])

## qda method(not so good for involving categorical predictors)
qda = qda(y ~ ., data = training) 
qda.p = predict(qda,newdata=validation)
tableqda = table(qda.p$class,validation$y)
tableqda
confusionMatrix(qda.p$class,validation$y,positive = levels(validation$y)[2])

## roc plot, auc, and threshold 
rocr.qdapred <- prediction(qda.p$posterior[,2], validation$y)
rocr.qdaperf <- performance(rocr.qdapred,"tpr","fpr")
plot(rocr.qdaperf)

rocr.qdaauc <- performance(rocr.qdapred,"auc")
qdaauc <- as.numeric(rocr.qdaauc@y.values) #0.75

rocr.qdaacc = performance(rocr.qdapred, "acc")
qdaac.val = max(unlist(rocr.qdaacc@y.values))
qdath = unlist(rocr.qdaacc@x.values)[unlist(rocr.qdaacc@y.values) == qdaac.val]

plot(rocr.qdaacc)
abline(v=qdath, col='red', lty=2)

## manually set the threshold
qda.pred = rep("no", 13564)
qda.pred[qda.p$posterior[,2] > 0.3] = "yes"
confusionMatrix(qda.pred,validation$y,positive = levels(validation$y)[2])

## random forest data preparation
training.rf = as.data.frame(cbind(x,training$y))
training.rfnames <- c(5:68)
training.rf[,training.rfnames] <- lapply(training.rf[,training.rfnames] , factor)
str(training.rf)

validation.rf = as.data.frame(cbind(v,validation$y))
validation.rfnames <- c(5:68)
validation.rf[,validation.rfnames] <- lapply(validation.rf[,validation.rfnames] , factor)
names(validation.rf) = gsub("validation", "training", names(validataion.rf))
str(validation.rf)

## rf model
rf.d = randomForest(V68 ~ ., data = training.rf, ntree = 50)
rf.d.p = predict(rf.d,newdata=validation.rf, predict.all=FALSE)
levels(rf.d.p)[levels(rf.d.p)=="1"] = "no"
levels(rf.d.p)[levels(rf.d.p)=="2"] = "yes"
tablerf.d = table(rf.d.p,validation$y)
tablerf.d
confusionMatrix(rf.d.p,validation$y,positive = levels(validation$y)[2])
plot(rf.d)
varImpPlot(rf.d,n.var=15)

## rf model after tuned 0.75 threshold for "yes"
rf.d.t = randomForest(V68 ~ ., data = training.rf, cutoff = c(0.75,0.25), ntree = 50)
rf.d.t.p = predict(rf.d.t,newdata=validation.rf, predict.all=FALSE)
levels(rf.d.t.p)[levels(rf.d.t.p)=="1"] = "no"
levels(rf.d.t.p)[levels(rf.d.t.p)=="2"] = "yes"
tablerf.d.t = table(rf.d.t.p,validation$y)
tablerf.d.t
confusionMatrix(rf.d.t.p,validation$y,positive = levels(validation$y)[2])
plot(rf.d.t)
varImpPlot(rf.d.t,n.var=15)


## random forest roc plot, auc, and threshold 
rfdtpredictions=as.vector(rf.d.t$votes[,2])
rocr.rfdtpred = prediction(rfdtpredictions, training$y)
rocr.rfdtperf = performance(rocr.rfdtpred,"tpr","fpr")
rocr.rfdtperf2 = performance(rocr.rfdtpred,"prec","rec")
plot(rocr.rfdtperf)
plot(rocr.rfdtperf2)

rocr.rfdtauc <- performance(rocr.rfdtpred,"auc")
rfdtauc <- as.numeric(rocr.rfdtauc@y.values) #0.73

rocr.rfdtacc = performance(rocr.rfdtpred, "acc")
rfdtacc.val = max(unlist(rocr.rfdtacc@y.values))
rfdtth = unlist(rocr.rfdtacc@x.values)[unlist(rocr.rfdtacc@y.values) == rfdtacc.val] #0.47

plot(rocr.rfdtacc)
abline(v=rfdtth, col='red', lty=5)

## rf with no dummies 
rf = randomForest(y ~ ., data = training, ntree = 100)
rf.p = predict(rf,newdata=validation, predict.all=FALSE)
tablerf = table(rf.p,validation$y)
confusionMatrix(rf.p,validation$y,positive = levels(validation$y)[2])
plot(rf)
summary(rf)
varImpPlot(rf,n.var=15)

## logistic model 
log = glm(y ~ ., data = stand.data, family = binomial(), subset = subdata)
summary(log)
log.p = predict(log,validation, type = "response")
log.pred = rep("no", 13564)
log.pred[log.p > 0.5] = "yes"
confusionMatrix(log.pred,validation$y,positive = levels(validation$y)[2])

## roc plot, auc, and threshold 
rocr.logpred <- prediction(log.p, validation$y)
rocr.logperf <- performance(rocr.logpred,"tpr","fpr")
plot(rocr.logperf)

rocr.logauc <- performance(rocr.logpred,"auc")
logauc <- as.numeric(rocr.logauc@y.values) #0.76

rocr.logacc = performance(rocr.logpred, "acc")
logacc.val = max(unlist(rocr.logacc@y.values))
logth = unlist(rocr.logacc@x.values)[unlist(rocr.logacc@y.values) == logacc.val] #0.517

plot(rocr.logacc)
abline(v=logth, col='red', lty=2)

## mannually setting threshold
log.pred2 = rep("no", 13564)
log.pred2[log.p > 0.517] = "yes"
confusionMatrix(log.pred2,validation$y, positive = levels(validation$y)[2])

## glmnet lasso variable selection 
xfactors = model.matrix(training$y ~ training$job + training$marital +training$education +training$default +training$housing +training$loan +training$day +training$month +training$poutcome)[, -1]
x = as.matrix(data.frame(training$age, training$balance, training$campaign, training$previous, xfactors))
dim(x)

glmnet.fit = glmnet(x, y=training$y, alpha=1, family="binomial")
plot(glmnet.fit, xvar = "dev", label = TRUE)

glmnet.cvfit = cv.glmnet(x, y=training$y, family = "binomial", type.measure = "class")
plot(glmnet.cvfit)
glmnet.cvfit$lambda.min ## lambda value at the minim misclassification error 
glmnet.cvfit$lambda.1se ## most regulized model selection 
coef(glmnet.cvfit, s = "lambda.min")
coef(glmnet.cvfit, s = "lambda.1se")
extract.coef(glmnet.cvfit)


vfactors = model.matrix(validation$y ~ validation$job + validation$marital +validation$education +validation$default +validation$housing +validation$loan +validation$day +validation$month +validation$poutcome)[, -1]
v = as.matrix(data.frame(validation$age, validation$balance, validation$campaign, validation$previous, vfactors))
glmnet.cvfit.p = predict(glmnet.cvfit, newx = v, s = "lambda.min", type = "response")
head(glmnet.cvfit.p)
glmnet.cvfit.pred = rep("no", 13564)
glmnet.cvfit.pred[glmnet.cvfit.p > 0.513] = "yes"
confusionMatrix(glmnet.cvfit.pred,validation$y,positive = levels(validation$y)[2])

## glmnet lasso roc plot, auc, and threshold 
rocr.glmnetcvpred <- prediction(glmnet.cvfit.p, validation$y)
rocr.glmnetcvperf <- performance(rocr.glmnetcvpred,"tpr","fpr")
rocr.glmnetcvperf2 <- performance(rocr.glmnetcvpred,"prec","rec")
plot(rocr.glmnetcvperf)
plot(rocr.glmnetcvperf2)

rocr.glmnetcvauc <- performance(rocr.glmnetcvpred,"auc")
glmnetcvauc <- as.numeric(rocr.glmnetcvauc@y.values) #0.75

rocr.glmnetcvacc = performance(rocr.glmnetcvpred, "acc")
glmnetcvacc.val = max(unlist(rocr.glmnetcvacc@y.values))
glmnetcvth = unlist(rocr.glmnetcvacc@x.values)[unlist(rocr.glmnetcvacc@y.values) == glmnetcvacc.val] #0.513 for least misclassifiction error model, and 0.512 for most regulized model

plot(rocr.glmnetcvacc)
abline(v=glmnetcvth, col='red', lty=5)


## svm


set.seed(604) # randomly set 70% training data set and 30% testing data set
subdata <- sample(nrow(stand.data), floor(nrow(stand.data)*0.7))
training <- stand.data[subdata,]
validation <- stand.data[-subdata,]

training.svm = training[sample(nrow(training), floor(nrow(training)*0.5)),]
validation.svm = validation[sample(nrow(validation), floor(nrow(validation)*0.5)),]

training.rf.svm = training.rf[sample(nrow(training.rf), floor(nrow(training.rf)*0.01)),]
validation.rf.svm = validation.rf[sample(nrow(validation.rf), floor(nrow(validation.rf)*0.01)),]


svm.new = svm(y ~.,data=training, cost = 100, gamma = 1, probability = TRUE)

svm.new.tuned = svm(y ~.,data=training.svm, cost = 0.01, gamma = 0.2, probability = TRUE)
svm.new.tuned
plot(svm.new)
svm.tune = e1071::tune.svm(V68 ~.,data=training.rf.svm,cost = c(0.01,0.1,1,10,100),gamma = c(0.2,0.5,1,5,10), probability = TRUE)
svm.tune
plot(svm.tune)


svm.new.pred = predict(svm.new, validation[,-14], probability = TRUE)
attr=attr(svm.new.pred, "probabilities")[,2]
summary(attr)
svm.new.p = rep("yes", 13564)
svm.new.p[py > 0.5] = "no"
confusionMatrix(svm.new.p,validation$y,positive = levels(validation$y)[2])

svm.new.tuned.pred = predict(svm.new.tuned, validation.svm[,-14], probability = TRUE)
tuned.attr=attr(svm.new.tuned.pred, "probabilities")[,2]
str(svm.new.tuned.pred)
svm.new.tuned.p = rep("yes", 6782)
svm.new.tuned.p[tuned.attr > 0.5] = "no"
confusionMatrix(svm.new.tuned.p,validation.svm$y,positive = levels(validation$y)[2])
plot(svm.new,data=training,balance ~ age)




rocr.glmnetcvacc = performance(rocr.glmnetcvpred, "acc")
glmnetcvacc.val = max(unlist(rocr.glmnetcvacc@y.values))
glmnetcvth = unlist(rocr.glmnetcvacc@x.values)[unlist(rocr.glmnetcvacc@y.values) == glmnetcvacc.val] #0.513 for least misclassifiction error model, and 0.512 for most regulized model

plot(rocr.glmnetcvacc)
abline(v=glmnetcvth, col='red', lty=5)


rbf <- rbfdot(sigma=0.1)
svm <- ksvm(y ~.,data=training,type="C-bsvc",kernel=rbf,C=10,prob.model=TRUE)
svm.pred = predict(svm, validation[,-14], type="probabilities")
svm.p = rep("no", 13564)
svm.p[svm.pred[,2] > 0.5] = "yes"
confusionMatrix(svm.p,validation$y,positive = levels(validation$y)[2])

rocr.svm.pred <- prediction(svm.pred[,2] , validation[,14])
rocr.svmperf <- performance(rocr.svm.pred,"tpr","fpr")
plot(rocr.svmperf)

rocr.svmauc <- performance(rocr.svm.pred,"auc")
svmauc <- as.numeric(rocr.svmauc@y.values) #0.72

## data preparation for knn
knn.data = stand.data
str(knn.data) # only contains num and factor variable types

job.d= model.matrix(~knn.data$job)[,-1]
mari.d= model.matrix(~knn.data$marital)[,-1]
edu.d = model.matrix(~knn.data$education)[,-1]
def.d = model.matrix(~knn.data$default)[,-1]
hous.d = model.matrix(~knn.data$housing)[,-1]
loan.d = model.matrix(~knn.data$loan)[,-1]
day.d = model.matrix(~knn.data$day)[,-1]
month.d = model.matrix(~knn.data$month)[,-1]
pout.d = model.matrix(~knn.data$poutcome)[,-1]
knn.data = knn.data[,c(1,6,11,12)]

knn.data = cbind(knn.data,job.d,mari.d,edu.d,def.d,hous.d,loan.d,day.d,month.d,pout.d,stand.data$y)
dim(knn.data)
str(knn.data)

training.knn = knn.data[subdata,]
validation.knn = knn.data[-subdata,]
training.knn$`stand.data$y`
## knn 
knn = knn(training.knn[,c(1:67)],validation.knn[,c(1:67)],training.knn[,68],13,prob=TRUE)
confusionMatrix(knn,validation.knn[,68],positive = levels(validation$y)[2])
prob.knn <- attr(knn, "prob")
prob <- ifelse(knn == "no", 1-prob.knn, prob.knn)


trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3333)
knn_fit <- train(V68 ~., data = training.rf, method = "knn",
                 trControl=trctrl,
                 preProcess = c("center", "scale"),
                 tuneLength = 10)
knn_fit
plot(knn_fit)


## knn roc plot, auc, and threshold 
rocr.knnpred = prediction(prob, validation$y)
rocr.knnperf <- performance(rocr.knnpred,"tpr","fpr")
rocr.knnperf2 <- performance(rocr.knnpred,"prec","rec")
plot(rocr.knnperf)
plot(rocr.knnperf2)

rocr.knnauc <- performance(rocr.knnpred,"auc")
knnauc <- as.numeric(rocr.knnauc@y.values) #0.66

rocr.glmnetcvacc = performance(rocr.glmnetcvpred, "acc")
glmnetcvacc.val = max(unlist(rocr.glmnetcvacc@y.values))
glmnetcvth = unlist(rocr.glmnetcvacc@x.values)[unlist(rocr.glmnetcvacc@y.values) == glmnetcvacc.val] #0.513 for least misclassifiction error model, and 0.512 for most regulized model

plot(rocr.glmnetcvacc)
abline(v=glmnetcvth, col='red', lty=5)


## neural network
ideal <- class.ind(training.knn[,68])
ann = nnet(training.knn[,c(1:67)],ideal, size=10, softmax=TRUE)
ann.p = predict(ann, validation.knn[,-68], type="raw") 
ann.c = predict(ann, validation.knn[,-68], type="class") 
head(ann.p[,2])
ann.pred = rep("no", 13564)
ann.pred[ann.p[,2]> 0.2] = "yes"
confusionMatrix(ann.pred,validation.knn[,68],positive = levels(validation$y)[2])
confusionMatrix(ann.c,validation.knn[,68],positive = levels(validation$y)[2])

## glmnet lasso roc plot, auc, and threshold 
rocr.annpred <- prediction(ann.p[,2], validation.knn[,68])
rocr.annperf <- performance(rocr.annpred,"tpr","fpr")
rocr.annperf2 <- performance(rocr.annpred,"prec","rec")
plot(rocr.annperf)
plot(rocr.annperf2)

rocr.annauc <- performance(rocr.annpred,"auc")
annauc <- as.numeric(rocr.annauc@y.values) #0.78

rocr.annacc = performance(rocr.annpred, "acc")
annacc.val = max(unlist(rocr.annacc@y.values))
annth = unlist(rocr.annacc@x.values)[unlist(rocr.annacc@y.values) == annacc.val] #0.513 for least misclassifiction error model, and 0.512 for most regulized model

plot(rocr.annacc)
abline(v=annth, col='red', lty=5)


# ## neural network data preparation
# nnet.data = data
# nnet.data$age =  scale(nnet.data$age, center = min(nnet.data$age), scale = max(nnet.data$age) -min(nnet.data$age)) 
# nnet.data$balance =  scale(nnet.data$balance, center = min(nnet.data$balance), scale = max(nnet.data$balance) -min(nnet.data$balance)) 
# nnet.data$campaign =  scale(nnet.data$campaign, center = min(nnet.data$campaign), scale = max(nnet.data$campaign) -min(nnet.data$campaign)) 
# nnet.data$previous =  scale(nnet.data$previous, center = min(nnet.data$previous), scale = max(nnet.data$previous) -min(nnet.data$previous)) 
# 
# nnet.training <- nnet.data[subdata,]
# nnet.validation <- nnet.data[-subdata,]
# 
# nnet.xfactors = model.matrix(nnet.training$y ~ nnet.training$job + nnet.training$marital +nnet.training$education +nnet.training$default +nnet.training$housing +nnet.training$loan +nnet.training$day +nnet.training$month +nnet.training$poutcome)[, -1]
# nnet.x = as.matrix(data.frame(nnet.training$age, nnet.training$balance, nnet.training$campaign, nnet.training$previous, nnet.xfactors))
# 
# nnet.data = as.data.frame(cbind(nnet.x,nnet.training$y))
# nnet.data$V68 = as.numeric(training$y)
# nnet.data$V68[nnet.data$V68 == 1] = 0
# nnet.data$V68[nnet.data$V68 == 2] = 1
# 
# 
# nnet.vfactors = model.matrix(nnet.validation$y ~ nnet.validation$job + nnet.validation$marital +nnet.validation$education +nnet.validation$default +nnet.validation$housing +nnet.validation$loan +nnet.validation$day +nnet.validation$month +nnet.validation$poutcome)[, -1]
# nnet.v = as.matrix(data.frame(nnet.validation$age, nnet.validation$balance, nnet.validation$campaign, nnet.validation$previous, nnet.vfactors))
# 
# nnet.validation = as.data.frame(cbind(nnet.v,nnet.validation$y))
# 
# summary(nnet.data$V68)
# str(nnet.data)
# nnet.datanames <- names(nnet.data[-68])
# # Concatenate strings
# f <- paste(nnet.datanames ,collapse=' + ')
# f <- paste('V68 ~',f)
# # Convert to formula
# f <- as.formula(f)
# f
# nn = neuralnet(f, nnet.data,
#                hidden=c(10,5,3), 
#                act.fct = "logistic",
#                linear.output = FALSE,
#                lifesign = "minimal")
# plot(nn)
# str(nn)
# predicted.nn.values = compute(nn, nnet.validation[,1:67])
# str(predicted.nn.values$net.result)
# predicted.nn.values$net.result <- sapply(predicted.nn.values$net.result,round,digits=0)
# summary(predicted.nn.values$net.result)
# 
# nnet.pred = rep("no", 13564)
# nnet.pred[predicted.nn.values$net.result > 0.5] = "yes"
# 
# confusionMatrix(predicted.nn.values$net.result,validation$y,positive = levels(validation$y)[2])
# 
# str(training.knn)
# str(nnet.data)
# 
# nnet <- train(f , data = nnet.data, method='nnet', linout=TRUE, trace = FALSE)
# #Grid of tuning parameters to try:
# tuneGrid=expand.grid(.size=c(1,5,10),.decay=c(0,0.001,0.1))) 