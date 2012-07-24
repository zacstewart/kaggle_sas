library('gbm')
library('randomForest')
library('e1071')

print('Loading datasets...')
train <- read.csv('data/train_features.csv', header=TRUE)
test  <- read.csv('data/public_leaderboard_features.tsv', header=TRUE)

print('Preprocessing...')

sas.formula = as.formula(paste('Score1 ~',
                               paste(names(train)[c(4:ncol(train))], collapse=" + ")))

validation = data.frame()
submission = data.frame()
for (i in 1:10) {
  print(paste('Training and predicting on essay set ', i))
  my_train <- train[which(train$EssaySet == i),]
  my_test <- test[which(test$EssaySet == i),]
  my_train$Color <- as.factor(my_train$Color)
  my_train$Score1 <- as.factor(my_train$Score1)
  my_test$Color  <- as.factor(my_test$Color)

  # Split out a CV
  n <- nrow(my_train)
  indices <- sort(sample(1:n, round(0.8 * n)))
  construct <- my_train[indices,]
  my_cv <- my_train[-indices,]

  print(unique(construct$Score1))
  model <- randomForest(construct[,5:ncol(construct)], construct$Score1,
                        sampsize = nrow(construct)*.75, ntree=500, do.trace=TRUE)
  print('Predict CV')
  my_cv$Pred <- predict(model, newdata=my_cv)
  validation <- rbind(validation, my_cv)
  print('Predict Test')
  my_test$Pred <- predict(model, newdata=my_test)
  submission <- rbind(submission, my_test)
}
final.submission <- data.frame(id = submission$Id, essay_score = as.numeric(submission$Pred))
write.csv(final.submission, 'submission.csv', row.names=FALSE)
