library('e1071')
source('metrics.r')

validation = data.frame()
submission = data.frame()
predsvm <- function(model,newdata) {
  prob<-attr(predict(model, newdata, probability = TRUE),"probabilities")
  n<-dim(prob)[1]
  m<-dim(prob)[2]
  me<-which(prob==apply(prob,1,max))
  return(as.factor(model$labels[floor((me-1)/n)+1]))
}

for (i in 1:10) {
  print(paste('Training and predicting on essay set ', i))
  print('Loading datasets...')
  train <- read.csv(paste(paste('data/features/train_', i,
                                sep=''), '.csv', sep=''), header=TRUE)
  test <- read.csv(paste(paste('data/features/test_', i,
                               sep=''), '.csv', sep=''), header=TRUE)

  print('Preprocessing...')
  train$Color <- as.factor(train$Color)
  train$Score1 <- as.factor(train$Score1)
  test$Color  <- as.factor(test$Color)

  # Split out a CV
  n <- nrow(train)
  indices <- sort(sample(1:n, round(0.8 * n)))
  construct <- train[indices,]
  cv        <- train[-indices,]

  print('Training SVM')
  first = 5 # ifelse(i == 10, 4, 5)
  model <- svm(construct[,first:ncol(construct)], construct$Score1, probability=TRUE,
               scale=0, kernel='polynomial')
  print('Predict CV')
  cv.pred <- predsvm(model, cv[,first:ncol(cv)])
  print('Adding to validation')
  validation <- rbind(validation,
                      data.frame(Id = cv$Id, Score1 = cv$Score1, Pred = cv.pred))

  print('Predict Test')
  test.pred <- predsvm(model, test[,first:ncol(test)])
  print('Adding to submission')
  submission <- rbind(submission,
  data.frame(id = test$Id, essay_score = test.pred))
}

print(paste('RMSE: ',
            rmse(as.numeric(validation$Score1), as.numeric(validation$Pred))))
print(paste('QWK: ',
            ScoreQuadraticWeightedKappa(validation$Score1, validation$Pred, 0, 3)))

final.submission <- data.frame(id = submission$id, essay_score = as.numeric(levels(submission$essay_score)[submission$essay_score]))
write.csv(final.submission, 'submission.csv', row.names=FALSE)
