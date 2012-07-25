library('gbm')
source('metrics.r')

validation = data.frame()
submission = data.frame()
for (i in 1:10) {
  print(paste('Training and predicting on essay set ', i))
  print('Loading datasets...')
  train <- read.csv(paste(paste('data/features/train_', i,
                                sep=''), '.csv', sep=''), header=TRUE)
  test <- read.csv(paste(paste('data/features/test_', i,
                               sep=''), '.csv', sep=''), header=TRUE)

  print('Preprocessing...')
  train$Color <- as.factor(train$Color)
  test$Color  <- as.factor(test$Color)

  # Split out a CV
  n <- nrow(train)
  indices <- sort(sample(1:n, round(0.8 * n)))
  construct <- train[indices,]
  cv        <- train[-indices,]

  sas.formula <- as.formula(paste('Score1 ~',
                                  paste(names(construct)[4:ncol(construct)], collapse=' + ')))
  model <- gbm(sas.formula, n.trees=500, data=construct,
               distribution='gaussian', interaction.depth=6,
               train.fraction=.75, cv.folds=5)
  optim.iter <- gbm.perf(model, method='cv')
  print('Predict CV')
  cv.pred <- predict(model, newdata=cv, optim.iter)
  cv.pred <- round(cv.pred)
  cv.pred <- sapply(cv.pred, function (p) { ifelse(p > 3, 3, p) })
  print('Adding to validation')
  validation <- rbind(validation,
                      data.frame(Id = cv$Id, Score1 = cv$Score1, Pred = cv.pred))

  print('Predict Test')
  test.pred <- predict(model, newdata=test, optim.iter)
  test.pred <- round(test.pred)
  test.pred <- sapply(test.pred, function (p) { ifelse(p > 3, 3, p) })
  print('Adding to submission')
  submission <- rbind(submission,
  data.frame(id = test$Id, essay_score = test.pred))
}

print(paste('RMSE: ',
            rmse(as.numeric(validation$Score1), as.numeric(validation$Pred))))
print(paste('QWK: ',
            ScoreQuadraticWeightedKappa(validation$Score1, validation$Pred, 0, 3)))

final.submission <- data.frame(id = submission$id, essay_score = submission$essay_score)
write.csv(final.submission, 'submission.csv', row.names=FALSE)
