library('tm')
library('gbm')
load('lexical_database.Rdata')

get.tdm <- function(doc.vec) {
  doc.corpus <- Corpus(VectorSource(doc.vec))
  control <- list(stopwords=TRUE, removePunctuation=TRUE, removeNumbers=TRUE, minDocFreq=2)
  doc.dtm <- TermDocumentMatrix(doc.corpus, control)
  return(doc.dtm)
}

get.tdf <- function(doc.vec, extra=TRUE) {
  tdm <- get.tdm(doc.vec)
  tdf.matrix <- as.matrix(tdm)
  tdf.counts <- rowSums(tdf.matrix)
  tdf.df <- data.frame(cbind(names(tdf.counts), as.numeric(tdf.counts)),
                       stringsAsFactors=FALSE)
  names(tdf.df) <- c('term', 'frequency')
  if (extra) {
    tdf.df$frequency <- as.numeric(tdf.df$frequency)
    tdf.occurence <- sapply(1:nrow(tdf.matrix), function (i) {
                            length(which(tdf.matrix[i,] > 0)) / ncol(tdf.matrix)})
    tdf.density <- tdf.df$frequency/sum(tdf.df$frequency)

    tdf.df <- transform(tdf.df, density=tdf.density, occurence=tdf.occurence)
  }
  return(tdf.df)
}

get.misspellings <- function (essay.vec) {
  return(sapply(essay.vec, function (essay) {
                essay.tdm <- get.tdm(essay)
                essay.matrix <- as.matrix(essay.tdm)
                essay.counts <- rowSums(essay.matrix)
                essay.df <- data.frame(cbind(names(essay.counts),as.numeric(essay.counts)),
                                       stringsAsFactors=FALSE)
                names(essay.df) <- c('term', 'frequency')
                essay.df$misspelled <- unlist(lapply(essay.df$term, function (term) {
                                                     freq <- lexical.database[[term]]
                                                     ifelse(is.null(freq) || is.na(freq), 1, 0)}))
                sum(essay.df$misspelled)}))
}

get.word.rarity <- function (essay.vec) {
  return(log(sapply(essay.vec, function (essay) {
                my.tdf <- get.tdf(essay, extra=FALSE)
                my.tdf$probability <- sapply(my.tdf$term, function (term) {
                                             prob <- lexical.database[[term]]
                                             return(ifelse(is.null(prob) || is.na(prob), 1, prob))})
                ifelse(length(my.tdf$probability) < 1, 1, prod(my.tdf$probability))})))
}

get.grade.prob <- function (essay, grade.df, prior=.25, c=1e-6) {
  essay.tdm <- get.tdm(essay)
  essay.freq <- rowSums(as.matrix(essay.tdm))
  essay.match <- intersect(names(essay.freq), grade.df$term)
  if (length(essay.match) < 1) {
    return(prior*c^(length(essay.freq)))
  } else {
    match.probs <- grade.df$occurance[match(essay.match, grade.df$term)]
    return(prior * prod(match.probs) * c^(length(essay.freq)-length(essay.match)))
  }
}

add.gen.features <- function (data.set) {
  print(' -- Various counts')
  data.set <- transform(data.set,
                        Length = sapply(data.set$EssayText, nchar),
                        Words = sapply(data.set$EssayText, function (e) {
                                       length(unlist(strsplit(e, '[^[:alnum:]_]')))}),
                        Sentences = sapply(data.set$EssayText, function (e) {
                                           length(unlist(strsplit(e, '\\.')))}))
  print(' -- Misspellings')
  data.set <- transform(data.set, Misspellings = get.misspellings(data.set$EssayText))
  print(' -- Word rarity')
  data.set <- transform(data.set, WordRarity = get.word.rarity(data.set$EssayText))
  print(' -- Bayes probabilities')
  grade0.df <- get.tdf(train$EssayText[which(train$Score1 == 0)])
  grade1.df <- get.tdf(train$EssayText[which(train$Score1 == 1)])
  grade2.df <- get.tdf(train$EssayText[which(train$Score1 == 2)])
  grade3.df <- get.tdf(train$EssayText[which(train$Score1 == 3)])
  data.set <- transform(data.set,
                        BayesProb0 = sapply(data.set$EssayText, function (essay) {
                                            get.grade.prob(essay, grade0.df)}),
                        BayesProb1 = sapply(data.set$EssayText, function (essay) {
                                            get.grade.prob(essay, grade1.df)}),
                        BayesProb2 = sapply(data.set$EssayText, function (essay) {
                                            get.grade.prob(essay, grade2.df)}),
                        BayesProb3 = sapply(data.set$EssayText, function (essay) {
                                            get.grade.prob(essay, grade3.df)}))
  return(data.set)
}

print('Loading datasets...')
train <- read.delim('data/train.tsv', sep="\t", header=TRUE)
test <- read.delim('data/public_leaderboard.tsv', sep="\t", header=TRUE)

print('Preprocessing...')
train$EssayText <- as.character(train$EssayText)
test$EssayText <- as.character(test$EssayText)

#This is to add features specific to the essay set
#train.1 <- train[which(train$EssaySet == 1),]

print('Adding general features...')
train <- add.gen.features(train)
test <- add.gen.features(test)

sas.formula = as.formula(paste('Score1 ~',
                               paste(names(train)[c(6:ncol(train))], collapse=" + ")))

print('Training model...')
print(' -- gbm')
gbm.model <- gbm(sas.formula, n.trees=5000, data=train,
             distribution='gaussian', interaction.depth=6,
             train.fraction=.8, cv.folds=5)

#print(' -- glm')
#glm.model <- glm(sas.formula, family=binomial)

#model <- randomForest(train[c('Length', 'Words', 'Sentences')], train$Score1, sampsize=nrow(train)*.3, ntree=1000)

print('Predicting...')
predictions <- predict(gbm.model, newdata=test, 5000)
round.predictions <- round(predictions)
round.predictions[which(round.predictions > 3)] <- 3

print('Saving prediction...')
submission <- data.frame(id=test$Id, essay_score=round.predictions)
write.csv(submission, file='submission.csv', row.names=FALSE)
