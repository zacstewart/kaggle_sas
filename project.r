library('tm')
library('randomForest')
load('lexical_database.Rdata')

get.tdm <- function(doc.vec) {
  doc.corpus <- Corpus(VectorSource(doc.vec))
  control <- list(stopwords=TRUE, removePunctuation=TRUE, removeNumbers=TRUE, minDocFreq=2)
  doc.dtm <- TermDocumentMatrix(doc.corpus, control)
  return(doc.dtm)
}

get.tdf <- function(doc.vec) {
  tdm <- get.tdm(doc.vec)
  tdf.matrix <- as.matrix(tdm)
  tdf.counts <- rowSums(tdf.matrix)
  tdf.df <- data.frame(cbind(names(tdf.counts), as.numeric(tdf.counts)),
                       stringsAsFactors=FALSE)
  names(tdf.df) <- c('term', 'frequency')
  return(tdf.df)
}

get.misspellings <- function (essay) {
  essay.tdm <- get.tdm(essay)
  essay.matrix <- as.matrix(essay.tdm)
  essay.counts <- rowSums(essay.matrix)
  essay.df <- data.frame(cbind(names(essay.counts), as.numeric(essay.counts)),
                         stringsAsFactors=FALSE)
  names(essay.df) <- c('term', 'frequency')
  essay.df$misspelled <- unlist(lapply(essay.df$term, function (term) {
                                       freq <- lexical.database[[term]]
                                       ifelse(is.null(freq) || is.na(freq), 1, 0)}))
  return(sum(essay.df$misspelled))
}

get.word.rarity <- function (essay.vec) {
  return(log(sapply(essay.vec, function (essay) {
                my.tdf <- get.tdf(essay)
                my.tdf$probability <- sapply(my.tdf$term, function (term) {
                                             prob <- lexical.database[[term]]
                                             return(ifelse(is.null(prob) || is.na(prob), 1, prob))})
                ifelse(length(my.tdf$probability) < 1, 1, prod(my.tdf$probability))})))
}

add.gen.features <- function (data.set) {
  print(' -- Various counts')
  data.set <- transform(data.set,
                        Length = sapply(data.set$EssayText, nchar),
                        Words = sapply(data.set$EssayText, function (e) {
                                       length(unlist(strsplit(e, '[^[:alnum:]_]')))}),
                        Sentences = sapply(data.set$EssayText, function (e) {
                                           length(unlist(strsplit(e, '\\.')))}))
  print(' -- Word "rarity"')
  data.set <- transform(data.set, WordRarity = get.word.rarity(data.set$EssayText))
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

#print('Training model...')
#model <- randomForest(train[c('Length', 'Words', 'Sentences')], train$Score1, sampsize=nrow(train)*.3, ntree=1000)

#print('Predicting...')
#predictions <- predict(model, test)
#round.predictions <- round(predictions)

#submission <- data.frame(id=test$Id, essay_score=round.predictions)
##write.csv(submission, file='submission.csv', row.names=FALSE)
