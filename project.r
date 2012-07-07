library('tm')
library('randomForest')

get.tdm <- function(doc.vec) {
  doc.corpus <- Corpus(VectorSource(doc.vec))
  control <- list(stopwords=TRUE, removePunctuation=TRUE, removeNumbers=TRUE, minDocFreq=2)
  doc.dtm <- TermDocumentMatrix(doc.corpus, control)
  return(doc.dtm)
}

add.gen.features <- function (data.set) {
  data.set$Length <- unlist(lapply(data.set$EssayText, nchar))
  data.set$Words <- unlist(lapply(data.set$EssayText, function (essay) {
                                 length(unlist(strsplit(essay, '[^[:alnum:]_]')))}))
  data.set$Sentences <- unlist(lapply(data.set$EssayText, function (essay) {
                                     length(unlist(strsplit(essay, '\\.')))}))
  return(data.set)
}

train <- read.delim('data/train.tsv', sep="\t", header=TRUE)
train$EssayText <- as.character(train$EssayText)
test <- read.delim('data/train.tsv', sep="\t", header=TRUE)
test$EssayText <- as.character(train$EssayText)

#This is to add features specific to the essay set
#train.1 <- train[which(train$EssaySet == 1),]

train <- add.gen.features(train)
test <- add.gen.features(test)

model <- randomForest(train[,6:ncol(train)], train$Score1, sampsize=nrow(train)*.3, ntree=1000)

predictions <- predict(model, test)
