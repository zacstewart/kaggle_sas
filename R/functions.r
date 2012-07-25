library('tm')

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

get.grade.prob <- function (essay, grade.df, prior=.25, c=1e-4) {
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
                                       length(unlist(strsplit(e, '[^[:alnum:]_]')))}))
  #print(' -- Misspellings')
  #data.set <- transform(data.set, Misspellings = get.misspellings(data.set$EssayText))
  #print(' -- Word rarity')
  #data.set <- transform(data.set, WordRarity = get.word.rarity(data.set$EssayText))
  print(' -- Bayes probabilities')
  grade0.df <- get.tdf(dat$EssayText[which(dat$Score1 == 0)])
  grade1.df <- get.tdf(dat$EssayText[which(dat$Score1 == 1)])
  grade2.df <- get.tdf(dat$EssayText[which(dat$Score1 == 2)])
  grade3.df <- get.tdf(dat$EssayText[which(dat$Score1 == 3)])
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
