from nltk import *
from progressbar import *
import string
import sys
import operator

stripper =  string.punctuation + string.whitespace
stemmer  =  PorterStemmer()
stopset  =  set(corpus.stopwords.words('english'))
stopset  |= set(stripper)
stopset  |= set('')
def getTokens(text):
  tokens = word_tokenize(text)
  tokens = [token for token in tokens if len(token) > 0]
  return tokens

def getStems(tokens):
  tokens = [token.strip(stripper).lower() for token in tokens]
  tokens = [stemmer.stem(token) for token in tokens if token not in stopset]
  tokens = [token for token in tokens if len(token) > 0]
  return tokens

punct = re.compile(r'^[\'"`~!@$\.,]*$').match
def getTags(tokens):
  tags = pos_tag(tokens)
  tags = [tag[1] for tag in tags]
  return tags

def getCorpus(tokenized_essays, n=200):
  ''' Generate a corpus of words used in example rows
  sorted by most popular and truncated at an arbitrary limit, n.

  Keywords arguments:
  essays  -- Essay vector
  n       -- Max words to include in corpus
  '''
  m = len(tokenized_essays)
  pwidgets =  ['Essay ', Counter(), '/', str(m), ' ', Percentage(), ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]
  pbar = ProgressBar(widgets=pwidgets, maxval=m)
  pbar.start()
  words = dict()
  for (i, essay) in enumerate(tokenized_essays):
    pbar.update(i)
    tokens = set(getStems(essay)) # Uniq tokens
    for token in tokens:
      if not token in words: words[token] = 0
      words[token] += 1
  pbar.finish()
  words = sorted(words.iteritems(), key=operator.itemgetter(1), reverse=True)
  words = words[:n] # I wish I could keep them all, but I can't! :(
  words = [word[0] for word in words]
  w = dict(zip(words, range(len(words))))
  return (w, words)

def getTagCorpus(tokenized_essays, n=200):
  '''Same as get corpus, but return parts of speech'''
  m = len(tokenized_essays)
  pwidgets =  ['Essay ', Counter(), '/', str(m), ' ', Percentage(), ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]
  pbar = ProgressBar(widgets=pwidgets, maxval=m)
  pbar.start()
  tags = dict()
  for (i, essay) in enumerate(tokenized_essays):
    pbar.update(i)
    essay_tags = set(getTags(essay))
    for tag in essay_tags:
      if not tag in tags: tags[tag] = 0
      tags[tag] += 1
  pbar.finish()
  tags = sorted(tags.iteritems(), key=operator.itemgetter(1), reverse=True)
  tags = tags[:n] # Keep n
  tags = [tag[0] for tag in tags]
  t = dict(zip(tags, range(len(tags))))
  return (t, tags)

def getFeatures(rows, words, tags, h, w, t, extras=[]):
  '''Create output header and feature rows.

  Keyword arguments:
  rows    -- Example rows
  words   -- Corpus generated from `getCorpus`
  h       -- Header dict
  w       -- Word-Corpus dict
  extras  -- Columns to copy from infile to outfile
  '''
  set_features = []
  outheader = ['Id'] + extras + ['Color', 'Length', 'Stems', 'Tags', 'PunctCt', 'LongestWord', 'AvgWord'] + words + tags
  oh = dict(zip(outheader, range(len(outheader))))
  discards = set()
  unavail_extras = set()
  m = len(rows)
  pwidgets =  ['Row ', Counter(), '/', str(m), ' ', Percentage(), ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]
  pbar = ProgressBar(widgets=pwidgets, maxval=len(rows))
  pbar.start()
  for (i, row) in enumerate(rows):
    pbar.update(i)
    features = [0 for j in range(len(outheader))]

    if row[h['EssaySet']] == '10':
      color = row[h['EssayText']].split('::')[0].strip().lower()
      essay_text = row[h['EssayText']].split('::')[1].strip()
      if len(color) <= 0: color = None
      elif len(color) > 0 and color[0] == '"': color = color[1:]
    else:
      color = -1
      essay_text = row[h['EssayText']]

    my_tokens = getTokens(essay_text)
    my_stems = getStems(my_tokens)
    my_tags = getTags(my_tokens)

    # Start with the Id
    features[oh['Id']] = row[h['Id']]

    # Copy extras from infile
    for extra in extras:
      try: features[oh[extra]] = row[h[extra]]
      except KeyError: unavail_extras.add(extra)

    # Add the color as a feature
    features[oh['Color']] = color

    # Add character length
    features[oh['Length']] = len(row[h['EssayText']])
    features[oh['Stems']] = len(my_stems)
    features[oh['Tags']] = len(my_tags)
    wordlens = map(lambda w: len(w), my_tokens)
    features[oh['LongestWord']] = max(wordlens)
    features[oh['AvgWord']] = sum(wordlens) / len(wordlens)

    features[oh['PunctCt']] = 0
    for char in list(row[h['EssayText']]):
      if char in set(string.punctuation): features[oh['PunctCt']] += 1

    # Add words
    for stem in my_stems:
      if stem in oh:
        features[oh[stem]] += 1

    # Add parts of speech
    for tag in my_tags:
      if tag in oh:
        features[oh[tag]] += 1

    features = [f for f in features]

    #Throw out all the guys who didn't give me a color >:[
    if not row[h['EssaySet']] == '10' or not color is None:
      set_features.append(features)
  pbar.finish()
  outheader = [col for col in outheader]
  outheadermap = dict(zip(outheader, range(len(outheader))))
  return (outheader, outheadermap, set_features, discards)

if __name__ == "__main__":
  if len(sys.argv) >= 4:
    trainin   = sys.argv[1]
    testin    = sys.argv[2]
    extras = sys.argv[3:] # Keep these extra columns from the infile

    ftrain = open(trainin, 'rb')
    ftest  = open(testin, 'rb')

    train_header = ftrain.readline().strip().split("\t")
    test_header = ftest.readline().strip().split("\t")
    htrain = dict(zip(train_header, range(len(train_header))))
    htest = dict(zip(test_header, range(len(test_header))))

    trainrows = [line.strip().split("\t") for line in ftrain]
    testrows = [line.strip().split("\t") for line in ftest]

    ftrain.close()
    ftest.close()

    for i in range(1, 11):
      print '''
      Building features for essay set %(set)i
      =======================================
      ''' % {'set': i}
      essays = []

      my_trainrows = [row for row in trainrows if row[htrain['EssaySet']] == str(i)]
      my_testrows = [row for row in testrows if row[htest['EssaySet']] == str(i)]

      # Get corpus for (train+test) essays from set i
      essays += essayVec(my_trainrows, htrain)
      essays += essayVec(my_testrows, htest)
      w, my_corpus = getCorpus(essays)

      trainoh, trainfeatures, traindiscards = getFeatures(my_trainrows, my_corpus, htrain, w, extras)
      testoh, testfeatures, testdiscards = getFeatures(my_testrows, my_corpus, htest, w, extras)

      ftrain = open('data/features/train_' + str(i) + '.csv', 'w')
      ftest = open('data/features/test_' + str(i) + '.csv', 'w')

      ftrain.write(','.join(trainoh) + "\n")
      for row in trainfeatures:
        ftrain.write(','.join(row) + "\n")
      ftrain.close()

      ftest.write(','.join(testoh) + "\n")
      for row in testfeatures:
        ftest.write(','.join(row) + "\n")
      ftest.close()
  else:
    print 'Usage: %(this)s TRAININ TESTIN TRAINOUT TESTOUT [COPY COLS...]' % {'this': sys.argv[0]}
