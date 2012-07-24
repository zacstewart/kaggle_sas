from nltk import *
from progressbar import *
import string
import sys
import operator

pwidgets = [Percentage(), ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]

def getEssays(rows, h):
  '''Gets essays from rows'''
  print 'Gettings essays from %(rows)i rows...' % {'rows': len(rows)}
  essays = []
  pbar = ProgressBar(widgets=pwidgets, maxval=len(rows))
  pbar.start()
  for (i, row) in enumerate(rows):
    pbar.update(i)
    if row[h['EssaySet']] == '10':
      essay = row[h['EssayText']].split('::')[1].strip()
    else:
      essay = row[h['EssayText']]
    essays.append(essay)
  pbar.finish()
  return essays

stripper  =  string.punctuation + string.whitespace
stemmer   =  PorterStemmer()
stopset   =  set(corpus.stopwords.words('english'))
stopset   |= set(stripper)
stopset   |= set('')
def getTokens(text):
  tokens = word_tokenize(text)
  tokens = [token.strip(stripper).lower() for token in tokens]
  tokens = [stemmer.stem(token) for token in tokens if token not in stopset]
  tokens = [token for token in tokens if len(token) > 0]
  return tokens

def getCorpus(essays, n=200):
  ''' Generate a corpus of words used in example rows
  sorted by most popular and truncated at an arbitrary limit, n.

  Keywords arguments:
  essays  -- Essay vector
  n       -- Max words to include in corpus
  '''
  print "Creating corpus from %(essays)i essays..." % {'essays': len(essays)}
  pbar = ProgressBar(widgets=pwidgets, maxval=len(essays))
  pbar.start()
  words = dict()
  for (i, essay) in enumerate(essays):
    pbar.update(i)
    tokens = set(getTokens(essay)) # Uniq tokens
    for token in tokens:
      try: words[token] += 1
      except: words[token] = 1
  pbar.finish()
  words = sorted(words.iteritems(), key=operator.itemgetter(1), reverse=True)
  words = words[:n] # I wish I could keep them all, but I can't! :(
  words = [word[0] for word in words]
  w = dict(zip(words, range(len(words))))
  print "Words: " + str(len(words))
  return (w, words)

def getFeatures(rows, words, h, w, extras=[]):
  '''Create output header and feature rows.

  Keyword arguments:
  rows    -- Example rows
  words   -- Corpus generated from `getCorpus`
  h       -- Header dict
  w       -- Word-Corpus dict
  extras  -- Columns to copy from infile to outfile
  '''
  set_features = []
  outheader = ['Id'] + extras + ['Color', 'Length', 'Tokens'] + words
  oh = dict(zip(outheader, range(len(outheader))))
  discards = set()
  unavail_extras = set()
  print 'Creating features: ' + ', '.join(outheader[3:15]) + '...'
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

    features[oh['Tokens']] = len(my_tokens)

    # Add words
    for token in my_tokens:
      try: features[oh[token]] += 1
      except KeyError: discards.add(token)

    # Cast all to strings
    features = [str(f) for f in features]

    #Throw out all the guys who didn't give me a color >:[
    if not row[h['EssaySet']] == '10' or not color is None:
      set_features.append(features)
  pbar.finish()
  outheader = ['"' + col + '"' for col in outheader]
  print "Discarded " + str(len(discards)) + " words"
  print "Unavailable extras: " + ", ".join(unavail_extras)
  return (outheader, set_features, discards)

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
      essays += getEssays(my_trainrows, htrain)
      essays += getEssays(my_testrows, htest)
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
