from nltk import *
from progressbar import *
import sys
import operator

pwidgets = [Percentage(), ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]
punctuation = re.compile(r'[`\'-.?!,":;()|0-9]')

def isWord(token):
  '''Determines whether a token is a word'''
  return not bool(punctuation.search(token))

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

def getTokens(text):
  stemmer = PorterStemmer()
  tokens = word_tokenize(text)
  tokens = [stemmer.stem(token).lower() for token in tokens]
  tokens = list(set(tokens) - set(corpus.stopwords.words('english')))
  tokens = filter(isWord, tokens)
  return tokens

def getCorpus(essays, n=1000):
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
  outheader = ['Id'] + extras + ['Color'] + words
  oh = dict(zip(outheader, range(len(outheader))))
  discards = set()
  unavail_extras = set()
  print 'Creating features: ' + ', '.join(outheader)
  pbar = ProgressBar(widgets=pwidgets, maxval=len(rows))
  pbar.start()
  for (i, row) in enumerate(rows):
    pbar.update(i)
    features = [0 for j in range(len(outheader))]

    # Start with the Id
    features[oh['Id']] = row[h['Id']]

    # Copy extras from infile
    for extra in extras:
      try: features[oh[extra]] = row[h[extra]]
      except KeyError: unavail_extras.add(extra)

    if row[h['EssaySet']] == '10':
      color = row[h['EssayText']].split('::')[0].strip().lower()
      essay_text = row[h['EssayText']].split('::')[1].strip()
      if len(color) <= 0: color = -1
      elif len(color) > 0 and color[0] == '"': color = color[1:]
    else:
      color = -1
      essay_text = row[h['EssayText']]

    # Add the color as a feature
    features[oh['Color']] = color

    # Add words
    my_tokens = getTokens(essay_text)
    for token in my_tokens:
      try: features[oh[token]] += 1
      except KeyError: discards.add(token)

    # Cast all to strings
    features = [str(f) for f in features]
    set_features.append(features)
  pbar.finish()
  print "Discarded " + str(len(discards)) + " words"
  print "Unavailable extras: " + ", ".join(unavail_extras)
  return (outheader, set_features, discards)

if __name__ == "__main__":
  if len(sys.argv) >= 4:
    trainin   = sys.argv[1]
    testin    = sys.argv[2]
    trainout  = sys.argv[3]
    testout   = sys.argv[4]
    extras = sys.argv[5:] # Keep these extra columns from the infile

    all_essays = []
    for i in range(1, 3):
      f = open(sys.argv[i], 'rb')
      header = f.readline().strip().split("\t")
      h = dict(zip(header, range(len(header))))
      rows = [line.strip().split("\t") for line in f] # split lines into cols
      f.close()
      essays = getEssays(rows, h)
      all_essays += essays

    w, my_corpus = getCorpus(all_essays)

    for i in range(1, 3):
      f = open(sys.argv[i], 'rb')
      header = f.readline().strip().split("\t")
      h = dict(zip(header, range(len(header))))
      rows = [line.strip().split("\t") for line in f] # split lines into cols

      outheader, features, discards = getFeatures(rows, my_corpus, h, w, extras)

      header = outheader
      f = open(sys.argv[i+2], 'w')
      f.write(','.join(header) + "\n")
      for row in features:
        f.write(','.join(row) + "\n")
      f.close()
  else:
    print 'Usage: %(this)s TRAININ TESTIN TRAINOUT TESTOUT [COPY COLS...]' % {'this': sys.argv[0]}
