from helpers import *
from nltk import *
from progressbar import *
import numpy as np
import string
import sys
import operator
import itertools

stripper  = string.punctuation + string.whitespace
stemmer   = PorterStemmer()
stopset   = set(corpus.stopwords.words('english'))
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
  tags = [tag[1] for tag in tags if not punct(tag[1])]
  return tags

def getCorpus(tokenized_essays, n=2, c=1000):
  ''' Generate a corpus of n-grams used in example rows
  sorted by most popular and truncated at an arbitrary limit, n.

  Keywords arguments:
  essays  -- List of lists
  n       -- n in n-grams
  c       -- Max words to include in corpus
  '''
  m = len(tokenized_essays)
  pwidgets =  ['Essay ', Counter(), '/', str(m), ' ', Percentage(), ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]
  pbar = ProgressBar(widgets=pwidgets, maxval=m)
  pbar.start()
  ngrams = [dict() for _ in range(n)]
  for (i, essay) in enumerate(tokenized_essays):
    pbar.update(i)
    tokens = getStems(essay) # Uniq tokens
    for g in range(n):
      for t in range(len(tokens)):
        gram = ' '.join(tokens[t:t+g+1])
        if not gram in ngrams[g]: ngrams[g][gram] = 0
        ngrams[g][gram] += 1

  for g in range(n):
    ngrams[g] = sorted(ngrams[g].iteritems(), key=operator.itemgetter(1), reverse=True)
    ngrams[g] = ngrams[g][:c]
    ngrams[g] = [ngram[0] for ngram in ngrams[g]]
  pbar.finish()

  # Flatten and return
  ngrams = list(itertools.chain(*ngrams))
  return ngrams

def getFeatures(rows, header, stem_corpus, n=2, extras=[]):
  '''Create training features from rows. Returns list of dicts.

  Keyword arguments:
  rows    -- Example rows (list of dicts)
  stem_corpus  -- Corpus generated from `getCorpus`
  extras  -- Columns to copy from infile to outfile

  TODO: arrayify this
  '''
  h = toMap(header)
  feature_names = extras + ['Color', 'Length', 'Stems', 'Tags', 'PunctCt', 'LongestWord', 'AvgWord'] + stem_corpus
  f = toMap(feature_names)
  features = np.array([[0 for _ in range(len(feature_names))] for _ in range(len(rows))])
  discards = set()
  unavail_extras = set()
  m = len(rows)
  pwidgets =  ['Row ', Counter(), '/', str(m), ' ', Percentage(), ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]
  pbar = ProgressBar(widgets=pwidgets, maxval=len(rows))
  pbar.start()
  for (i, row) in enumerate(rows):
    pbar.update(i)

    if row[h['EssaySet']] == '10':
      color = row[h['EssayText']].split('::')[0].strip().lower()
      essay_text = row[h['EssayText']].split('::')[1].strip()
      if len(color) <= 0: color = None
      elif len(color) > 0 and color[0] == '"': color = color[1:]
    else:
      color = 0
      essay_text = row[h['EssayText']]

    # Get tokens and stems
    tokens = getTokens(essay_text)
    stems = getStems(tokens)

    # Copy extras from infile
    for extra in extras:
      try: features[i, f[extra]] = row[h[extra]]
      except KeyError: unavail_extras.add(extra)

    # Add the color as a feature
    features[i, f['Color']] = color

    features[i, f['Length']] = len(row[h['EssayText']]) # Character length
    features[i, f['Stems']] = len(stems) # Stem count
    wordlens = map(lambda w: len(w), tokens)
    features[i, f['LongestWord']] = max(wordlens) # Longest word length
    features[i, f['AvgWord']] = sum(wordlens) / len(wordlens) # Average word length

    # Punctuation count
    features[i, f['PunctCt']] = 0
    for char in list(row[h['EssayText']]):
      if char in set(string.punctuation): features[i, f['PunctCt']] += 1

    # Add n-grams
    for g in range(n):
      for t in range(len(stems)):
        gram = ' '.join(stems[t:t+g+1])
        if gram in feature_names: features[i, f[gram]] += 1

    #Throw out all the guys who didn't give me a color >:[
    #if not row[h['EssaySet']] == '10' or not color is None:
      #set_features.append(features)
  pbar.finish()
  return (features, feature_names)
