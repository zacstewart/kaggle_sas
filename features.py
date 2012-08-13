from nltk import *
from progressbar import *
import string
import sys
import operator
import itertools

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
  ngrams = [dict() for i in range(n)]
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
  print ngrams
  return ngrams

def getTagCorpus(tokenized_essays, n=1000):
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
  return tags

def getFeatures(rows, words, tags, h, w, t, extras=[]):
  '''Create output header and feature rows.

  Keyword arguments:
  rows    -- Example rows
  corpus  -- Corpus generated from `getCorpus`
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
    features = dict(zip(outheader, [0 for j in range(len(outheader))]))

    if row['EssaySet'] == '10':
      color = row['EssayText'].split('::')[0].strip().lower()
      essay_text = row['EssayText'].split('::')[1].strip()
      if len(color) <= 0: color = None
      elif len(color) > 0 and color[0] == '"': color = color[1:]
    else:
      color = -1
      essay_text = row['EssayText']

    my_tokens = getTokens(essay_text)
    my_stems = getStems(my_tokens)
    my_tags = getTags(my_tokens)

    # Start with the Id
    features['Id'] = row['Id']

    # Copy extras from infile
    for extra in extras:
      try: features[extra] = row[extra]
      except KeyError: unavail_extras.add(extra)

    # Add the color as a feature
    features['Color'] = color

    # Add character length
    features['Length'] = len(row['EssayText'])
    features['Stems'] = len(my_stems)
    features['Tags'] = len(my_tags)
    wordlens = map(lambda w: len(w), my_tokens)
    features['LongestWord'] = max(wordlens)
    features['AvgWord'] = sum(wordlens) / len(wordlens)

    features['PunctCt'] = 0
    for char in list(row['EssayText']):
      if char in set(string.punctuation): features['PunctCt'] += 1

    # Add words
    for stem in my_stems:
      if stem in oh:
        features[stem] += 1

    # Add parts of speech
    for tag in my_tags:
      if tag in oh:
        features[tag] += 1

    features = [f for f in features]

    #Throw out all the guys who didn't give me a color >:[
    if not row['EssaySet'] == '10' or not color is None:
      set_features.append(features)
  pbar.finish()
  return (set_features, discards)
