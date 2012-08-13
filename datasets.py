from random import *

######################
# Basic IO
######################
def formatRow(row):
  '''Splits one line into a list of values.'

  Keyword arguments:
  row -- a row dict
  '''
  row['Id'] = int(row['Id'])
  if 'Score1' in row: row['Score1'] = int(row['Score1'])
  if 'Score2' in row: row['Score2'] = int(row['Score1'])
  row['EssaySet'] = int(row['EssaySet'])
  return row

def readFile(filename, delimiter="\t", vtype=None):
  '''Reads a file and returns a list of dicts'''
  f = open(filename, 'rb')
  header = f.readline().strip().split(delimiter)
  if vtype == 'numeric':
    rows = [dict(zip(header, [int(v) for v in line.strip().split(delimiter)])) for line in f]
  elif vtype == 'strings':
    rows = [dict(zip(header, [str(v) for v in line.strip().split(delimiter)])) for line in f]
  else:
    rows = [formatRow(dict(zip(header, [v for v in line.strip().split(delimiter)]))) for line in f]
  return rows

def writeFile(rows, filename):
  header = rows[0].keys()
  f = open(filename, 'wb')
  f.write(','.join(header) + '\n')
  for row in rows:
    row = [str(int(v)) for v in row.values()]
    f.write(','.join(row) + '\n')
  f.close()

######################
# Specific stuff
######################
def essaySets(rows=None):
  '''Determines a unique set of essay sets'''
  TRAIN_FILE = 'data/train_rel_2.tsv'
  LEADERBOARD_FILE = 'data/public_leaderboard_rel_2.tsv'
  if rows is None: rows = readFile(TRAIN_FILE)
  return set(row['EssaySet'] for row in rows)

def essaySet(s, rows):
  '''Get the rows for a specific essay set'''
  if rows is None: rows = readFile(TRAIN_FILE)
  return [row for row in rows if row['EssaySet'] == s]

def essayVec(rows):
  '''Return a list of just the essays in +rows+. useful for making
  a corpus'''
  essays = []
  for (i, row) in enumerate(rows):
    if row['EssaySet'] == '10':
      essay = row['EssayText'].split('::')[1].strip()
    else:
      essay = row['EssayText']
    essays.append(essay)
  return essays

def kFold(rows, k):
  '''Split rows in to k folds. Returns list of lists.'''
  n = len(rows)
  folds = [[] for i in range(k)]
  for row in rows:
    i = int(round(random()*k)) - 1
    folds[i].append(row)
  return folds

def cvSplit(rows, cvfrac):
  n = len(rows)
  cv = []
  train = []
  for row in rows:
    if random() <= cvfrac:
      cv.append(row)
    else:
      train.append(row)
  return (cv, train)
