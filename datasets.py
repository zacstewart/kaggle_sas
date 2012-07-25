from random import *

######################
# Basic IO
######################
def parseLine(line, h):
  '''Splits one line into a list of values. Casts values according to
  the headermap, +h+.

  Keyword arguments:
  line -- a one-line string from a data file
  h    -- a header map produced from the first line of the datafile
  '''
  row = line.strip().split('\t')
  row[h['Id']] = int(row[h['Id']])
  try: row[h['Score1']] = int(row[h['Score1']])
  except: 'Ignoring Score1'
  row[h['EssaySet']] = int(row[h['EssaySet']])
  return row

def readFile(filename, delimiter="\t"):
  '''Reads a file and returns a header, a header index map and rows'''
  f = open(filename, 'rb')
  header = f.readline().strip().split('\t')
  headermap = dict(zip(header, range(len(header))))
  rows = [parseLine(line, headermap) for line in f]
  return (header, headermap, rows)

######################
# Specific stuff
######################
def essaySets(rows=None, h=None):
  '''Determines a unique set of essay sets'''
  TRAIN_FILE = 'data/train_rel_2.tsv'
  LEADERBOARD_FILE = 'data/public_leaderboard_rel_2.tsv'
  if rows is None or h is None: _, h, rows = readFile(TRAIN_FILE)
  return set(row[h['EssaySet']] for row in rows)

def essaySet(s, rows=None, h=None):
  '''Get the rows for a specific essay set'''
  TRAIN_FILE = 'data/train_rel_2.tsv'
  LEADERBOARD_FILE = 'data/public_leaderboard_rel_2.tsv'
  if rows is None or h is None: _, h, rows = readFile(TRAIN_FILE)
  return [row for row in rows if row[h['EssaySet']] == s]

def essayVec(rows=None, h=None):
  '''Return a list of just the essays in +rows+. useful for making
  a corpus'''
  if rows is None or h is None: _, h, rows = readFile(TRAIN_FILE)
  essays = []
  for (i, row) in enumerate(rows):
    if row[h['EssaySet']] == '10':
      essay = row[h['EssayText']].split('::')[1].strip()
    else:
      essay = row[h['EssayText']]
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
    if random() >= cvfrac:
      cv.append(row)
    else:
      train.append(row)
  return (cv, train)

if __name__ == '__main__':
  print essaySets()
