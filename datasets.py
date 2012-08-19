from helpers import *
from random import *
import numpy as np

######################
# Basic IO
######################
def formatRow(line, header, delimiter="\t"):
  '''Splits one line into a list of values.

  Keyword arguments:
  line   -- a line
  header -- a header
  '''
  row = line.split(delimiter)
  h = toMap(header)
  row[h['Id']] = int(row[h['Id']])
  if 'Score1' in row: row[h['Score1']] = int(row[h['Score1']])
  if 'Score2' in row: row['Score2'] = int(row[h['Score1']])
  row[h['EssaySet']] = int(row[h['EssaySet']])
  return row

def readFile(filename, delimiter="\t", vtype=None):
  '''Reads a file and returns a list of lists and header list'''
  f = open(filename, 'rb')
  header = f.readline().strip().split(delimiter)
  if vtype == 'numeric':
    rows = [[int(v) for v in line.strip().split(delimiter)] for line in f]
  elif vtype == 'strings':
    rows = [[str(v) for v in line.strip().split(delimiter)] for line in f]
  else:
    rows = [formatRow(line.strip(), header) for line in f]
  f.close()
  return (rows, header)

def writeFile(rows=None, header=[], filename=None, delimiter="\t"):
  f = open(filename, 'wb')
  f.write(delimiter.join(header) + "\n")
  for row in rows:
    row = [str(col) for col in row]
    f.write(delimiter.join(row) + "\n")
  f.close()

######################
# Specific stuff
######################
def essaySets(rows, header):
  '''Determines a unique set of essay sets'''
  h = toMap(header)
  return set(row[h['EssaySet']] for row in rows)

def essaySet(s, rows, header):
  '''Get the rows for a specific essay set'''
  h = toMap(header)
  return [row for row in rows if row[h['EssaySet']] == s]

def essayVec(rows, header):
  '''Return a list of just the essays in +rows+. useful for making
  a corpus'''
  h = toMap(header)
  essays = []
  for (i, row) in enumerate(rows):
    if row[h['EssaySet']] == '10':
      essay = row[h['EssayText']].split('::')[1].strip()
    else:
      essay = row[h['EssayText']]
    essays.append(essay)
  return essays
