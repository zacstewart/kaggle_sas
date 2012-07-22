import nltk

# Id  EssaySet  Score1  Score2  EssayText
f = open('data/train_rel_2.tsv')
header = f.readline().strip().split("\t")
h = dict(zip(header, range(len(header))))
rows = [line.strip().split("\t") for line in f] # split lines into cols
f.close()

set_features = [['Id', 'Score1', 'EssaySet', 'Color']]
for (i, row) in enumerate(rows):
  print str(i) + "/" + str(len(rows))
  # Start with the Id
  features = [row[h['Id']]]

  # Add score
  features.append(row[h['Score1']])

  # Add essay set
  features.append(row[h['EssaySet']]) 

  # Add color if essay set 10
  if row[h['EssaySet']] == '10':
    color = row[h['EssayText']].split('::')[0].strip().lower()
    if len(color) > 0 and color[0] == '"':
      color = color[1:]
  else:
    color = '-1'
  features.append(color)

  #tokens = nltk.word_tokenize(row[2])
  set_features.append(features)

f = open('data/train_features.csv', 'w')
for row in set_features:
  f.write(','.join(row) + "\n")
f.close()
