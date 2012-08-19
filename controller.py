import sys
import pickle
import numpy as np
from datasets import *
from features import *
from learn import *

if __name__ == "__main__":
  # Read training and LB sets and initalize submission set
  all_training_rows, training_header = readFile('data/train_rel_2.tsv')
  all_leaderboard_rows, leaderboard_header = readFile('data/public_leaderboard_rel_2.tsv')
  all_submission_rows = []
  caching = len(sys.argv) > 1 and sys.argv[1] == 'cached' # use cached data
  k = 10 # For k-folds

  for essay_set in essaySets(all_training_rows, training_header):
    print \
      '''
      Modeling essay set %(essay_set)2i
      =====================
      ''' % dict(essay_set=essay_set)

    # Get training and LB rows for this essay set
    training_rows = essaySet(essay_set, all_training_rows, training_header)
    leaderboard_rows = essaySet(essay_set, all_leaderboard_rows, leaderboard_header)

    # Extract training essays
    training_essays = essayVec(training_rows, training_header)

    # Corpus cache filenames
    stem_corpus_file = 'data/cache/stem_corpus_' + str(essay_set) + '.pkl'
    tag_corpus_file = 'data/cache/tag_corpus_' + str(essay_set) + '.pkl'

    # Get tokens if I'm going to need them later
    if not caching \
        or not os.path.isfile(stem_corpus_file) \
        or not os.path.isfile(tag_corpus_file):
      training_tokens = [getTokens(essay) for essay in training_essays]

    # Build stem corpus or load from cache
    if caching and os.path.isfile(stem_corpus_file):
      print 'Loading cached corpuses...'
      f = open(stem_corpus_file, 'rb')
      stem_corpus = pickle.load(f)
      f.close()
    else:
      stem_corpus = getCorpus(training_tokens)
      if caching:
        print 'Caching corpuses...'
        f = open(stem_corpus_file, 'wb')
        pickle.dump(stem_corpus, f)
        f.close()

    # Feature cache filenames
    training_features_file = 'data/cache/training_features_' + str(essay_set) + '.pkl'
    leaderboard_features_file = 'data/cache/leaderboard_features_' + str(essay_set) + '.pkl'

    # Generate training features or load from cache
    if caching and os.path.isfile(training_features_file):
      print 'Loading cached features...'
      f = open(training_features_file, 'rb')
      training_features = pickle.load(f)
      f.close()
    else:
      training_features, training_features_header = \
          getFeatures(training_rows, training_header, stem_corpus, extras=['Id', 'Score1'])
      if caching:
        print 'Caching features...'
        f = open(training_features_file, 'wb')
        pickle.dump(training_features, f)
        f.close()

    if caching and os.path.isfile(leaderboard_features_file):
      print 'Loading cached features...'
      f = open(leaderboard_features_file, 'rb')
      leaderboard_features = pickle.load(f)
      f.close()
    else:
      leaderboard_features, leaderboard_features_header = \
          getFeatures(leaderboard_rows, leaderboard_header, stem_corpus, extras=['Id'])
      if caching:
        print 'Caching features...'
        f = open(leaderboard_features_file, 'wb')
        pickle.dump(leaderboard_features, f)
        f.close()

    # Get predictions from all models to be used as super features
    super_training_features, super_training_features_header, super_leaderboard_features, super_leaderboard_features_header = \
        getPredictions(training_features, training_features_header, leaderboard_features, leaderboard_features_header)

    model = getBestModel(super_training_features, super_training_features_header)

    lb_id = leaderboard_features[:, 0]
    preds = model.predict(super_leaderboard_features)

    sub_row = zip(lb_id, preds)
    print sub_row

    all_submission_rows += sub_row
    print len(all_submission_rows)
  writeFile(all_submission_rows, ['id', 'essay_score'], filename='submission.csv', delimiter=',')
