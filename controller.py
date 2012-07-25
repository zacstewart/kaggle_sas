from datasets import *
from features import *
from random_forest import *

if __name__ == "__main__":
  thead, th, trows = readFile('data/train_rel_2.tsv')
  lhead, lh, lrows = readFile('data/public_leaderboard_rel_2.tsv')
  for i in essaySets():
    print '''
    Modeling essay set %(set)2i
    =====================
    ''' % {'set': i}

    # Get all essays for train+lb
    strows = essaySet(i, trows, th)
    slrows = essaySet(i, lrows, lh)
    all_essays =  essayVec(strows, th)
    all_essays += essayVec(slrows, lh) # TODO: Try without this

    # Create corpus with all_essays
    print 'Building corpus...'
    w, my_corpus = getCorpus(all_essays)

    print 'Generating features for train and leadboard sets...'
    thead, tfh, tfeatures, _ = getFeatures(strows, my_corpus, th, w, ['EssaySet', 'Score1'])
    lhead, lfh, lfeatures, _ = getFeatures(slrows, my_corpus, lh, w, ['EssaySet', 'Score1'])

    lx = [row[lfh['Length']:] for row in lfeatures]

    print 'Training and cross validating models...'
    models = {'rf': RfModel()}
    scores = dict()
    k = 10
    folds = kFold(tfeatures, k)
    for mname, model in models.items():
      print '  == Model: ' + mname
      cv_scores = []
      for j in range(k):
        print "  - Fold %(fold)i..." % {'fold': j + 1}
        # Get train and lb rows for this essay set
        a = [row for rows in folds[:j] for row in rows]
        b = [row for rows in folds[j+1:] for row in rows]
        tffeatures = a + b
        cffeatures = folds[j]

        tx = [row[tfh['Length']:] for row in tffeatures]
        ty = [row[tfh['Score1']] for row in tffeatures]

        cx = [row[tfh['Length']:] for row in cffeatures]
        cy = [row[tfh['Score1']] for row in cffeatures]


        model.train(tx, ty)
        score = model.validate(cx, cy)
        cv_scores.append(score)
        print '  -- CV score: ' + str(score)

      cv_mean = sum(cv_scores) / len(cv_scores)
      print '  - Mean CV score for %(model)s: %(score)f' % {'model': mname, 'score': cv_mean}
      scores[mname] = cv_mean

    print 'Building super training set...'

    cv, train = cvSplit(tfeatures, .3)
    trainx = [row[tfh['Length']:] for row in train]
    trainy = [row[tfh['Score1']] for row in train]
    cvx = [row[tfh['Length']:] for row in cv]
    cvy = [row[tfh['Score1']] for row in cv]

    predictions = dict()
    supertrainhead = ['Id', 'Score1'] + models.keys()
    sth = dict(zip(supertrainhead, range(len(supertrainhead))))
    supertrain = []
    for mname, model in models.items():
      print '  == Model: ' + mname
      print '  -- training...'
      model.train(trainx, trainy)
      print '  -- predicting...'
      predictions[mname] = model.predict(cvx)
    for (i, row) in enumerate(cv):
      supertrainrow = [None for _ in range(len(supertrainhead))]
      supertrainrow[sth['Id']] = row[th['Id']]
      supertrainrow[sth['Score1']] = row[th['Score1']]
      for mname in models.keys():
        supertrainrow[sth[mname]] = predictions[mname][i]
