from datasets import *
from features import *
from sklearn.ensemble import *
from sklearn.lda import LDA
from sklearn.naive_bayes import *
from sklearn.linear_model import *
from sklearn import tree
from sklearn import svm
from ml_metrics import *

if __name__ == "__main__":
  thead, th, trows = readFile('data/train_rel_2.tsv')
  lhead, lh, lrows = readFile('data/public_leaderboard_rel_2.tsv')
  all_sub_rows = []
  k = 10
  for i in essaySets():
    print '''
    Modeling essay set %(set)2i
    =====================
    ''' % {'set': i}

    # Get all essays for train+lb
    strows = essaySet(i, trows, th)
    slrows = essaySet(i, lrows, lh)
    all_essays =  essayVec(strows, th)
    #all_essays += essayVec(slrows, lh) # TODO: Try without this

    # Create corpus with all_essays
    print 'Tokenizing essays...'
    tokenized_essays = [getTokens(essay) for essay in all_essays]
    print 'Building corpus...'
    w, my_corpus = getCorpus(tokenized_essays)
    print 'Building tag corpus...'
    t, tag_corpus = getTagCorpus(tokenized_essays)

    print 'Generating features for train and leadboard sets...'
    thead, tfh, tfeatures, _ = getFeatures(strows, my_corpus, tag_corpus, th, w, t, ['EssaySet', 'Score1'])
    lhead, lfh, lfeatures, _ = getFeatures(slrows, my_corpus, tag_corpus, lh, w, t, ['EssaySet', 'Score1'])

    lx = [row[lfh['Length']:] for row in lfeatures]

    print 'Training and cross validating models...'
    models = {
        'gnb': GaussianNB(),
        'mnb': MultinomialNB(),
        'dtc': tree.DecisionTreeClassifier(),
        'lda': LDA(),
        'lr': LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
          fit_intercept=True, intercept_scaling=1, class_weight=None),
        'rc': RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False,
          copy_X=True, tol=0.001, class_weight=None),
        'svc': svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
          gamma=0.001, kernel='rbf', probability=False, shrinking=True, tol=0.001,
          verbose=False),
        'lsvc': svm.LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0,
          multi_class='ovr', fit_intercept=True, intercept_scaling=1,
          class_weight=None, verbose=0),
        'rf': RandomForestClassifier(n_estimators=150, min_samples_split=2, n_jobs=-1)}
    scores = dict()
    folds = kFold(tfeatures, k)
    for mname, model in models.items():
      cv_scores = []
      qwks = []
      pwidgets =  [mname, ' Fold ', Counter(), ' ', ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]
      foldbar = ProgressBar(widgets=pwidgets, maxval=k)
      foldbar.start()
      for i, j in enumerate(range(k)):
        foldbar.update(i)
          # Get train and lb rows for this essay set
        a = [row for rows in folds[:j] for row in rows]
        b = [row for rows in folds[j+1:] for row in rows]
        tffeatures = a + b
        cffeatures = folds[j]

        tx = [row[tfh['Length']:] for row in tffeatures]
        ty = [row[tfh['Score1']] for row in tffeatures]

        cx = [row[tfh['Length']:] for row in cffeatures]
        cy = [row[tfh['Score1']] for row in cffeatures]

        model.fit(tx, ty)
        score = model.score(cx, cy)
        p = model.predict(cx)
        p = [int(pe) for pe in p]
        qwk = quadratic_weighted_kappa(cy, p, 0, 3)
        cv_scores.append(score)
        qwks.append(qwk)
      foldbar.finish()

      cv_mean = sum(cv_scores) / len(cv_scores)
      qwk_mean = sum(qwks) / len(qwks)
      print '  - Mean score for %(model)s: %(score)f' % {'model': mname, 'score': cv_mean}
      print '  - Mean QWK for %(model)s: %(score)f' % {'model': mname, 'score': qwk_mean}
      scores[mname] = cv_mean

    print 'Building super training set...'

    cv, train = cvSplit(tfeatures, .3)
    trainx = [row[tfh['Length']:] for row in train]
    trainy = [row[tfh['Score1']] for row in train]
    cvx = [row[tfh['Length']:] for row in cv]
    cvy = [row[tfh['Score1']] for row in cv]

    cv_predictions = dict()
    lb_predictions = dict()
    supertrainhead = ['Id', 'Score1'] + models.keys()
    superlbhead = ['Id'] + models.keys()
    sth = dict(zip(supertrainhead, range(len(supertrainhead))))
    slh = dict(zip(superlbhead, range(len(superlbhead))))
    for mname, model in models.items():
      print '  == Model: ' + mname
      print '  -- training...'
      model.fit(trainx, trainy)
      print '  -- predicting cv...'
      cv_predictions[mname] = model.predict(cvx)
      print '  -- predicting lb...'
      lb_predictions[mname] = model.predict(lx)

    supertrain = []
    for (i, row) in enumerate(cv):
      supertrainrow = [None for _ in range(len(supertrainhead))]
      supertrainrow[sth['Id']] = row[th['Id']]
      supertrainrow[sth['Score1']] = row[th['Score1']]
      for mname in models.keys():
        supertrainrow[sth[mname]] = int(cv_predictions[mname][i])
      supertrain.append(supertrainrow)

    superlb = []
    for (i, row) in enumerate(lfeatures):
      superlbrow = [None for _ in range(len(superlbhead))]
      superlbrow[slh['Id']] = row[lh['Id']]
      for mname in models.keys():
        superlbrow[slh[mname]] = int(lb_predictions[mname][i])
      superlb.append(superlbrow)

    print 'Cross validating super model...'
    folds = kFold(supertrain, k)
    modelscores = dict()
    for mname, model in models.items():
      superduperscores = []
      superduperqwks = []
      pwidgets =  [mname, ' Fold ', Counter(), ' ', ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]
      foldbar = ProgressBar(widgets=pwidgets, maxval=k)
      foldbar.start()
      for i, j in enumerate(range(k)):
        foldbar.update(i)
        a = [row for rows in folds[:j] for row in rows]
        b = [row for rows in folds[j+1:] for row in rows]
        superdupertrain = a + b
        superdupercv = folds[j]
        trainx = [row[2:] for row in superdupertrain]
        trainy = [row[1] for row in superdupertrain]
        cvx = [row[2:] for row in superdupercv]
        cvy = [row[1] for row in superdupercv]
        model.fit(trainx, trainy)
        score = model.score(cvx, cvy)
        superduperscores.append(score)
        preds = model.predict(cvx)
        preds = [int(p) for p in preds]
        qwk = quadratic_weighted_kappa(cvy, preds, 0, 3)
        superduperqwks.append(qwk)
      foldbar.finish()
      mean_superduper_score = sum(superduperscores) / len(superduperscores)
      mean_superduper_qwk = sum(superduperqwks) / len(superduperqwks)
      print 'Mean super duper CV score: ' + str(mean_superduper_score)
      print 'Mean super duper CV QWK: ' + str(mean_superduper_qwk)
      modelscores[mname] = mean_superduper_qwk

    bestmodel = sorted(modelscores.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]

    final_model = models[bestmodel]
    final_trainx = [row[2:] for row in supertrain]
    final_trainy = [row[1] for row in supertrain]
    final_lbx = [row[1:] for row in superlb]
    final_lb_ids = [row[0] for row in superlb]
    final_model.fit(final_trainx, final_trainy)
    predictions = final_model.predict(final_lbx)
    sub_rows = zip(final_lb_ids, predictions)
    all_sub_rows += sub_rows
  sub_header = ['id', 'essay_score']
  writeFile(sub_header, all_sub_rows, 'submission.csv')
