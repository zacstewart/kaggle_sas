from datasets import *
from features import *
from ml_metrics import *
from sklearn import gaussian_process
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import *
from sklearn.lda import LDA
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.neighbors.nearest_centroid import NearestCentroid
import sys

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

    classifiers = {
        'ncn': NearestCentroid(metric='euclidean', shrink_threshold=None),
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
    regressors = {
        'gbr': GradientBoostingRegressor(n_estimators=100, learn_rate=1.0,
          max_depth=1, random_state=0, loss='ls'),
        'dtr': tree.DecisionTreeRegressor(),
        #'gp': gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1),
        'svr': svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3,
          epsilon=0.1, gamma=0.5, kernel='rbf', probability=False, shrinking=True,
          tol=0.001, verbose=False),
        'ridge': Ridge(alpha=0.5, copy_X=True, fit_intercept=True, normalize=False, tol=0.001),
        'knnr': KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto',
          leaf_size=30, warn_on_equidistant=True, p=2)}
    models = dict(classifiers.items() + regressors.items())

    # Get all essays for train+lb
    strows = essaySet(i, trows, th)
    slrows = essaySet(i, lrows, lh)
    all_essays =  essayVec(strows, th)
    #all_essays += essayVec(slrows, lh) # TODO: Try without this


    if len(sys.argv) > 1 and sys.argv[1] == 'cached':
      print 'Loading saved corpuses...'
      my_corpus, w, _ = readFile('data/stem_corpus.csv', ',', strings=True)
      tag_corpus, t, _ = readFile('data/tag_corpus.csv', ',', strings=True)
    else:
      # Create corpus with all_essays
      print 'Tokenizing essays...'
      tokenized_essays = [getTokens(essay) for essay in all_essays]
      print 'Building corpus...'
      w, my_corpus = getCorpus(tokenized_essays)
      print 'Building tag corpus...'
      t, tag_corpus = getTagCorpus(tokenized_essays)
      writeFile(my_corpus, [], 'data/stem_corpus.csv')
      writeFile(tag_corpus, [], 'data/tag_corpus.csv')

    if len(sys.argv) > 1 and sys.argv[1] == 'cached':
      print 'Loading saved features...'
      thead, tfh, tfeatures = readFile('data/train_features.csv', ',', numeric=True)
      lhead, lfh, lfeatures = readFile('data/lb_features.csv', ',', numeric=True)
    else:
      print 'Generating features for train and leadboard sets...'
      thead, tfh, tfeatures, _ = getFeatures(strows, my_corpus, tag_corpus, th, w, t, ['EssaySet', 'Score1'])
      lhead, lfh, lfeatures, _ = getFeatures(slrows, my_corpus, tag_corpus, lh, w, t, ['EssaySet', 'Score1'])
      writeFile(thead, tfeatures, 'data/train_features.csv')
      writeFile(lhead, lfeatures, 'data/lb_features.csv')

    lx = [row[lfh['Length']:] for row in lfeatures]

    print 'Training and cross validating models...'
    scores = dict()
    folds = kFold(tfeatures, k)
    for mname, model in models.items():
      cv_scores = []
      qwks = []
      rmses = []
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
        if mname in classifiers:
          qwk = quadratic_weighted_kappa(cy, p, 0, 3)
          qwks.append(qwk)
        else:
          my_rmse = rmse(cy, p)
          rmses.append(my_rmse)
        cv_scores.append(score)
      foldbar.finish()

      cv_mean = sum(cv_scores) / len(cv_scores)
      print '  - Mean score for %(model)s: %(score)f' % {'model': mname, 'score': cv_mean}
      scores[mname] = cv_mean
      if mname in classifiers:
        qwk_mean = sum(qwks) / len(qwks)
        print '  - Mean QWK for %(model)s: %(score)f' % {'model': mname, 'score': qwk_mean}
      if mname in regressors:
        rmse_mean = sum(rmses) / len(rmses)
        print '  - Mean RMSE for %(model)s: %(score)f' % {'model': mname, 'score': rmse_mean}

    print 'Building super training set...'

    cv, train = cvSplit(tfeatures, .3)
    trainx = [row[tfh['Length']:] for row in train]
    trainy = [row[tfh['Score1']] for row in train]
    cvx = [row[tfh['Length']:] for row in cv]
    cvy = [row[tfh['Score1']] for row in cv]

    cv_predictions = dict()
    lb_predictions = dict()
    supertrainhead = ['Id', 'Score1'] + classifiers.keys()
    superlbhead = ['Id'] + classifiers.keys()
    sth = dict(zip(supertrainhead, range(len(supertrainhead))))
    slh = dict(zip(superlbhead, range(len(superlbhead))))
    for mname, model in classifiers.items():
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
      for mname in classifiers.keys():
        supertrainrow[sth[mname]] = int(cv_predictions[mname][i])
      supertrain.append(supertrainrow)

    superlb = []
    for (i, row) in enumerate(lfeatures):
      superlbrow = [None for _ in range(len(superlbhead))]
      superlbrow[slh['Id']] = row[lh['Id']]
      for mname in classifiers.keys():
        superlbrow[slh[mname]] = int(lb_predictions[mname][i])
      superlb.append(superlbrow)

    print 'Cross validating super model...'
    folds = kFold(supertrain, k)
    modelscores = dict()
    for mname, model in classifiers.items():
      superduperscores = []
      superduperqwks = []
      superduperrmses = []
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
        if mname in classifiers:
          qwk = quadratic_weighted_kappa(cvy, preds, 0, 3)
          superduperqwks.append(qwk)
        elif mname in regressors:
          my_rmse = rmse(cvy, preds)
          superduperrmses.append(my_rmse)
      foldbar.finish()
      mean_superduper_score = sum(superduperscores) / len(superduperscores)
      print 'Mean super duper CV score: ' + str(mean_superduper_score)
      mean_superduper_qwk = sum(superduperqwks) / len(superduperqwks)
      print 'Mean super duper CV QWK: ' + str(mean_superduper_qwk)
      modelscores[mname] = mean_superduper_qwk

    bestmodel = sorted(modelscores.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]

    final_model = classifiers[bestmodel]
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
