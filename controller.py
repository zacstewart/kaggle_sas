from datasets import *
from features import *
from train import *
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
import scipy
import numpy
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

    models = {
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

    models = [
      {'constructor': NearestCentroid,
        'params': dict(metric='euclidean', shrink_threshold=None),
        'name': 'Nearest Centroid', },
      {'constructor': GaussianNB,
        'params': dict(),
        'name': 'Gaussian Naive Bayes'},
      {'constructor': MultinomialNB,
        'params': dict(),
        'name': 'Multinominal Naive Bayes'},
      {'constructor': tree.DecisionTreeClassifier,
        'params': dict(),
        'name': 'Decision Tree Classifier'},
      {'constructor': LDA,
        'params': dict(),
        'name': 'lda'},
      {'constructor': LogisticRegression,
        'params': dict(penalty='l2', dual=False, tol=0.0001, C=1.0,
          fit_intercept=True, intercept_scaling=1, constructor_weight=None),
        'name': 'Logistic Regression'},
      {'constructor': RidgeClassifier,
        'params': dict(alpha=1.0, fit_intercept=True, normalize=False,
          copy_X=True, tol=0.001, constructor_weight=None),
        'name': 'Ridge Classifier'},
      {'constructor': svm.SVC,
        'params': dict(C=100.0, cache_size=200, constructor_weight=None, coef0=0.0, degree=3,
          gamma=0.001, kernel='rbf', probability=False, shrinking=True, tol=0.001, verbose=False),
        'name': 'SVC'},
      {'constructor': svm.LinearSVC,
        'params': dict(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0,
          multi_constructor='ovr', fit_intercept=True, intercept_scaling=1, constructor_weight=None, verbose=0),
        'name': 'Linear SVC'},
      {'constructor': RandomForestClassifier,
        'params': dict(n_estimators=150, min_samples_split=2, n_jobs=-1),
        'name': 'Random Forest Classifier'}]

    # Get all essays for train+lb
    strows = essaySet(i, trows, th)
    slrows = essaySet(i, lrows, lh)
    all_essays =  essayVec(strows, th)
    #all_essays += essayVec(slrows, lh) # TODO: Try without this


    stem_corpus_fn = "data/cache/stem_corpus_%(essay)2i.csv" % {'essay': i}
    tag_corpus_fn = "data/cache/tag_corpus_%(essay)2i.csv" % {'essay': i}
    if len(sys.argv) > 1 and sys.argv[1] == 'cached' and \
      os.path.isfile(stem_corpus_fn) and \
      os.path.isfile(tag_corpus_fn):
        print 'Loading saved corpuses...'
        my_corpus, w, _ = readFile(stem_corpus_fn, ',', strings=True)
        tag_corpus, t, _ = readFile(tag_corpus_fn, ',', strings=True)
    else:
      # Create corpus with all_essays
      print 'Tokenizing essays...'
      tokenized_essays = [getTokens(essay) for essay in all_essays]
      print 'Building corpus...'
      w, my_corpus = getCorpus(tokenized_essays)
      print 'Building tag corpus...'
      t, tag_corpus = getTagCorpus(tokenized_essays)
      writeFile(my_corpus, [], stem_corpus_fn)
      writeFile(tag_corpus, [], tag_corpus_fn)

    train_features_fn = "data/cache/train_features_%(essay)2i" % {'essay': i}
    lb_features_fn = "data/cache/lb_features_%(essay)2i" % {'essay': i}
    if len(sys.argv) > 1 and sys.argv[1] == 'cached' and \
      os.path.isfile(train_features_fn) and \
      os.path.isfile(lb_features_fn):
        print 'Loading saved features...'
        thead, tfh, tfeatures = readFile(train_features_fn, ',', numeric=True)
        lhead, lfh, lfeatures = readFile(lb_features_fn, ',', numeric=True)
    else:
      print 'Generating features for train and leadboard sets...'
      thead, tfh, tfeatures, _ = getFeatures(strows, my_corpus, tag_corpus, th, w, t, ['EssaySet', 'Score1'])
      lhead, lfh, lfeatures, _ = getFeatures(slrows, my_corpus, tag_corpus, lh, w, t, ['EssaySet', 'Score1'])
      writeFile(thead, tfeatures, train_features_fn)
      writeFile(lhead, lfeatures, lb_features_fn)

    lx = [row[lfh['Length']:] for row in lfeatures]

    print 'Training and cross validating models...'
    scores = dict()
    folds = kFold(tfeatures, k)
    for model in models:
      pnames = model['params'].keys()
      params = numpy.array(model['params'].values())
      optims = scipy.optimize.fmin_bfgs(lambda p: validate(model['constructor'], dict(zip(pnames, p)), folds), params, tick)
      model['optims'] = optims

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
    for model in models:
      mname = model['name']
      instance = model['constructor'](model['optims'])
      print '  == Model: ' + mname
      print '  -- training...'
      instance.fit(trainx, trainy)
      print '  -- predicting cv...'
      cv_predictions[mname] = instance.predict(cvx)
      print '  -- predicting lb...'
      lb_predictions[mname] = instance.predict(lx)

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
      print 'Mean super duper CV score: ' + str(mean_superduper_score)
      mean_superduper_qwk = sum(superduperqwks) / len(superduperqwks)
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
