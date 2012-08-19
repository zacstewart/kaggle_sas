from helpers import *
from datasets import *
from ml_metrics import *
from scipy.optimize import *
from sklearn import cross_validation
from sklearn import gaussian_process
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import *
from sklearn.lda import LDA
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.neighbors.nearest_centroid import NearestCentroid

MODELS = [
  {'constructor': NearestCentroid,
    'params': dict(),
    'name': 'Nearest Centroid'},
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
    'name': 'Linear Discriminant Analysis'},
  {'constructor': LogisticRegression,
    'params': dict(tol=0.0001, C=1.0, intercept_scaling=1),
    'name': 'Logistic Regression'},
  {'constructor': RidgeClassifier,
    'params': dict(alpha=1.0, tol=0.001),
    'name': 'Ridge Classifier'},
  {'constructor': svm.SVC,
    'params': dict(C=100.0, cache_size=200, coef0=0.0, degree=3,
      gamma=0.001, tol=0.001),
    'name': 'SVC'},
  {'constructor': svm.LinearSVC,
    'params': dict(tol=0.0001, C=1.0, intercept_scaling=1, verbose=0),
    'name': 'Linear SVC'},
  {'constructor': RandomForestClassifier,
    'params': dict(n_estimators=150, min_samples_split=2, n_jobs=-1),
    'name': 'Random Forest Classifier'}]

def getPredictions(training_rows, training_header, leaderboard_rows, leaderboard_header, k=10):
  '''Returns predictions for enntire set for all models by way of kFolds
  TODO: Improve this by k-folding
  '''
  th = toMap(training_header)
  lh = toMap(leaderboard_header)

  # Split set into train and cv
  (cv, construct) = cvSplit(training_rows, .2)
  # Separate features and targets
  train_y       = training_rows[:, th['Score1']]
  train_X       = training_rows[:, th['Score1']+1:]
  construct_y   = construct[:, th['Score1']]
  construct_X   = construct[:, th['Score1']+1:]
  cv_y          = cv[:, th['Score1']]
  cv_X          = cv[:, th['Score1']+1:]
  leaderboard_X = leaderboard_rows[:, lh['Id']+1:]

  cv_header = ['Score1'] + [model['name'] for model in MODELS]
  cph = toMap(cv_header)
  cv_m = len(cv_header)
  cv_n = len(cv_X)
  cv_preds = np.array([[0 for _ in range(cv_m)] for _ in range(cv_n)])
  cv_preds[:,0] = cv_y

  lb_header = [model['name'] for model in MODELS]
  lph = toMap(lb_header)
  lb_m = len(lb_header)
  lb_n = len(leaderboard_X)
  leaderboard_preds = np.array([[0 for _ in range(lb_m)] for _ in range(lb_n)])

  for model in MODELS:
    model_name = model['name']
    # TODO: Optimize these
    print 'Training %(model)s...' % dict(model=model_name)
    instance = model['constructor'](**model['params'])

    instance.fit(construct_X.tolist(), construct_y)
    cv_preds[:, cph[model_name]] = instance.predict(cv_X)

    instance.fit(train_X, train_y)
    leaderboard_preds[:, lph[model_name]] = instance.predict(leaderboard_X)
  return (cv_preds, cv_header, leaderboard_preds, lb_header)

def getBestModel(training_rows, header, k=10):
  '''Cross validate a bunch of models and optimize their parameters.
  Returns best performing model trained on the set given

  Keyword arguments:
  training_rows -- example rows matrix
  header        -- list header
  k         -- Number of folds
  '''
  h = toMap(header)

  folds = cross_validation.KFold(n=training_rows.shape[0], k=10)
  best = dict(qwk=0, model='', params={})
  for model in MODELS:
    print 'Optimizing %(name)s...' % model
    for construct_idx, cv_idx in folds:
      construct_X, cv_X = training_rows[construct_idx, h['Score1']+1:], training_rows[cv_idx, h['Score1']+1:]
      construct_y, cv_y = training_rows[construct_idx, h['Score1']], training_rows[cv_idx, h['Score1']]
      constructor = model['constructor']

      pk = model['params'].keys()
      pv = model['params'].values()
      pt = map(lambda v: type(v), model['params'].values())

      if len(model['params']) > 0:
        pv = minimize(lambda pv: scoreModel(constructor, pk, pv, pt, construct_X, construct_y, cv_X, cv_y), pv).x
      pv = [pt[i](v) for (i, v) in enumerate(pv)]

      params = dict(zip(pk, pv))
      instance = constructor(**params)
      instance.fit(construct_X, construct_y)
      preds = instance.predict(cv_X).astype(int)
      qwk = quadratic_weighted_kappa(cv_y, preds, 0, 3)

      if qwk > best['qwk']:
        best = dict(qwk=qwk, model=constructor, params=params)
  final = best['model'](**best['params']).fit(training_rows[:, h['Score1']+1:], training_rows[:, h['Score1']])
  return final

def scoreModel(model, pk, pv, pt, construct_X, construct_y, cv_X, cv_y):
  pv = [pt[i](v) for (i, v) in enumerate(pv)]
  params = dict(zip(pk,pv))
  instance = model(**params)
  try:
    instance.fit(construct_X, construct_y)
  except ZeroDivisionError:
    return .999
  preds = instance.predict(cv_X).astype(int)
  qwk = quadratic_weighted_kappa(cv_y, preds, 0, 3)
  return 1. - qwk

def listsToDicts(rows):
  '''Implement me!'''

def dictsToLists(rows):
  '''Implement me!'''

def kFold(rows, k):
  '''Split rows in to k folds. Returns list of lists.'''
  n = len(rows)
  folds = [[] for i in range(k)]
  for row in rows:
    i = int(round(random()*k)) - 1
    folds[i].append(row)
  return folds

def cvSplit(matrix, cvfrac):
  np.random.shuffle(matrix)
  n = int(matrix.shape[0] * cvfrac)
  cv, train = matrix[:n], matrix[n:]
  return (cv, train)
