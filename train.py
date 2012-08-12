from progressbar import *

def tick(xk):
  print xk

def validate(constructor, params, folds):
  cv_scores = []
  qwks = []
  pwidgets =  ['Fold ', Counter(), ' ', ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]
  foldbar = ProgressBar(widgets=pwidgets, maxval=len(folds))
  foldbar.start()
  model = constructor(params)
  for i, j in enumerate(range(len(folds))):
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
    qwks.append(qwk)
    cv_scores.append(score)
  foldbar.finish()

  cv_mean = sum(cv_scores) / len(cv_scores)
  print '  - Mean score for %(model)s: %(score)f' % {'model': 'Blah', 'score': cv_mean}
  qwk_mean = sum(qwks) / len(qwks)
  print '  - Mean QWK for %(model)s: %(score)f' % {'model': 'Blah', 'score': qwk_mean}
