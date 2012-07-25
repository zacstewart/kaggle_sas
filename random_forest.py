from sklearn.ensemble import RandomForestClassifier
from ml_metrics import *
import scipy

class RfModel:
  def __init__(self):
    self.rf = RandomForestClassifier(n_estimators=150, min_samples_split=2, n_jobs=-1)
  def train(self, x, y):
    self.rf.fit(x, y)
  def predict(self, x):
    return self.rf.predict(x)
  def validate(self, x, y):
    return self.rf.score(x, y)
