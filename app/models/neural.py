import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class NeuralSignalClassifier:
    def __init__(self,random_state=42):
        self.model=Pipeline([('scale',StandardScaler()),('mlp',MLPClassifier(hidden_layer_sizes=(64,32,16),max_iter=300,early_stopping=True,random_state=random_state))])
        self.fitted=False
    def fit(self,X,y):
        self.model.fit(np.asarray(X,dtype=float),np.asarray(y)); self.fitted=True; return self
    def predict(self,X):
        if not self.fitted: raise RuntimeError('model must be fitted before prediction')
        return self.model.predict(np.asarray(X,dtype=float)).tolist()
    def predict_proba(self,X):
        if not self.fitted: raise RuntimeError('model must be fitted before prediction')
        return self.model.predict_proba(np.asarray(X,dtype=float)).tolist()
