import numpy as np
from app.models.neural import NeuralSignalClassifier
from app.services.evaluation import classification_report_binary

def make_dataset(n=600,seed=42):
    rng=np.random.default_rng(seed); X=rng.normal(size=(n,8)); y=(X[:,0]+0.8*X[:,1]-0.4*X[:,2]+rng.normal(0,0.35,n)>0).astype(int); return X,y

def train(seed=42):
    X,y=make_dataset(seed=seed); split=int(len(X)*0.8); model=NeuralSignalClassifier(seed).fit(X[:split],y[:split]); pred=model.predict(X[split:]); return model,classification_report_binary(y[split:],pred)

if __name__=='__main__':
    _,metrics=train(); print(metrics)
