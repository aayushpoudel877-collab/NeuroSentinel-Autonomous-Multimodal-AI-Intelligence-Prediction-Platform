import numpy as np
from app.models.neural import NeuralSignalClassifier
def test_neural_classifier():
    rng=np.random.default_rng(7); X=rng.normal(size=(80,5)); y=(X[:,0]>0).astype(int); model=NeuralSignalClassifier(7).fit(X,y); assert len(model.predict(X[:5]))==5
