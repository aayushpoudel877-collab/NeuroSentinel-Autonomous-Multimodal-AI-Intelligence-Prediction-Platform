import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
def regression_report(y_true,y_pred):
    true=np.asarray(y_true,dtype=float); pred=np.asarray(y_pred,dtype=float)
    return {'mae':float(mean_absolute_error(true,pred)),'rmse':float(np.sqrt(mean_squared_error(true,pred))),'samples':int(len(true))}
def classification_report_binary(y_true,y_pred):
    true=np.asarray(y_true); pred=np.asarray(y_pred)
    return {'accuracy':float((true==pred).mean()),'samples':int(len(true))}
