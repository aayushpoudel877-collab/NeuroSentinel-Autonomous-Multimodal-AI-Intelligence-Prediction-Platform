from app.services.evaluation_v2 import evaluate_classification,evaluate_regression

def test_classification_report():
    report=evaluate_classification([0,1,1],[0,1,0])
    assert report.metrics['accuracy']==2/3
    assert report.confusion_matrix is not None

def test_regression_report():
    report=evaluate_regression([1,2,3],[1,3,2])
    assert report.metrics['mae'] > 0
