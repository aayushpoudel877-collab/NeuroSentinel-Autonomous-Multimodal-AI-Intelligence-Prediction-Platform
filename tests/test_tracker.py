from app.experiments.tracker import ExperimentTracker

def test_experiment_tracker_round_trip(tmp_path):
    tracker=ExperimentTracker(tmp_path); tracker.log('run-001','demo',{'lr':0.001},{'accuracy':0.9}); result=tracker.load('run-001'); assert result['model']=='demo'; assert result['metrics']['accuracy']==0.9
