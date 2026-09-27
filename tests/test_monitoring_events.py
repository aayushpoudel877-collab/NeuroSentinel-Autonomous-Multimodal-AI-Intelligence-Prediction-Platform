from app.monitoring.events import MonitoringEventStore

def test_monitoring_events_are_persisted(tmp_path):
    store=MonitoringEventStore(tmp_path/"events.jsonl")
    store.append("drift","demo",{"score":0.4},"evt-1")
    assert store.recent(10)[0]["event_id"]=="evt-1"
