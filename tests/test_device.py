from app.training.device import resolve_device

def test_device_auto_is_valid(): assert resolve_device('auto') in {'cpu','cuda'}
