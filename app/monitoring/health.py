from datetime import datetime,timezone
import time
class HealthMonitor:
    def __init__(self): self.started=time.monotonic()
    def snapshot(self,model_count):
        return {'status':'healthy','uptime_seconds':round(time.monotonic()-self.started,3),'models_loaded':model_count,'timestamp':datetime.now(timezone.utc).isoformat()}
