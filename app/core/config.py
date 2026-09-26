from dataclasses import dataclass
import os
@dataclass(frozen=True)
class Settings:
    app_name: str=os.getenv('NEUROSENTINEL_APP_NAME','NeuroSentinel')
    environment: str=os.getenv('NEUROSENTINEL_ENV','development')
    log_level: str=os.getenv('NEUROSENTINEL_LOG_LEVEL','INFO')
    model_version: str=os.getenv('NEUROSENTINEL_MODEL_VERSION','0.2.0')
    random_seed: int=int(os.getenv('NEUROSENTINEL_SEED','42'))
    max_batch_size: int=int(os.getenv('NEUROSENTINEL_MAX_BATCH','256'))
settings=Settings()
