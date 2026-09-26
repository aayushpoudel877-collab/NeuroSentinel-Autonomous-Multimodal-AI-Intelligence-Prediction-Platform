from dataclasses import dataclass
import numpy as np
@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
def validate_series(values,minimum=8):
    errors=[]; warnings=[]; arr=np.asarray(values,dtype=float)
    if len(values)<minimum: errors.append(f'at least {minimum} observations are required')
    if not np.isfinite(arr).all(): errors.append('series contains non-finite values')
    if len(arr) and np.std(arr)==0: warnings.append('series has zero variance')
    return ValidationResult(not errors,errors,warnings)
