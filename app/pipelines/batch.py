from app.data.validation import validate_series
from app.features.time_series import rolling_statistics
class BatchPipeline:
    def process_series(self,batches):
        results=[]
        for values in batches:
            check=validate_series(values)
            if not check.valid: results.append({'valid':False,'errors':check.errors}); continue
            results.append({'valid':True,'features':rolling_statistics(values),'warnings':check.warnings})
        return results
