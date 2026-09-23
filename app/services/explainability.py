def explain_prediction(task: str, prediction: dict):
    return {
        "task": task,
        "method": "model-output attribution placeholder",
        "explanation": "This interface is designed for SHAP/LIME or gradient-based attribution in production models.",
        "prediction": prediction
    }
