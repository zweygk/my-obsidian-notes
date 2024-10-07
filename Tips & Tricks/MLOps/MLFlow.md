To ensure proper logging and model loading you need to log the following:

- Signature
- Model
- Pip requirements

If your model is even slightly more complicated than a single sklearn pipe, you need to wrap your model in a class that inherits from mlflow.pyfunc.PythonModel and has a predict function.

Example:

```python
import mlflow

class MyPipeline(mlflow.pyfunc.PythonModel):
	def __init__(self):
		'... do something'

	def fit(self, X, y=None):
		signature = mlflow.models.infer_signature(X)
		'... fit your model'
		# After fitting
		with mlflow.start_run():
			mlflow.pyfunc.log_model(
				"MyPipeline",
				python_model=self,
				signature=signature,
				pip_requirements=["pandas==1.5.3"]
			)

	def predict(self, context, X, params=None):
		'... do something'
```