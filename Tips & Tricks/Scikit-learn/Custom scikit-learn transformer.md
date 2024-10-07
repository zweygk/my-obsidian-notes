
Custom transformers must inherit from the BaseEstimator and TransformerMixin classes. Furthermore, it should contain

- A fit method that returns the object itself
- A transform method that returns a transformed input (X)
- (Optional) An init method if you need to pass variables to fit or transform
- (Optional) A fit_transform method if you need to override the default one that combines fit and transform

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, my_var):
		self.my_var = my_var
	def fit(self, X, y=None):
		# do something if necessary
		# It is necessary to return self in fit method
		return self
	def transform(self, X, y=None):
		# apply transform to X
		X_transformed = X * self.my_var   # example
		return X_transformed
	def fit_transform(self, X, y=None):
		# Only include this method if absolutely necessary
		return X

transformer = MyTransformer(my_var=100)
```
