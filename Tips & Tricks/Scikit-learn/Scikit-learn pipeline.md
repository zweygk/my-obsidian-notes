
Normal scikit-learn pipeline boilerplate:

```python
from sklearn.pipeline import Pipeline

my_pipeline = Pipeline(
   steps = [
	   ('step1', transformer1()),
	   ('step2', transformer2()),
	   ('step3', my_classifier())
   ]
)
```

If you are using imbalanced-learn transformers or models you need to use that module's own pipeline object. The syntax stays the same though. Example:

```python
from imblearn.pipeline import Pipeline

my_pipeline = Pipeline(
   steps = [
	   ('step1', transformer1()),
	   ('step2', transformer2()),
	   ('step3', my_classifier())
   ]
)
```