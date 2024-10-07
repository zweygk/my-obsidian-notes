## Hyperopt (binary classification)

### As a class

```python
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, SparkTrials, space_eval
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import time

class LogisticRegressionOptimizer():
	def __init__(self, X, y):
		self.X = X
		self.y = y
		self.best_hyperparameters = None
		self.space = {
			'C': hp.lognormal('LR_C', 0, 1.0),
			'solver': hp.choice('solver', ['liblinear', 'lbfgs']),
			'class_weight': hp.choice('class_weight', ['balanced'])
		}

	def objective(self, space):
		cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
		clf = LogisticRegression(**space)
		best_score = cross_val_score(clf, self.X, self.y, scoring='balanced_accuracy', cv=cv).mean()
		print('SCORE:', best_score)
		return {'loss': -best_score, 'status': STATUS_OK}

	def find_best_hyperparameters(self):
		start = time.time()
		trials = SparkTrials()   # or Trials()
		best_hyperparameters = fmin(
			fn = self.objective,
			space = self.space,
			max_evals = 200,
			algo = tpe.suggest,
			trials = trials
		)
		print('\nHyperparameter optimization took %s minutes' % int(np.round((time.time() - start)/60, 0)))
		print('\nBest loss:', trials.best_trial['result']['loss'])
		self.best_hyperparameters = space_eval(self.space, best_hyperparameters)
		self.X = None
		self.y = None
		return self.best_hyperparameters
```

Usage:

```python
opt = LogisticRegressionOptimizer(X_train, y_train)
best_params = opt.find_best_hyperparameters()

clf = LogisticRegression(**best)
clf.fit(X_train, y_train)
```