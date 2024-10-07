## Hyperopt (binary classification)

### As a class

```python
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, SparkTrials, space_eval
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
import numpy as np
import xgboost as xgb
import time

class XGBoostBinaryClassifierOptimizer():
	def __init__(self, X, y):
		self.X = X
		self.y = y
		self.best_hyperparameters = None
		self.space = {
			'max_depth': hp.randint('max_depth', 3, 10),
			'gamma': hp.uniform('gamma', 0, 9),
			'reg_alpha': hp.uniform('reg_alpha', 1e-2, 100),
			'reg_lambda': hp.uniform('reg_lambda', 0, 1),
			'subsample': hp.uniform('subsample', 0.5, 1),
			'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
			'min_child_weight': hp.randint('min_child_weight', 0, 10),
			'learning_rate': hp.choice('learning_rate', [0.1]),
			'n_estimators': hp.choice('n_estimators', [140]),
			'scale_pos_weight': hp.choice('scale_pos_weight', [pos_weight]),
			'objective': hp.choice('objective', ['binary:logistic']),
			'seed': hp.choice('seed', [42]),
		}

	def objective(self, space):
		cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
		clf = xgb.XGBClassifier(**space)
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
opt = XGBoostBinaryClassifierOptimizer(X_train, y_train)
best_params = opt.find_best_hyperparameters()

clf = xgb.XGBClassifier(**best)
clf.fit(X_train, y_train)
```


### As procedural code

```python
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, SparkTrials, space_eval
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
import numpy as np
import xgboost as xgb
import time

pos_weight = (len(y)-y.sum())/y.sum()   # y = target array for binary clf

parameter_space = {
	'max_depth': hp.randint('max_depth', 3, 10),
	'gamma': hp.uniform('gamma', 0, 9),
	'reg_alpha': hp.uniform('reg_alpha', 1e-2, 100),
	'reg_lambda': hp.uniform('reg_lambda', 0, 1),
	'subsample': hp.uniform('subsample', 0.5, 1),
	'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
	'min_child_weight': hp.randint('min_child_weight', 0, 10),
	'learning_rate': hp.choice('learning_rate', [0.1]),
	'n_estimators': hp.choice('n_estimators', [140]),
	'scale_pos_weight': hp.choice('scale_pos_weight', [pos_weight]),
	'objective': hp.choice('objective', ['binary:logistic']),
	'seed': hp.choice('seed', [42]),
}

def objective(space):
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
	clf = xgb.XGBClassifier(**space)
	best_score = cross_val_score(clf, X, y, scoring='balanced_accuracy', cv=cv).mean()
	print('SCORE:', best_score)
	return {'loss': -best_score, 'status':STATUS_OK}


start = time.time()

trials = SparkTrials()   # or Trials() if no spark

best_hyperparameters = fmin(
	fn = objective,
	space = parameter_space,
	max_evals = 100,
	algo = tpe.suggest,
	trials = trials
)
best_hyperparameters = space_eval(parameter_space, best_hyperparameters)

print('\nHyperparameter optimization took %s minutes' % int(np.round((time.time() - start)/60, 0)))
print('The best hyperparameters are:')
for param_name, param_val in best_hyperparameters.items():
	print('-', param_name, ':', param_val)
print('\nBest loss:', trials.best_trial['result']['loss'])


clf = xgb.XGBClassifier(**best_hyperparameters)
clf.fit(X,y)
```

If there, for some reason, is no time for hyperparameter optimization, the following default parameters can be tried for a decent-ish result (but YMMV):

```python
import xgboost as xgb

clf = xgb.XGBClassifier(
	learning_rate = 0.1,
	n_estimators = 140,
	subsample = 0.8,
	max_depth = 5,
	objective = 'binary:logistic',
	n_thread = 4,
	random_state = 42,
	seed = 27
)
```

