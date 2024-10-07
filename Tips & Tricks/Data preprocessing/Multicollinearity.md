
(Multi)collinearity is a situation where the predictors in a regression model are linearly dependent. There are multiple ways to deal with this problem, such as PCA or VIF-based elimination. 

==If you don't deal with this problem, you might end up selecting redundant predictors for your model.==

While PCA is more simple to include in your code, VIF-based elimination can be especially useful in a situation where the model behavior needs to be easily explainable as it doesn't transform the features. But with VIF you need to decide on a threshold:

- 5 is a good threshold value when almost no collinearity is acceptable
- 10 is a good threshold value if you can afford to be more "liberal"

Below is a VIF-based multicollinearity detector and remover implemented as a [[Custom scikit-learn transformer]].

```python
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

class RedundancyDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
        self.columns_to_keep = None

	def fit(self, X, y=None):
		def remove_redundant_feature(X, threshold):
			vif = pd.DataFrame()
			vif['features'] = X.columns
			vif['vif'] = [ variance_inflation_factor(X.values, i) for i in range(X.shape[1]) ]
			vif_over_threshold = vif[vif['vif'] > threshold].sort_values(by='vif', ascending=False)
			if vif_over_threshold.shape[0] == 0:
				return X
			col_to_be_removed = vif_over_threshold['features'].values[0]
			vif_to_be_removed = vif_over_threshold['vif'].values[0]
			print(f'Removing {col_to_be_removed}. VIF = {vif_to_be_removed}')
			result = remove_redundant_feature(X.drop(columns=[col_to_be_removed], axis=1), threshold)
			return result
		X_ = remove_redundant_feature(X, self.threshold)
		self.columns_to_keep = X_.columns.values
		return self

	def transform(self, X, y=None):
		return X[list(self.columns_to_keep)]
```

Usage

```python
rd = RedundancyDropper(threshold=10)
X_ = rd.fit_transform(X)
```
