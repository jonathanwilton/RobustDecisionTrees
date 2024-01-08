# Robust Loss Functions for Training Decision Trees with Noisy Labels

Official implementation for "Robust Loss Functions for Training Decision Trees with Noisy Labels". 

Accepted into AAAI Conference on Artificial Intelligence 2024.

Paper: https://arxiv.org/abs/2312.12937

## Overview

A python implementation of tree methods for classification.
Supported models include decision tree, random forest and extra-trees.




Supports training with several loss functions and splitting criteria:
- Negative Exponential Loss
- Conservative Loss (misclassification impurity)
- Mean Squared Error (Gini impurity)
- Cross Entropy Loss (entropy impurity)
- Generalized Cross Entropy Loss
- Twoing split criteria
- Imprecise information gain
- Ranking Loss (pairwise gain)


## Installation
To pip install this package locally, run:
```
pip install git+https://github.com/jonathanwilton/RobustDecisionTrees.git@main
```



## Examples
A minimal working example with decision tree classifier:

```python
from sklearn.datasets import make_classification
from rdts.tree import DecisionTree

X,y = make_classification()
g = DecisionTree()
g.fit(X,y)
print("Accuracy:", g.score(X,y))
```

A random forest with 10 trees, 
negative exponential loss ($\lambda=1/\pi$), 
no restriction on tree depth or number of leaf nodes, 
and a random seed for reproducibility. 

```python
import matplotlib.pyplot as plt
import numpy as np
from rdts.trees import DecisionForest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=0
)

g = DecisionForest(
    n_estimators=10,
    loss=f"ne {1/np.pi}",
    max_depth=None,
    max_leaf_nodes=None,
    min_samples_leaf=1,
    splitter="best",
    max_features="sqrt",
    min_risk_decrease=0.0,
    bootstrap=True,
    random_state=0,
)
g.fit(X_train, y_train)
preds = g.predict(X_test)
print("Test accuracy:", (preds == y_test).mean())

print("Number of leaf nodes in tree 5:", g.n_leaves(5))
print("Depth of tree 5:", g.get_depth(5))
print("Maximum tree depth:", g.get_max_depth())

importances = g.feature_importances()

plt.figure()
plt.bar(np.arange(X.shape[1]), importances)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()
```

See ``custom_examples.py`` for more examples of ways to use the implemented tree methods.
