# Robust Loss Functions for Training Decision Trees with Noisy Labels

Official implementation for "Robust Loss Functions for Training Decision Trees with Noisy Labels". 

Accepted into AAAI Conference on Artificial Intelligence 2024.

Paper: https://arxiv.org/abs/2312.12937

## Overview

A python implementation of tree methods for classification.
Supported models are decision tree, random forest and extra trees.




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



## Usage:
A minimal working example:
```
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from rdts.trees import DecisionForest

X, y = make_classification()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
g = DecisionForest()
g.fit(X_train, y_train)
print("Test accuracy:", g.score(X_test, y_test))
```

See ``custom_examples.py`` for more examples of ways to use the implemented tree methods.
