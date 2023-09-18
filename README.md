# Robust Loss Functions for Training Decision Trees with Noisy Labels

---Reference or link to paper---

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


<!---
### An example experiment from the paper

``python train_model.py --classifier DT --labels multiclass --dataset MNIST --loss_function mse --noise_rate "uniform 0.1" --replications 5 --seed 0``

Note: dataset csv files too large (>50MB) to submit through CMT, however all datasets are publicly available, with notes in the main paper on how preprocessing was done.
--->

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

See ``custom_example.py`` for more examples of ways to use the implemented tree methods.


<!---
## Reproduce experiments
Clone repository

download and process data according to the paper

run experiments 
--->
