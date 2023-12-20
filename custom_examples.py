import numpy as np
from rdts.tree import DecisionTree
from rdts.trees import DecisionForest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

rs = 1
X, y = make_classification(n_samples=1000, random_state=rs)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=rs
)

print("Decision Tree")
# negative exponential loss, lambda=1/pi
g = DecisionTree(
    loss=f"ne {1/np.pi}",
    max_depth=None,
    max_leaf_nodes=None,
    min_samples_leaf=1,
    splitter="best",
    max_features="all",
    min_risk_decrease=0.0,
    random_state=rs,
)
g.fit(X_train, y_train)
print("Test accuracy NE 1/pi:", g.score(X_test, y_test))

# generalized cross entropy loss, q=1/e
g = DecisionTree(
    loss=f"gce {np.exp(-1)}",
    max_depth=None,
    max_leaf_nodes=None,
    min_samples_leaf=1,
    splitter="best",
    max_features="all",
    min_risk_decrease=0.0,
    random_state=rs,
)
g.fit(X_train, y_train)
print("Test accuracy GCE 1/e:", g.score(X_test, y_test))

# some other loss functions/splitting criteria
loss_functions = [
    "conservative",  # misclassification impurity
    "ne",  # negative exponential loss lambda=0.5
    "ne 0.1",  # negative exponential loss lambda=0.1
    "ne 0.9",  # negative exponential loss lambda=0.9
    "sqrt-gini",  # sqrt of Gini impurity
    "mse",  # Gini impurity
    "ce",  # entropy impurity
    "gce",  # generalized cross entropy impurity with q=0.7
    "gce 0.1",  # generalized cross entropy impurity with q=0.1
    "gce 0.9",  # generalized cross entropy impurity with q=0.9
    "ranking",  # ranking loss
    "twoing",  # twoing split criteria
    "credal",  # Credal-C4.5 with s=1
]
for lf in loss_functions:
    g = DecisionTree(
        loss=lf,
        max_depth=None,
        max_leaf_nodes=None,
        min_samples_leaf=1,
        splitter="random",
        max_features="sqrt",
        max_thresholds=1,
        min_risk_decrease=0.0,
        random_state=rs,
    )
    g.fit(X_train, y_train)
    print(f"Test accuracy {lf} loss:", g.score(X_test, y_test))

print("\nRandom Forest")
# conservative loss/misclassification impurity
g = DecisionForest(
    n_estimators=10,
    loss="conservative",
    max_depth=None,
    max_leaf_nodes=None,
    min_samples_leaf=1,
    splitter="best",
    max_features="sqrt",
    min_risk_decrease=0.0,
    bootstrap=True,
    random_state=rs,
)
g.fit(X_train, y_train)
print("Test accuracy conservative loss:", g.score(X_test, y_test))

print("\nExtremely Randomized Trees")
# cross entropy loss
g = DecisionForest(
    n_estimators=10,
    loss="ce",
    max_depth=None,
    max_leaf_nodes=None,
    min_samples_leaf=1,
    splitter="random",
    max_features="sqrt",
    max_thresholds=1,
    min_risk_decrease=0.0,
    bootstrap=False,
    random_state=rs,
)
g.fit(X_train, y_train)
print("Test accuracy CE loss:", g.score(X_test, y_test))

print("Number of leaf nodes in tree 5:", g.n_leaves(5))
print("Depth of tree 5:", g.get_depth(5))
print("Maximum tree depth:", g.get_max_depth())
print("Feature importances:", g.feature_importances())
