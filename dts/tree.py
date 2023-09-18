import time

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from dts import helper_functions as hf
from dts import tree_builder


class DecisionTree:
    def __init__(
        self,
        loss="mse",
        max_depth=None,
        max_leaf_nodes=None,
        min_samples_leaf=1,
        splitter="random",
        max_features="sqrt",
        max_thresholds=1,
        min_risk_decrease=0,
        random_state=None,
    ):
        """
        A decision tree classifier.

        Parameters
        ----------
        loss : {'conservative', 'mse', 'ce', 'gce', 'ne', 'ranking', 'twoing', 'credal','sqrt-gini'}, optional
            The loss function to optimise during training.
            Equivalent to choosing the impurity measure, with 'mse'='gini', 'ce'='entropy','conservative'='missclassification'. The default is "mse".
        max_depth : int or None, optional
            Maximum depth of the tree. E.g., a decision stump has depth 1. The default is None, which imposes no restriction on depth.
        max_leaf_nodes : int or None, optional
            Maximum number of leaf nodes allowed in the tree. If not None, then tree is grown in best-first fashion. The default is None, which imposes no restriction.
        min_samples_leaf : int, optional
            minimum samples allowed at a leaf. Increasing this will lead to shallower trees and can help reduce overfitting. The default is 1.
        splitter : {'random','best'}, optional
            Method of looking for candidate splits for a node. 'random' finds the best random split, 'best' finds the optimal split. The default is 'random'.
        max_features : {'all','sqrt','log2'} or int, optional
            Maximum number of features to consider when looking for a split at a node.
            If 'all', then max_features=n_features.
            If 'sqrt' (default), then max_features=ceil(sqrt(n_features)).
            If 'log2', then max_features=ceil(log2(n_features)).
        max_thresholds : int, optional
            Number of candidate thresholds to sample for each feature when looking for splits. Only used when splitter='random'. Ignored for splitter='best'. The default is 1.
        min_risk_decrease : nonnegative float, optional
            Minimum decrease in risk/loss/impurity before a node is called a leaf. The default is 0.
        random_state : None, int or instance of RandomState
            If reproducibility not needed, leave random_state=None.
            Otherwise, set equal to an integer or a numpy.random.RandomState instance to seed pseudo-random number generation during training.
            Default is None.

        Returns
        -------
        None.

        """

        self.loss = loss
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.splitter = splitter
        self.max_thresholds = max_thresholds
        self.min_risk_decrease = min_risk_decrease
        self.random = check_random_state(random_state)

        # check arguments are valid
        hf._check_init_args(
            self.loss,
            self.max_depth,
            self.max_leaf_nodes,
            self.min_samples_leaf,
            self.max_features,
            self.max_thresholds,
            self.splitter,
            self.min_risk_decrease,
        )

        if self.max_leaf_nodes is None:
            self.max_leaf_nodes = float("inf")
        if self.max_depth is None:
            self.max_depth = float("inf")

        self.is_trained = False  # indicate if tree empty/trained

    def __str__(self):
        return f"DecisionTree(loss={self.loss}, max_depth={self.max_depth}, max_leaf_nodes={self.max_leaf_nodes}, min_samples_leaf={self.min_samples_leaf}, splitter={self.splitter}, max_features={self.max_features}, max_thresholds={self.max_thresholds}, min_risk_decrease={self.min_risk_decrease})"

    def fit(self, X, y, label_encoder=None):
        """
        Fit a decision tree classifier to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training feature matrix.
        y : array-like of shape (n_samples,)
            The target labels.
        label_encoder, optional
            A method for encoding labels to numeric data type. The default is None.

        Returns
        -------
        self
            The fitted DecisionTreeClassifier instance.

        """
        try:
            X = np.array(X, dtype=np.float32)  # convert to float32 to save memory
        except:
            raise Warning("Please pass input X that can be converted to float32 array")

        self.n_samples, self.n_features = X.shape

        try:
            y = np.array(y)
            if len(y.shape) > 1:
                raise Warning("please provide only 1 label per sample")
            if X.shape[0] != y.shape[0]:
                raise Warning("X and y have inconsistent number of samples")

        except:
            raise Warning(
                "Please specify y that can be converted to flattened numpy array."
            )
        else:
            if label_encoder is None:
                self.enc = LabelEncoder()
                self.enc.fit(y.ravel())
            else:
                self.enc = label_encoder

            y = self.enc.transform(y.ravel()).ravel().astype(np.int32)
            self.n_classes = self.enc.classes_.shape[0]
            self.classes_ = self.enc.classes_

        if (self.n_classes > 2) and (self.loss == "ranking"):
            raise Warning("ranking loss only compatible with binary classification")

        # hyperparameters
        if len(self.loss.split()) > 1:
            self.gamma = self.loss.split()[1:]
            if len(self.gamma) < 2:
                self.gamma += [-1]
            self.gamma = np.array(self.gamma, dtype=np.float64)
        else:
            self.gamma = np.array([-1, -1], dtype=np.float64)
        if len(self.gamma) != 2:
            raise Warning("gamma not intialised to size 2")

        # defaults
        if self.loss[:2] == "ne":
            self.loss = self.loss[:2]
            if self.gamma[0] == -1:
                self.gamma[0] = 0.5
        elif self.loss[:6] == "credal":
            self.loss = self.loss[:6]
            if self.gamma[0] == -1:
                self.gamma[0] = 1
        elif self.loss[:3] == "gce":
            self.loss = self.loss[:3]
            if self.gamma[0] == -1:
                self.gamma[0] = 0.7

        if self.max_features == "sqrt":
            self.max_features = int(np.ceil(np.sqrt(self.n_features)))
        elif self.max_features == "all":
            self.max_features = self.n_features
        elif self.max_features == "log2":
            self.max_features = int(np.ceil(np.log2(self.n_features)))

        # arguments to pass to the tree builder method -> note max features left out, this will be specified fully in .fit
        self.init_args = [
            self.loss,
            self.max_depth,
            self.max_leaf_nodes,
            self.min_samples_leaf,
            self.splitter,
            self.max_thresholds,
            self.min_risk_decrease,
        ]

        self.additional_args = [
            self.n_samples,
            self.n_features,
            self.max_features,
            self.n_classes,
            self.gamma,
        ]

        self.nodes, self.n_leaf_nodes, self.tree_depth = tree_builder.grow_dt(
            self.init_args,
            X,
            y,
            self.additional_args,
            random=self.random,
        )  # return n_leaf_nodes and tree_depth so this info does not have to be recomputed from nodes
        self.is_trained = True
        return self

    def predict(self, X, classifier="tree", leaf_node_predictor="props"):
        """
        Predict class labels using the trained decision tree classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predict class values for instances in X.
        classifier : {"tree","trees"}, optional
            Whether .predict is being called from a DecisionTree (tree) or DecisionForest (trees). The default is "tree".
        leaf_node_predictor : {"leaf-node-idx","props","gce","one-hot"}, optional
            Type of prediction to make from the leaf node information, only used when classifier="tree".
            Options include
                - "props": vector of the label distribution at the leaf node.
                - "leaf-node-idx": leaf node number.
                - "one-hot": one hot vector, with 1 in position of most prevalent class at the leaf node.
                - "gce": Alternate label distribution arising from use of the GCE loss.
            The default is "props".

        Returns
        -------
        numpy.ndarray
            The predicted class labels or values for the input samples.
            Shape is (n_samples,) when classifier == "tree" and (n_samples,n_classes) when classifier == "trees".

        """
        try:
            X = np.array(X, dtype=np.float32)
        except:
            raise Warning(
                "Please pass to .predict input X that can be converted to float32 array"
            )

        n_samples = X.shape[0]
        if not self.is_trained:
            print("tree not finished training!")
            if classifier == "tree":
                return np.nan * np.ones((n_samples,), dtype=np.int64)
            else:
                return np.nan * np.ones((n_samples, self.n_classes), dtype=np.int64)
        else:
            if classifier == "tree":
                preds = np.zeros((n_samples,), dtype=np.int64)
            elif classifier == "trees":
                preds = np.zeros((n_samples, self.n_classes), dtype=np.float64)

            for i in range(n_samples):
                current_node = 0
                while not self.nodes[current_node].is_leaf:
                    go_left = (
                        X[i][self.nodes[current_node].attribute]
                        <= self.nodes[current_node].threshold
                    )
                    if go_left:
                        current_node = self.nodes[current_node].child_idxs[0]
                    else:
                        current_node = self.nodes[current_node].child_idxs[1]

                if classifier == "tree":
                    if leaf_node_predictor == "leaf-node-idx":
                        preds[i] = current_node
                    else:
                        onehot_label = self.nodes[current_node].g(
                            "one-hot", self.random
                        )
                        preds[i] = self.classes_[np.argmax(onehot_label)]
                elif classifier == "trees":
                    if self.loss == "gce":
                        preds[i] = self.nodes[current_node].g(
                            leaf_node_predictor, self.random, self.gamma
                        )
                    else:
                        preds[i] = self.nodes[current_node].g(
                            leaf_node_predictor, self.random
                        )

            return preds

    def score(self, X, y, metric="acc"):
        """
        Calculate the predictive performance of the trained classifier on X and y.

        Parameters
        ----------
        X : array-like of shape (n_sampels, n_features)
            The input feature matrix.
        y : array-like of shape (n_samples,)
            The true target labels.
        metric : {"acc", "loss"}, optional
            The evaluation metric to use.
            Currently supported options are:
                - "acc": Accuracy.
                - "loss": Empirical risk based on the chosen loss function.
            The default is "acc".

        Returns
        -------
        float
            Measure of predictive performance of the trained classifier.

        """
        if metric == "acc":
            preds = self.predict(X)
            return (preds == y).mean()
        elif metric == "loss":
            loss = 0
            # compute total partial risk at leaf nodes with the new data.
            preds = self.predict(
                X, classifier="tree", leaf_node_predictor="leaf-node-idx"
            )
            leaf_nodes = np.unique(preds)
            n_leaf_nodes = leaf_nodes.shape[0]
            for i in range(n_leaf_nodes):
                idx = np.where(preds == leaf_nodes[i])[0]
                weights = np.bincount(
                    self.enc.transform(y[idx].ravel()).ravel().astype(np.int32),
                    minlength=self.n_classes,
                )
                loss += hf._partial_risk(
                    y, self.loss, self.n_classes, weights, self.gamma
                )
            return loss
        else:
            raise Warning("acc and loss only implemented score functions for now")

    def feature_importances(self):
        """
        Compute the risk reduction feature importances.

        Returns
        -------
        FI : array-like of shape (n_features,)
            Risk reduction based feature importances.

        """
        FI = np.zeros([self.n_features])
        for node in self.nodes:
            if not self.nodes[node].is_leaf:
                FI[self.nodes[node].attribute] += self.nodes[node].risk_reduction

        return FI
