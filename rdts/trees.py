import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state, resample

from rdts.tree import DecisionTree


class DecisionForest:
    def __init__(
        self,
        n_estimators=10,
        loss="mse",
        max_depth=None,
        max_leaf_nodes=None,
        min_samples_leaf=1,
        splitter="random",
        max_features="sqrt",
        max_thresholds=1,
        min_risk_decrease=0,
        bootstrap=False,
        n_jobs=1,
        random_state=None,
    ):
        """
        An ensemble of decision trees.
        Individual trees are trained independently.
        Predictions are made by majority vote of average class distribution over all trees.

        Parameters
        ----------
        n_estimators : int, optional
            Number of decision trees to include in the ensemble. The default is 10.
        loss : str, optional
            The loss function to be optimized during tree training. The default is "mse", equivalent to gini impurity.
        max_depth : int or None, optional
            The maximum depth of each tree. The default is None.
        max_leaf_nodes : int or None, optional
            The maximum number of leaf nodes in each tree. The default is None.
        min_samples_leaf : int, optional
            The minimum number of samples required to be at a leaf node. The default is 1.
        splitter : {"best","random"}, optional
            The strategy used to choose the split at each node. The default is "random".
        max_features : str, optional
            The number of features to consider when looking for the best split. The default is "sqrt".
        max_thresholds : int, optional
            The maximum number of thresholds to evaluate for each feature when looking for best random split. Ignored when splitter=="best". The default is 1.
        min_risk_decrease : float, optional
            The minimum decrease in empirical risk required to make a split. The default is 0.
        bootstrap : bool, optional
            Whether to use bootstrap samples when training each tree. The default is False.
        n_jobs : int, optional
            Number of trees to train/test in parallel. Capped at the number of processors of the device. The default is 1.
        random_state : None, int or instance of RandomState
            If reproducibility not needed, leave random_state=None.
            Otherwise, set equal to an integer or a numpy.random.RandomState instance to seed pseudo-random number generation during training.
            Default is None.


        Returns
        -------
        None.

        """

        self.n_estimators = n_estimators
        self.loss = loss
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.splitter = splitter
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.min_risk_decrease = min_risk_decrease
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random = check_random_state(random_state)

        self.is_trained = False  # indicate if tree empty/trained
        self.max_seed = 2**23 - 1

    def train_tree(self, X, y, label_encoder, random_state):
        """
        Train a single decision tree on the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Array of training labels.
        label_encoder : sklearn.preprocessing.LabelEncoder
            Fitted label encoder for mapping class labels to integers.

        Returns
        -------
        g : dts.tree.DecisionTree
            The trained decision tree.
        """

        g = DecisionTree(
            loss=self.loss,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            splitter=self.splitter,
            max_features=self.max_features,
            max_thresholds=self.max_thresholds,
            min_risk_decrease=self.min_risk_decrease,
            random_state=random_state,
        )
        if self.bootstrap:
            # X_tr, y_tr = resample(X, y)
            X_tr, y_tr = resample(X, y, random_state=random_state)
            g.fit(X_tr, y_tr, label_encoder)
        else:
            g.fit(X, y, label_encoder)
        return g

    def fit(self, X, y):
        """
        Fit the decision forest on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : DecisionForest
            The fitted decision forest.

        """

        self.enc = LabelEncoder().fit(y.ravel())

        self.n_samples, self.n_features = X.shape
        self.gs = Parallel(
            n_jobs=min(self.n_jobs, self.n_estimators), prefer="threads"
        )(
            delayed(self.train_tree)(X, y, self.enc, self.random.randint(self.max_seed))
            for i in range(self.n_estimators)
        )

        self.is_trained = True
        self.classes = self.gs[0].classes_
        return self

    def predict_tree(self, g, X, lnp):
        """
        Predict class labels using a single tree in the ensemble.

        Parameters
        ----------
        g : dts.tree.DecisionTree
            The decision tree to use for prediction.
        X : array-like of shape (n_samples, n_features)
            Testing input samples.
        lnp : str
            The leaf node predictor strategy to use.

        Returns
        -------
        predictions : array-like
            Predicted class labels for the input samples.

        """

        return g.predict(X, classifier="trees", leaf_node_predictor=lnp)

    def predict(self, X, leaf_node_predictor="props"):
        """
        Predict class labels using the ensemble of trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Testing input samples.
        leaf_node_predictor : str, optional
            The leaf node predictor strategy to use. The default is "props".

        Returns
        -------
        predictions : numpy.ndarray of shape (n_samples,)
            Predicted class labels for the input samples.

        """

        if self.is_trained:
            self.preds = Parallel(
                n_jobs=min(self.n_jobs, self.n_estimators), prefer="threads"
            )(delayed(self.predict_tree)(g, X, leaf_node_predictor) for g in self.gs)
            probs = np.mean(self.preds, axis=0)
            return self.classes[np.argmax(probs, axis=1)]
        else:
            raise Warning("Not finished training yet.")

    def score(self, X, y, metric="acc", leaf_node_predictor="props"):
        """
        Calculate the performance score of the ensemble on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Testing input samples.
        y : array-like of shape (n_samples,)
            Array of target values.
        metric : str, optional
            The performance metric to use. Currently, only "acc" (accuracy) is supported. The default is "acc".
        leaf_node_predictor : str, optional
            The leaf node predictor strategy to use. The default is "props".

        Returns
        -------
        score : float
            The calculated performance score.
        """

        preds = self.predict(X, leaf_node_predictor=leaf_node_predictor)
        if metric == "acc":
            return (preds == y).mean()
        else:
            raise Warning("acc only implemented score function for now")

    def n_leaves(self, tree):
        """
        Get the number of leaf nodes in a specific tree of the ensemble.

        Parameters
        ----------
        tree : int
            Index of the tree in the ensemble.

        Returns
        -------
        n_leaves : int
            Number of leaf nodes in the specified tree.

        """

        return self.gs[tree].n_leaf_nodes

    def get_depth(self, tree):
        """
        Get the depth of a specific tree in the ensemble.

        Parameters
        ----------
        tree : int
            Index of the tree in the ensemble.

        Returns
        -------
        depth : int
            Depth of the specified tree.

        """

        return self.gs[tree].tree_depth

    def get_max_depth(self):
        """
        Get the maximum depth among all trees in the ensemble.

        Returns
        -------
        max_depth : int
            Maximum depth among all trees in the ensemble.

        """

        depths = []
        for tree in self.gs:
            depths += [tree.tree_depth]
        return np.max(depths)

    def feature_importances(self):
        """
        Calculate the importances of each feature in the training data using the trained ensemble.
        Feature importance measured by the total reduction in risk due to that feature, averaged over all trees in the ensemble.

        Returns
        -------
        importances : array-like of shape (n_features,)
            An array of feature importances, one value for each feature.

        """

        importances = np.zeros([self.gs[0].n_features])
        for tree in self.gs:
            importances += tree.feature_importances() / self.n_estimators

        return importances
