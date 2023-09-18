# node class
import numpy as np


class Node:
    def __init__(
        self,
        depth=None,
        is_leaf=None,
        partial_risk=None,
        risk_reduction=None,
        parent_idx=None,
        child_idxs=None,
        relation_from_parent=None,
        attribute=None,
        threshold=None,
        varying_attributes=None,
        idx=None,
        n_local_samples=None,
        weights=None,
    ):
        """
        Node class representing a node in a decision tree.

        Parameters
        ----------
        depth : int, optional
            Depth of the node in the tree. The default is None.
        is_leaf : bool, optional
            Indicates whether the node is a leaf node. The default is None.
        partial_risk : float, optional
            The partial empirical risk associated with the data at the node. The default is None.
        risk_reduction : float, optional
            The reduction in risk achieved by splitting the node. The default is None.
        parent_idx : int, optional
            Index of the parent node. The default is None.
        child_idxs : list of int, optional
            Indices of the child nodes. The default is None.
        relation_from_parent : str, optional
            The relationship of the node with its parent ("left" or "right"). The default is None.
        attribute : int, optional
            Index of the attribute used for splitting the node. The default is None.
        threshold : float, optional
            The threshold value for splitting the node. The default is None.
        varying_attributes : list of int, optional
            Indices of attributes that have varying values within the node. The default is None.
        idx : int, optional
            Index of the node. The default is None.
        n_local_samples : int, optional
            Number of samples at the node. The default is None.
        weights : numpy.ndarray of shape (n_classes,), optional
            Number of samples in each class at the node. The default is None.

        """

        self.depth = depth
        self.is_leaf = is_leaf
        self.partial_risk = partial_risk
        self.risk_reduction = risk_reduction
        self.attribute = attribute
        self.threshold = threshold
        self.varying_attributes = varying_attributes
        self.idx = idx
        self.n_local_samples = n_local_samples
        self.weights = weights

        # info about node itself and tree structure
        self.parent_idx = parent_idx
        self.child_idxs = child_idxs
        self.relation_from_parent = relation_from_parent  # {'left','right'}

    def g(self, classifier, random, gamma=np.array([0.7, -1], dtype=np.float64)):
        """
        Calculate the g vector used for prediction based on the specified classifier.

        Parameters
        ----------
        classifier : str
            The classifier type. Supported options are "gce", "one-hot", and "props".
        random : numpy.random.RandomState
            The random state object for controlling pseudo-random number generation.
        gamma : numpy.ndarray, optional
            Hyperparameters for the loss function. The default is np.array([0.7, -1], dtype=np.float64) used for GCE.

        Returns
        -------
        g : numpy.ndarray
            Optimal constant prediction for the labels at the node.

        Raises
        ------
        Warning
            If an unsupported classifier is specified.

        """

        if classifier == "gce":
            num = np.float64(self.weights / self.n_local_samples) ** (
                1 / (1 - gamma[0])
            )
            denom = num.sum()
            return num / denom

        elif classifier == "one-hot":
            temp = np.zeros((self.weights.shape[0],), dtype=np.float64)
            largest_weight = np.max(self.weights)
            if np.sum(self.weights == largest_weight) > 1:
                # choose uniformly from most likely classes
                argmax = random.choice(np.where(self.weights == largest_weight)[0])
                temp[argmax] = 1
            else:
                temp[np.argmax(self.weights)] = 1
            return temp

        elif classifier == "props":
            return np.float64(self.weights / self.n_local_samples)

        else:
            raise Warning('classifier should be one of {"gce", "one-hot", "props"}')
