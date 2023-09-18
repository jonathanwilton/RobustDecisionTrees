# helper functions to keep main module clean
import numpy as np
from numba import njit


def _check_init_args(
    loss_,
    max_depth_,
    max_leaf_nodes_,
    min_samples_leaf_,
    max_features_,
    max_thresholds_,
    splitter_,
    min_risk_decrease_,
):
    """
    A function to check input arguments are valid.
    """
    # choose one of the implemented loss functions/splitting criteria
    if loss_.split()[:1][0] in [
        "conservative",
        "mse",
        "ce",
        "gce",
        "ne",
        "ranking",
        "twoing",
        "credal",
        "sqrt-gini",
    ]:
        pass
    else:
        raise Warning(
            "Loss function should be one of {'conservative', 'mse', 'ce', 'gce', 'ne', 'ranking', 'twoing', 'credal','sqrt-gini'}"
        )

    # max depth should be int or None
    if max_depth_ is None:
        max_depth_ = np.inf
    elif isinstance(max_depth_, int):
        pass
    elif max_depth_ % 1 == 0:
        max_depth_ = int(max_depth_)
    else:
        raise Warning("max_depth should be an int or None")

    # max leaf nodes should be int or None
    if max_leaf_nodes_ is None:
        max_leaf_nodes_ = np.inf
    elif isinstance(max_leaf_nodes_, int):
        pass
    elif max_leaf_nodes_ % 1 == 0:
        max_leaf_nodes_ = int(max_leaf_nodes_)
    else:
        raise Warning("max_leaf_nodes should be an int or None")

    # min number samples should be int
    if isinstance(min_samples_leaf_, int):
        pass
    elif min_samples_leaf_ % 1 == 0:
        min_samples_leaf_ = int(min_samples_leaf_)
    else:
        raise Warning("min_samples_leaf should be an int or can be converted to int")

    # max_features should be one of {'all', 'sqrt', 'log2'} or int
    if max_features_ in ["all", "sqrt", "log2"]:
        pass
    elif isinstance(max_features_, int):
        pass
    elif max_features_ % 1 == 0:
        max_features_ = int(max_features_)
    else:
        raise Warning("max_featutes should be one of {'all', 'sqrt', 'log2'}")

    # max_thresholds int or None, only used for random splits
    if max_thresholds_ is None:
        pass
    elif isinstance(max_thresholds_, int):
        pass
    elif max_thresholds_ % 1 == 0:
        max_thresholds_ = int(max_thresholds_)
    else:
        raise Warning("max_thresholds should be integer type or None")

    # only two options for splitter argument
    if splitter_ == "best":
        pass
    elif splitter_ == "random":
        if max_thresholds_ is None:
            raise Warning(
                "max_thresholds cannot be None if splitter is random. If unsure, perhaps try max_thresholds = 1 for efficiency"
            )
    else:
        raise Warning("splitter must be one of {'best', 'random'}")

    # min_risk_decrease must be non-neg float
    try:
        min_risk_decrease_ = float(min_risk_decrease_)
    except:
        raise Warning("min_risk_decrease should be numeric type")
    else:
        if min_risk_decrease_ < 0:
            print(
                "Warning: Perhaps it would make more sense if min_risk_decrease >= 0?"
            )
            print("Continuing ...")


@njit
def _splitting_function(X, idx, attribute, threshold):
    """
    Applies a splitting function to the data at a node.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix containing np.float32 values.
    idx : array-like of shape (n_local_samples,)
        Array indicating which samples are at the current node.
    attribute : int
        The feature number to split on.
    threshold : float
        Threshold to split the attribute at.

    Returns
    -------
    left_child_idx, right_child_idx : tuple of two arrays
        Partitioned indices into two arrays (left_child_idx, right_child_idx).

    """
    n_local_samples = idx.shape[0]
    left_child_idx = np.zeros((n_local_samples,), dtype=np.int32)
    right_child_idx = np.zeros((n_local_samples,), dtype=np.int32)
    (kl, kr) = (0, 0)
    for i in range(n_local_samples):
        go_left = X[idx[i], attribute] <= threshold
        if go_left:
            left_child_idx[kl] = idx[i]
            kl += 1
        else:
            right_child_idx[kr] = idx[i]
            kr += 1
    return left_child_idx[:kl], right_child_idx[:kr]


@njit
def _varying_attributes(X, rows, n_local_samples, cols):
    """
    Find the features of X that have non-zero variance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix containing np.float32 values.
    rows : array-like of shape (n_local_samples,)
        Array indicating which samples are at the current node.
    n_local_samples : int
        Number of local samples at the current node.
    cols : array-like of shape (n_columns,)
        Collection of features to be checked for variability, typically features that have variability at the parent node.

    Returns
    -------
    varying_cols : 1D array
        Array of columns numbers with non-zero variability.

    """
    n_cols = cols.shape[0]

    if n_local_samples == 0:
        print("No samples at node, no attributes have any variability")
        return np.zeros((0,), dtype=np.int32)
    elif n_cols == 0:
        return np.zeros((0,), dtype=np.int32)

    out = np.zeros((n_cols,), dtype=np.int32)
    k = 0
    for j in range(n_cols):
        first = X[rows[0], cols[j]]
        for i in range(n_local_samples):
            if X[rows[i], cols[j]] != first:
                out[k] = cols[j]
                k += 1
                break

    return out[:k]


# def _regional_prediction_function(n_classes, weights, n_local_samples) -> tuple:
#     """
#     ***OBSOLETE***
#     Return the most prevalent class for computing the label at a leaf node.

#     Parameters
#     ----------
#     n_classes : int
#         Number of unique classes in the classification problem.
#     weights : array-like of shape (n_classes,)
#         Array containing the weights or counts associated with each class.
#     n_local_samples : int
#         Number of local samples at the current node.

#     Returns
#     -------
#     predicted_label : int
#         The most prevalent class label at the current node. It ranges from 0 to (n_classes - 1).
#     is_random : boolean
#         Indicates whether a random sample has been returned instead of the most prevalent class.

#     """
#     if n_local_samples <= 0:
#         print("There are no samples at the current node, will assign random label here")
#         return np.random.randint(n_classes), True

#     largest_weight = np.max(weights)

#     # check if there is a majority class
#     if np.sum(weights == largest_weight) > 1:
#         # choose uniformly from most likely classes
#         out = np.random.choice(np.where(weights == largest_weight)[0])
#         return out, False
#     else:
#         return np.argmax(weights), False


def _candidate_splits(
    X, idx, attributes, splitter, max_thresholds, n_local_samples, random
):
    """
    Find potential splits for the current node based on attribute and threshold combinations.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix containing np.float32 values.
    idx : array-like of shape (n_samples,)
        Array indicating which samples are at the current node.
    attributes : array-like of shape (n_attributes,)
        Features that have non-zero variance at the parent node.
    splitter : {'best', 'random'}
        Method for finding splits. 'best' finds all possible splits, 'random' finds random splits between the bounds of each feature at the current node.
    max_thresholds : int
       Number of thresholds per attribute to sample randomly. Default in ExtraTrees is 1.
    n_local_samples : int
        Number of local samples at the current node.
    random : numpy.random.RandomState
        The random state object for controlling pseudo-random number generation.

    Returns
    -------
    thresholds : array of shape (n_thresholds,)
        Array of thresholds, sorted for each attribute.
    lens : array of shape (n_attributes,)
        Array indicating the number of thresholds for each attribute.

    Notes
    -----
    The function computes potential splits for the current node based on attribute and threshold combinations.
    The splits are defined by the selected attribute and threshold values.

    If there are no local samples (n_local_samples = 0 or 1), or no valid attributes to split on (n_attr = 0),
    the function returns empty arrays.

    If the splitter is set to 'random', the function generates random thresholds between the minimum and maximum
    values of each attribute. The number of thresholds per attribute is specified by max_thresholds.

    If the splitter is set to 'best', the function computes all possible thresholds for each attribute. The thresholds
    are computed as the midpoint between each unique value of the attribute.

    """

    n_attr = attributes.shape[0]
    lens = np.zeros((n_attr,), dtype=np.int32)
    n_samples = X.shape[0]

    if n_local_samples in [0, 1]:
        print(
            f" only {n_local_samples} samples at current node, there are no candidate splits"
        )
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int32)
    if n_attr == 0:
        print("there are no valid attributes to split on")
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int32)

    if splitter == "random":
        lens += max_thresholds
        if n_samples == n_local_samples:
            (a, b, c, d) = np.partition(
                X[:, attributes], kth=(1, n_local_samples - 2), axis=0
            )[[0, 1, n_local_samples - 2, n_local_samples - 1]]
        else:
            (a, b, c, d) = np.partition(
                X[np.ix_(idx, attributes)], kth=(1, n_local_samples - 2), axis=0
            )[[0, 1, n_local_samples - 2, n_local_samples - 1]]

        thresholds = (
            (a + 2 * (b - a) / 5)
            + (c + 3 * (d - c) / 5 - (a + 2 * (b - a) / 5))
            * random.rand(max_thresholds, n_attr)
        ).flatten(order="F")

    elif splitter == "best":
        thresholds = np.zeros((n_local_samples * n_attr,), dtype=np.float64)
        for i in range(n_attr):
            sorted_ = np.unique(X[idx, attributes[i]])
            n_unique = sorted_.shape[0]
            new_thresholds = (sorted_[1:] + sorted_[: n_unique - 1]) / 2
            num_thresholds = new_thresholds.shape[0]
            thresholds[np.sum(lens) : np.sum(lens) + num_thresholds] = new_thresholds
            lens[i] = num_thresholds
        thresholds = thresholds[: np.sum(lens)]

    if thresholds.shape[0] == 0:
        raise Warning(
            "something went wrong, no candidate splits in hf._candidate_splits"
        )

    return thresholds, lens


@njit
def _risk_reductions(
    attributes,
    thresholds,
    lens,
    X,
    y,
    idx,
    partial_risk,
    loss,
    n_classes,
    min_samples_leaf,
    gamma,
):
    """

    Parameters
    ----------
    attributes : array-like of shape (n_attributes,)
        Array of indices correpsonding to features that have non-zero variance at the parent node. Expected type: np.int32.
    thresholds : array-like of shape (n_thresholds,)
        Array of thresholds, sorted for each attribute. Expected type: np.float64.
    lens : array of shape (n_attributes,)
        Array indicating the number of thresholds for each attribute. Expected type: np.int32.
    X : array-like of shape (n_samples, n_features)
        Data matrix. Expected type: np.float32.
    y : array-like of shape (n_samples,)
        Array of labels. Expected type: np.int32.
    idx : array-like of shape (n_samples,)
        Indices of samples to consider in this node. Expected type: np.int32.
    partial_risk : float
        Partial risk of the parent node before the split.
    loss : str
        Loss function/splitting criteria for evaluating splits.
    n_classes : int
        Number of classes in the classification problem.
    min_samples_leaf : int
        Minimum number of samples required in each resulting leaf node.
    gamma : array-like of shape (2,)
        Array of hyperparameter values for specific loss functions/impurity measures.


    Returns
    -------
    best_rr : float
        The maximum risk reduction achieved among all potential splits.
    best_attribute : int
        The index of the attribute that yielded the best split.
    best_threshold : float
        The threshold value for the best split.
    best_left_child : (tuple)
        - idx (numpy.ndarray): Indices of samples in the left child node.
        - weights (numpy.ndarray): Class distribution of samples in the left child node.
        - partial_risk (float): Partial empirical risk of the left child node.
    best_right_child : (tuple)
        - idx (numpy.ndarray): Indices of samples in the right child node.
        - weights (numpy.ndarray): Class distribution of samples in the right child node.
        - partial_risk (float): Partial empirical risk of the right child node.
    """

    n_local_samples = idx.shape[0]
    n_attributes = attributes.shape[0]
    if (n_local_samples == 0) or (n_attributes == 0):
        raise Warning("No splits available")

    if np.sum(lens) != thresholds.shape[0]:
        raise Warning("number of thresholds does not match sum(lens)")

    first_thres_idx = np.concatenate((np.zeros((1,), dtype=np.int32), np.cumsum(lens)))[
        :n_attributes
    ]

    best_rr = np.float64(0)
    best_attribute = np.int32(-1)
    best_threshold = np.nan
    (best_left_child_idx, best_left_child_weights, best_left_child_pr) = (
        np.zeros((0,), dtype=np.int32),
        np.zeros((0,), dtype=np.int64),
        np.nan,
    )
    (best_right_child_idx, best_right_child_weights, best_right_child_pr) = (
        np.zeros((0,), dtype=np.int32),
        np.zeros((0,), dtype=np.int64),
        np.nan,
    )

    EPSILON = np.finfo(np.float64).eps

    for j in range(n_attributes):
        n_thresholds = lens[j]

        sorted_args = X[idx, attributes[j]].argsort()
        X_srt = X[idx, attributes[j]][sorted_args]
        y_srt = y[idx][sorted_args]

        bottom_idx = 0
        # first threshold for this attribute
        first_threshold = thresholds[first_thres_idx[j]]
        for k1 in range(bottom_idx, n_local_samples):
            if X_srt[k1] > first_threshold:
                break
        top_idx = k1

        left_weights = np.bincount(y_srt[bottom_idx:top_idx], minlength=n_classes)
        right_weights = np.bincount(y_srt[top_idx:], minlength=n_classes)
        prl = _partial_risk(y, loss, n_classes, left_weights, gamma)
        prr = _partial_risk(y, loss, n_classes, right_weights, gamma)

        if (np.sum(left_weights) < min_samples_leaf) or (
            np.sum(right_weights) < min_samples_leaf
        ):
            rr = np.float64(0)  # ignore this split
        else:
            if loss == "ranking":
                rr = (
                    abs(
                        left_weights[0] * right_weights[1]
                        - left_weights[1] * right_weights[0]
                    )
                    / 2
                )
            elif loss == "twoing":
                nl = left_weights.sum()
                nr = right_weights.sum()
                rr = (
                    (nl * nr / n_local_samples**2)
                    * np.abs(left_weights / nl - right_weights / nr).sum() ** 2
                    / 4
                )
            elif loss == "credal":
                if prl > prr:
                    rr = partial_risk - (
                        prl * (left_weights.sum() + gamma[0])
                        + prr * right_weights.sum()
                    ) / (n_local_samples + gamma[0])
                else:
                    rr = partial_risk - (
                        prl * left_weights.sum()
                        + prr * (right_weights.sum() + gamma[0])
                    ) / (n_local_samples + gamma[0])
            else:
                rr = partial_risk - (prl + prr)

        if rr > best_rr:
            # a better split found I
            best_rr = rr
            best_attribute = attributes[j]
            best_threshold = first_threshold
            (best_left_child_idx, best_left_child_weights, best_left_child_pr) = (
                idx[sorted_args][:top_idx].copy(),
                left_weights.copy(),
                prl,
            )
            (best_right_child_idx, best_right_child_weights, best_right_child_pr) = (
                idx[sorted_args][top_idx:],
                right_weights.copy(),
                prr,
            )

        if best_rr + EPSILON >= partial_risk:
            # stop looking for splits early I
            return (
                best_rr,
                best_attribute,
                best_threshold,
                (best_left_child_idx, best_left_child_weights, best_left_child_pr),
                (best_right_child_idx, best_right_child_weights, best_right_child_pr),
            )

        for i in range(1, n_thresholds):
            new_threshold = thresholds[first_thres_idx[j] + i]
            bottom_idx = top_idx
            for k2 in range(bottom_idx, n_local_samples):
                if X_srt[k2] > new_threshold:
                    break
            top_idx = k2

            weight_change = np.bincount(y_srt[bottom_idx:top_idx], minlength=n_classes)
            left_weights += weight_change
            right_weights -= weight_change
            prl = _partial_risk(y, loss, n_classes, left_weights, gamma)
            prr = _partial_risk(y, loss, n_classes, right_weights, gamma)

            if (np.sum(left_weights) <= min_samples_leaf) or (
                np.sum(right_weights) <= min_samples_leaf
            ):
                rr = np.float64(0)  # ignore this split
            else:
                if loss == "ranking":
                    rr = (
                        abs(
                            left_weights[0] * right_weights[1]
                            - left_weights[1] * right_weights[0]
                        )
                        / 2
                    )
                elif loss == "twoing":
                    nl = left_weights.sum()
                    nr = right_weights.sum()
                    rr = (
                        (nl * nr / n_local_samples**2)
                        * np.abs(left_weights / nl - right_weights / nr).sum() ** 2
                        / 4
                    )
                elif loss == "credal":
                    if prl > prr:
                        rr = partial_risk - (
                            prl * (left_weights.sum() + gamma[0])
                            + prr * right_weights.sum()
                        ) / (n_local_samples + gamma[0])
                    else:
                        rr = partial_risk - (
                            prl * left_weights.sum()
                            + prr * (right_weights.sum() + gamma[0])
                        ) / (n_local_samples + gamma[0])
                else:
                    rr = partial_risk - (prl + prr)

            if rr > best_rr:
                # a better split found II
                best_rr = rr
                best_attribute = attributes[j]
                best_threshold = new_threshold
                (best_left_child_idx, best_left_child_weights, best_left_child_pr) = (
                    idx[sorted_args][:top_idx].copy(),
                    left_weights.copy(),
                    prl,
                )
                (
                    best_right_child_idx,
                    best_right_child_weights,
                    best_right_child_pr,
                ) = (idx[sorted_args][top_idx:], right_weights.copy(), prr)

            if best_rr + EPSILON >= partial_risk:
                # stop looking for splits early II
                return (
                    best_rr,
                    best_attribute,
                    best_threshold,
                    (best_left_child_idx, best_left_child_weights, best_left_child_pr),
                    (
                        best_right_child_idx,
                        best_right_child_weights,
                        best_right_child_pr,
                    ),
                )

    return (
        best_rr,
        best_attribute,
        best_threshold,
        (best_left_child_idx, best_left_child_weights, best_left_child_pr),
        (best_right_child_idx, best_right_child_weights, best_right_child_pr),
    )


@njit
def _partial_risk(labels, loss, n_classes, weights, gamma):
    """
    Computes the partial risk of a node in a decision tree based on the given loss function.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Labels of the samples in the node.
    loss : str
        Loss function/impurity measure to be used for calculating the partial risk.
    n_classes : int
        Number of classes in the classification problem.
    weights : array-like of shape (n_classes)
        Counts of the number of samples from each class at the node.
    gamma : array-like of shape (2,)
        Hyperparameters for the specified loss function or impurity measure.

    Returns
    -------
    float
        The partial empirical risk for the node.

    Note
    ----
    The function uses numba njit for compatibility with other helper functions.

    """
    n_samples = labels.shape[0]
    EPSILON = np.finfo(np.float64).eps

    if weights.min() < 0:
        raise Warning("something went wrong, there are negative weights!")

    n_local_samples = np.sum(weights)
    total_weight = n_local_samples / n_samples  # proportion of all samples at node
    if total_weight <= 0:
        return 0  # if no samples at node

    props = weights / n_local_samples

    if loss == "mse":
        out = (1 - np.linalg.norm(props, ord=2) ** 2) * n_classes / (n_classes - 1)

    elif loss == "sqrt-gini":
        out = np.sqrt(
            (1 - np.linalg.norm(props, ord=2) ** 2) * n_classes / (n_classes - 1)
        )

    elif loss == "ce":
        out = np.sum(-props * np.log(props + EPSILON)) / np.log(n_classes)

    elif loss == "gce":
        out = (1 - np.linalg.norm(props, ord=1 / (1 - gamma[0]))) / (
            1 - 1 / (n_classes ** gamma[0])
        )

    elif loss == "conservative":
        out = (1 - np.max(props)) * n_classes / (n_classes - 1)

    elif loss == "ranking":
        out = weights[0] * weights[1] / (2 * total_weight)

    elif loss == "twoing":
        out = (1 - np.max(props)) * n_classes / (n_classes - 1)

    elif loss == "ne":
        rob = (1 - np.max(props)) * n_classes / (n_classes - 1)
        non = np.sqrt(
            (1 - np.linalg.norm(props, ord=2) ** 2) * n_classes / (n_classes - 1)
            + EPSILON
        )
        out = min(gamma[0] * non, rob)

    elif loss == "credal":
        props_new = weights.astype(np.float64)
        A = np.where(weights == weights.min())[0]
        props_new[A] += gamma[0] / A.shape[0]
        props_new /= n_local_samples + gamma[0]
        return np.sum(-props_new * np.log(props_new + EPSILON))

    else:
        raise Warning("an invalid loss function was specified")

    if abs(out) < EPSILON:
        out = np.float64(0)

    return out * total_weight


def _split(
    X,
    y,
    idx,
    valid_attributes,
    partial_risk,
    splitter,
    max_thresholds,
    loss,
    n_classes,
    n_local_samples,
    min_samples_leaf,
    gamma,
    random,
):
    """
    Split a node.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix. Expected type: np.float32.
    y : array-like of shape (n_samples,)
        Array of labels. Expected type: np.int32.
    idx : array-like of shape (n_samples,)
        Array indicating which samples are at the current node.
    valid_attributes : array-like of shape (n_features,)
        Features that have non-zero variance at the parent node.
    partial_risk : float
        Partial risk (weighted impurity) of the current node.
    splitter : {'random', 'best'}
        Method for finding splits.
    max_thresholds : int
        Number of thresholds to sample per attribute if splitter == 'random'.
    loss : str
        Loss function used to train the tree.
    n_classes : int
        Total number of classes in the classification problem.
    n_local_samples : int
        Number of samples at the node.
    random : numpy.random.RandomState
        The random state object for controlling pseudo-random number generation.

    Returns
    -------
    tuple
        Returns a tuple containing the result of _risk_reductions.

    """
    candidate_thresholds, lengths = _candidate_splits(
        X, idx, valid_attributes, splitter, max_thresholds, n_local_samples, random
    )
    return _risk_reductions(
        valid_attributes,
        candidate_thresholds,
        lengths,
        X,
        y,
        idx,
        partial_risk,
        loss,
        n_classes,
        min_samples_leaf,
        gamma,
    )


def _update_queue(
    queue=None, children=None, max_leaf_nodes=None, queue_RRs=None, child_RRs=None
):
    """
    Update the queue of nodes to visit next.

    There are currently two supported node traversal methods:
        - pre-order: child nodes get visited first
        - best-first: visit most promising nodes first

    Parameters
    ----------
    queue : list, optional
        A list of node labels in the order that they should be visited. Default is None.
    children : array of shape (n_child_nodes), optional
        The child nodes to be added to the queue. Default is None.
    max_leaf_nodes : int, optional
        Maximum number of leaf nodes of the entire tree. Default is None.
    queue_RRs : list, optional
        The risk reductions of each of the nodes in the queue. Only used in best-first search. Default is None.
    child_RRs : list, optional
        Risk reductions of the child nodes to be added to the queue. Default is None.

    Raises
    ------
    Warning
        Raised if the length of the queue and queue_RRs is different, indicating an error.

    Returns
    -------
    list
        Updated queue.
    list
        Risk reductions of updated queue (zeros if using pre-order traversal).

    """
    # remove last visited node and add children
    if max_leaf_nodes < np.inf:
        # best-first search
        # grab the risk reductions of nodes already in the queue, grab the rrs of child nodes, then sort (desc) the list of nodes by the RRs

        if len(queue) != len(queue_RRs):
            raise Warning(
                f"queue and queue_RRs should have the same len ({len(queue)} and {len(queue_RRs)}, resp.)"
            )
        all_RRs = queue_RRs[1:] + child_RRs
        all_nodes = queue[1:] + list(children)
        sorted_args = np.argsort(-np.array(all_RRs))
        return list(np.array(all_nodes)[sorted_args]), list(
            np.array(all_RRs)[sorted_args]
        )
    else:
        # pre-order traversal
        # put children at front of queue
        return list(children) + queue[1:], list(
            np.zeros(len(children) + len(queue[1:]))
        )
