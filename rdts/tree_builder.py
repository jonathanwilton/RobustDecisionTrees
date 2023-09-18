import time

import numpy as np

import dts.helper_functions as hf
from dts.node import Node


def grow_dt(init_args, X, y, additional_args, random):
    """
    Build a decision tree based on the provided training data.

    Parameters
    ----------
    init_args : tuple
        A tuple of initial arguments and hyperparameters for the tree.
    X : array-like of shape (n_samples, n_features)
        The input feature matrix.
    y : array-like of shape (n_samples,)
        The target labels.
    additional_args : tuple
        A tuple of additional arguments and hyperparameters.
    random : numpy.random.RandomState
        The random state object for controlling pseudo-random number generation.

    Returns
    -------
    nodes : dict
        A dictionary where keys are node indices and values are corresponding Node objects.
    n_leaf_nodes : int
        The number of leaf nodes in the trained tree.
    tree_depth : int
        The depth of the trained tree.

    """

    (
        loss,
        max_depth,
        max_leaf_nodes,
        min_samples_leaf,
        splitter,
        max_thresholds,
        min_risk_decrease,
    ) = init_args
    n_samples, n_features, max_features, n_classes_, gamma = additional_args

    tree_depth = 0
    tentative_leaf_nodes = [
        0
    ]  # leaf nodes and nodes that could possibly turn into leaf nodes, used when max_leaf_nodes is not None
    queue = [0]  # queue of nodes to visit next, start with root node (0)
    queue_RRs = [
        float("inf")
    ]  # risk reductions of nodes in queue, initial value is placeholder
    root_node = Node(
        depth=0,
        idx=np.arange(n_samples, dtype=np.int32),
        n_local_samples=n_samples,
    )
    root_node.weights = np.bincount(y, minlength=n_classes_)
    root_node.partial_risk = hf._partial_risk(
        y, loss, n_classes_, root_node.weights, gamma
    )
    nodes = {queue[0]: root_node}  # tree stored here
    newest_node = 0
    ghost_nodes = (
        {}
    )  # keys: parent_node, values: dictionary with keys {0,1}, values: Node objects. Used in best first search since splitting happens before queue updates

    # utilize a switch to avoid doing uncessary computations
    switch = False
    EPSILON = np.finfo(np.float64).eps

    # grow tree until no nodes left to visit
    while len(queue) > 0:
        current_node = queue[0]

        # incrementally check termination conditions -> avoids unnecessary computation (sometimes)
        c0 = nodes[current_node].depth < max_depth  # check max depth reached
        c1 = (
            nodes[current_node].n_local_samples > min_samples_leaf
        )  # minimum samples in node reached
        c2 = (
            len(tentative_leaf_nodes) < max_leaf_nodes
        )  # check num leaf nodes # -> if this limit is reached then remainder of nodes in the queue must be turned into leaves
        if np.any(~np.array([c0, c1, c2])):
            nodes, queue, queue_RRs = create_leaf(nodes, queue, queue_RRs)
            switch = False  # do not check subsequent termination criteria
        else:
            switch = True

        # check node purity
        if switch:
            if nodes[current_node].partial_risk <= EPSILON:
                nodes, queue, queue_RRs = create_leaf(nodes, queue, queue_RRs)
                switch = False
            else:
                switch = True

        # now check to see if there is any variability in data at node
        if switch:
            if nodes[current_node].varying_attributes is None:
                if current_node == 0:
                    nodes[current_node].varying_attributes = hf._varying_attributes(
                        X,
                        nodes[current_node].idx,
                        nodes[current_node].n_local_samples,
                        np.arange(n_features, dtype=np.int32),
                    )
                else:
                    parent_node = nodes[current_node].parent_idx
                    nodes[current_node].varying_attributes = hf._varying_attributes(
                        X,
                        nodes[current_node].idx,
                        nodes[current_node].n_local_samples,
                        nodes[parent_node].varying_attributes,
                    )
            n_varying_atts = nodes[current_node].varying_attributes.shape[0]
            if n_varying_atts <= 0:
                nodes, queue, queue_RRs = create_leaf(nodes, queue, queue_RRs)
                switch = False
            else:
                switch = True

        # now for the RR check
        if switch:
            n_candidate_atts = min(n_varying_atts, max_features, n_features)
            candidate_atts = random.choice(
                nodes[current_node].varying_attributes,
                size=n_candidate_atts,
                replace=False,
            )
            if nodes[current_node].risk_reduction is None:
                (
                    nodes[current_node].risk_reduction,
                    nodes[current_node].attribute,
                    nodes[current_node].threshold,
                    (left_child_idx, left_child_weights, left_partial_risk),
                    (right_child_idx, right_child_weights, right_partial_risk),
                ) = hf._split(
                    X,
                    y,
                    nodes[current_node].idx,
                    candidate_atts,
                    nodes[current_node].partial_risk,
                    splitter,
                    max_thresholds,
                    loss,
                    n_classes_,
                    nodes[current_node].n_local_samples,
                    min_samples_leaf,
                    gamma,
                    random,
                )
            else:
                pass

            queue_RRs[0] = nodes[current_node].risk_reduction

            if (
                nodes[current_node].risk_reduction <= min_risk_decrease + EPSILON
            ):  # check RR > threshold
                nodes, queue, queue_RRs = create_leaf(nodes, queue, queue_RRs)
                switch = False
            else:
                # do not turn into leaf node -> split
                nodes[current_node].is_leaf = False
                tentative_leaf_nodes.remove(current_node)

                # create child nodes
                children = np.array([newest_node + 1, newest_node + 2], dtype=np.int32)
                nodes[
                    current_node
                ].child_idxs = children  # register new children with parent
                tentative_leaf_nodes += list(
                    children
                )  # before checking if children are leaves
                newest_node += 2
                if len(ghost_nodes) == 0:
                    nodes[children[0]] = Node(
                        depth=nodes[current_node].depth + 1,
                        parent_idx=current_node,
                        relation_from_parent="left",
                        idx=left_child_idx,
                        n_local_samples=left_child_idx.shape[0],
                        weights=left_child_weights,
                        partial_risk=left_partial_risk,
                    )
                    nodes[children[1]] = Node(
                        depth=nodes[current_node].depth + 1,
                        parent_idx=current_node,
                        relation_from_parent="right",
                        idx=right_child_idx,
                        n_local_samples=right_child_idx.shape[0],
                        weights=right_child_weights,
                        partial_risk=right_partial_risk,
                    )
                else:
                    # load ghost nodes
                    nodes[children[0]] = ghost_nodes[current_node][0]
                    nodes[children[1]] = ghost_nodes[current_node][1]
                    del ghost_nodes[current_node]  # no longer 'ghost' nodes

                # check if depth of tree has increased
                if nodes[current_node].depth + 1 > tree_depth:
                    tree_depth = nodes[current_node].depth + 1
                    # print(f"tree depth increased to {tree_depth}")

                # update queue
                # pre-order traversal
                if max_leaf_nodes == float("inf"):
                    queue, queue_RRs = hf._update_queue(
                        queue=queue, children=children, max_leaf_nodes=max_leaf_nodes
                    )  # add new leaf nodes to queue and remove current node

                # otherwise we may need the risk reductions of the new children to update queue in best first fashion
                else:
                    child_RRs = []
                    for child_node in children:
                        stop_child_early = False

                        c0 = nodes[child_node].depth >= max_depth
                        c1 = nodes[child_node].n_local_samples <= min_samples_leaf
                        c2 = len(tentative_leaf_nodes) >= max_leaf_nodes
                        c3 = nodes[child_node].partial_risk <= 0
                        if (c0 or c1 or c2 or c3) and not stop_child_early:
                            nodes[child_node].risk_reduction = np.float64(0)
                            child_RRs += [
                                1.1
                            ]  # RR 1.1 guarantees position at start of queue
                            stop_child_early = True

                        if not stop_child_early:
                            nodes[
                                child_node
                            ].varying_attributes = hf._varying_attributes(
                                X,
                                nodes[child_node].idx,
                                nodes[child_node].n_local_samples,
                                nodes[current_node].varying_attributes,
                            )
                            n_varying_atts = nodes[child_node].varying_attributes.shape[
                                0
                            ]
                            if n_varying_atts <= 0:
                                nodes[child_node].risk_reduction = np.float64(0)
                                child_RRs += [1.1]
                                stop_child_early = True

                        if not stop_child_early:
                            n_candidate_atts = min(
                                n_varying_atts, max_features, n_features
                            )
                            candidate_atts = random.choice(
                                nodes[child_node].varying_attributes,
                                size=n_candidate_atts,
                                replace=False,
                            )
                            (
                                nodes[child_node].risk_reduction,
                                nodes[child_node].attribute,
                                nodes[child_node].threshold,
                                (
                                    left_ghost_idx,
                                    left_ghost_weights,
                                    left_ghost_partial_risk,
                                ),
                                (
                                    right_ghost_idx,
                                    right_ghost_weights,
                                    right_ghost_partial_risk,
                                ),
                            ) = hf._split(
                                X,
                                y,
                                nodes[child_node].idx,
                                candidate_atts,
                                nodes[child_node].partial_risk,
                                splitter,
                                max_thresholds,
                                loss,
                                n_classes_,
                                nodes[child_node].n_local_samples,
                                min_samples_leaf,
                                gamma,
                                random,
                            )

                            if (
                                nodes[child_node].risk_reduction
                                <= min_risk_decrease + EPSILON
                            ):
                                child_RRs += [1.1]
                            else:
                                # by finding child node's RR, some "hyper"-child nodes were implicitly created. Store this information to use later
                                child_RRs += [nodes[child_node].risk_reduction]
                                ghost_nodes[child_node] = {
                                    0: Node(
                                        depth=nodes[child_node].depth + 1,
                                        parent_idx=child_node,
                                        relation_from_parent="left",
                                        idx=left_ghost_idx,
                                        n_local_samples=left_ghost_idx.shape[0],
                                        weights=left_ghost_weights,
                                        partial_risk=left_ghost_partial_risk,
                                    ),
                                    1: Node(
                                        depth=nodes[child_node].depth + 1,
                                        parent_idx=child_node,
                                        relation_from_parent="right",
                                        idx=right_ghost_idx,
                                        n_local_samples=right_ghost_idx.shape[0],
                                        weights=right_ghost_weights,
                                        partial_risk=right_ghost_partial_risk,
                                    ),
                                }

                    queue, queue_RRs = hf._update_queue(
                        queue=queue,
                        children=children,
                        max_leaf_nodes=max_leaf_nodes,
                        queue_RRs=queue_RRs,
                        child_RRs=child_RRs,
                    )

                switch = True

    return nodes, len(tentative_leaf_nodes), tree_depth


def create_leaf(nodes, queue, queue_RRs):
    """
    Create a leaf node in the decision tree and update the queue accordingly.

    Parameters
    ----------
    nodes : dict
        A dictionary where keys are node indices and values are corresponding Node objects.
    queue : list
        A list representing the queue of nodes to visit.
    queue_RRs : list
        A list of risk reductions corresponding to the nodes in the queue.

    Returns
    -------
    nodes : dict
        The updated dictionary of nodes with the created leaf nodes.
    queue : list
        The updated queue with the first node removed.
    queue_RRs : list
        The updated list of risk reductions with the first element removed.

    """
    nodes[queue[0]].is_leaf = True
    queue = queue[1:]
    queue_RRs = queue_RRs[1:]
    return nodes, queue, queue_RRs
