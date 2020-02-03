""" This file is created as the solution template for question 2.3 in DD2434 - Assignment 2.

    Please keep the fixed parameters in the function templates as is (in 2_3.py file).
    However if you need, you can add parameters as default parameters.
    i.e.
    Function template: def calculate_likelihood(tree_topology, theta, beta):
    You can change it to: def calculate_likelihood(tree_topology, theta, beta, new_param_1=[], new_param_2=123):

    You can write helper functions however you want.

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the subsolutions.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you three different trees (q_2_3_small_tree, q_2_3_medium_tree, q_2_3_large_tree).
    Each tree have 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.
"""
import numpy as np
from Tree import Tree
from Tree import Node
from collections import defaultdict


# probability of child with assigned value cat with a parrent with assigned value parent_cat
def CPD(theta, node, cat, parent_cat):
    if parent_cat is None:
        return theta[node][cat]
    else:
        return theta[node][int(parent_cat)][int(cat)]

# non-nan values are assignments for leaves. nan values are inner nodes
def find_leaves(beta):
    leaves = []
    for index, value in enumerate(beta):
        if not np.isnan(value):
            leaves.append(index)
    return leaves

# if the parent value of a node checked in topology equals given node, then the checked node is a child of given node
def find_children(p, topology):
    children = []
    for index, parent in enumerate(topology):
        if parent == p:
            children.append(index)
    return children

# if the parent is the same but the node has a different index
def find_sibling(u, topology):
    for index, parent in enumerate(topology):
        if parent == topology[u] and u != index:
            return index
    return None

def calculate_likelihood(tree_topology, theta, beta):
    # initialize dictionaries for s and t
    likelihood = 0
    S_VALUES = defaultdict(dict)
    t_VALUES= defaultdict(dict)

    def find_s(tree_topology, theta, beta):
        def subproblem_S(u, j, children):
            if S_VALUES[u].get(j) is not None: # If it has already been calculated
                return S_VALUES[u][j]
            # Visit the vertices from leaves to root
            if len(children) < 1:           # identify leaves
                if int(beta[u]) == j:
                    S_VALUES[u][j] = 1
                    return 1
                else:
                    S_VALUES[u][j] = 0
                    return 0
            subsolution = np.zeros(len(children))
            # run down the tree solving the subproblem for children of children recursively
            for index, child in enumerate(children):
                for category in range(0, len(theta[0])):
                    subsolution[index] += subproblem_S(child, category, find_children(child, tree_topology)) * CPD(theta, child,category, j)
            s_subsolution = np.prod(subsolution)
            S_VALUES[u][j] = s_subsolution
            return s_subsolution
        # Start from root and find run the subproblem for children of the root node
        for i in range(len(theta[0])):
            subproblem_S(0, i, find_children(0, tree_topology))
        return S_VALUES

    S_VALUES = find_s(tree_topology, theta, beta)
    # Do the same for t, now syblings are taken into consideration
    def subproblem_t(u, i, parent, sibling):
        if t_VALUES[u].get(i) is not None:  # If it has already been calculated
            return t_VALUES[u][i]

        if np.isnan(parent):  # identify the root
            return CPD(theta, u, i, None) * S_VALUES[u][i]
        subsolution = 0
        if sibling is None:  # simplified if there is no siblings
            for j in range(0, len(theta[0])):
                subsolution += CPD(theta, u, i, j) * t(parent, j, tree_topology[parent],subproblem_sibling(parent, tree_topology))
                t_VALUES[u][i] = subsolution
            return subsolution
        parent = int(parent)
        for j in range(0, len(theta[0])):
            for k in range(0, len(theta[0])):
                subsolution += CPD(theta, u, i, j) * CPD(theta, sibling, k, j) * S_VALUES[sibling][k] * subproblem_t(parent,j,tree_topology[parent],find_sibling(parent,tree_topology))
        t_VALUES[u][i] = subsolution
        return subsolution

    # Expressing the joint
    for leaf, cat in enumerate(beta):
        if not np.isnan(cat):
            likelihood = subproblem_t(leaf, cat, int(tree_topology[leaf]),find_sibling(leaf, tree_topology)) * S_VALUES[leaf][cat]
    print("Calculating the likelihood...")
    #likelihood = np.random.rand()

    return likelihood


def main():
    print("Hello World!")
    print("This file is the solution template for question 2.3.")

    print("\n1. Load tree data from file and print it\n")
    filename = "data/q2_3_small_tree.pkl"  # "data/q2_3_medium_tree.pkl", "data/q2_3_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
