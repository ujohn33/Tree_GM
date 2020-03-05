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

# if the parent value of a node checked in topology equals given node, then the checked node is a child of given node
def find_children(p, topology):
    children = []
    for index, parent in enumerate(topology):
        if parent == p:
            children.append(index)
    return children

def calculate_likelihood(tree_topology, theta, beta):
    # number of categories of assignments
    cat = len(theta[0])
    # calculate s for the node
    def calculate_s(node):
        # nodes have 2 children: left and right
        children = find_children(node, tree_topology)
        # identify leaves and reveal assignment likelihood (observed => 1)
        if(len(children)<1):
            likelihood=np.zeros(cat)
            likelihood[int(beta[node])]=1
            return likelihood
        s_left=calculate_s(children[0])
        s_right=calculate_s(children[1])
        left_likelihood=np.zeros(cat)
        for i in range(cat):
            left_likelihood[i]=np.dot(theta[children[0]][i],s_left)
        right_likelihood=np.zeros(cat)
        for i in range(cat):
            right_likelihood[i]=np.dot(theta[children[1]][i],s_right)
        return left_likelihood*right_likelihood
    likelihood=np.dot(calculate_s(0), theta[0])
    return likelihood


def main():
    print("Hello World!")
    print("This file is the solution template for question 2.3.")

    print("\n1. Load tree data from file and print it\n")
    filename = "data/q2_3_medium_tree.pkl"  # "data/q2_3_medium_tree.pkl", "data/q2_3_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        #print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        print("\tSample: ", sample_idx)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
