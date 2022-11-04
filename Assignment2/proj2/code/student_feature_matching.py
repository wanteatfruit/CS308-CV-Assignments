import numpy as np


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                        #
    #############################################################################
    match_pairs = []
    confidences = []

    for i in range(features1.shape[0]):
        distances = []
        for j in range(features2.shape[0]):
                dist = np.linalg.norm(features1[i]-features2[j])
                distances.append(dist)
        distances = np.array(distances)
        sorted_dist = np.argsort(distances) # ascending order
        best = distances[sorted_dist[0]]
        second_best = distances[sorted_dist[1]]
        ratio = best/second_best
        if ratio <0.8:
                match_pairs.append([i,sorted_dist[0]])
                confidences.append(ratio)

    match_pairs = np.array(match_pairs)
    confidences = np.array(confidences)
    matches = match_pairs.astype(int)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences
