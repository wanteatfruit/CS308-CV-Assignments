import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################

    image = np.transpose(image) # row, column
    print(image.shape)
    key_points = list(zip(x,y)) # row, column
    size = int(feature_width/4)
    image = np.pad(image, ((size, size),(size,size))) #prevent out of bound
    # gradients
    gx = np.gradient(image,axis=1)
    gy = np.gradient(image,axis=0)
    grad_mag = np.sqrt(gx**2+gy**2)
    grad_dir = np.arctan2(gy, gx)*180/np.pi


    fv=[]
    for kp in key_points:
        des = find_descriptor(kp,size,image,gx,gy,grad_mag,grad_dir)
        fv.append(des)
        
    return np.array(fv).reshape(len(key_points),128)


def find_descriptor(key_point, size, image, gx, gy, grad_mag, grad_dir):
    # create 4*4 cell
    # ----
    desc = []  # 8 histogram
    x, y = key_point
    for ix in range(-2, 2):
        for iy in range(-2, 2):
            # up left coordinate
            cell_x = x+size*ix
            cell_y = y+size*iy
            hist = find_histogram(gx, gy, image, cell_x,
                                  cell_y, size, grad_mag, grad_dir)
            desc.append(hist)
    descriptor = np.array(desc).reshape((128, 1))

    # normalize
    descriptor = descriptor/np.linalg.norm(descriptor)
    descriptor = np.clip(descriptor, 0, 0.2) # follow HOG paper
    descriptor = descriptor / np.linalg.norm(descriptor)

    return descriptor


def find_histogram(gx, gy, image, cell_x, cell_y, size, grad_mag, grad_dir):
    hist = np.zeros(8)  # [0,45,90,135,180,225,270,325]
    bins = [0,45,90,135,180,225,270,325,360]
    for x in range(size):
        for y in range(size):
            cur_x = cell_x+x
            cur_y = cell_y+y
            magnitude = grad_mag[int(cur_x), int(cur_y)]
            direction = grad_dir[int(cur_x), int(cur_y)]
            if direction < 0:
                direction += 360
            if direction == 360:
                direction = 0
            bin_left = int(direction//45)
            bin_right = int(direction//45+1)
            dist_left = np.absolute(direction-bins[bin_left])
            dist_right = np.absolute(bins[bin_right]-direction)  # could be 360
            # calculate proportion, smaller distance has greater weights
            prop_left = (45-dist_left)/45
            prop_right = (45-dist_right)/45

            hist[bin_left] += prop_left*magnitude
            if bin_right==8:
                bin_right=0
            hist[bin_right] += prop_right*magnitude

            # Note that if a pixel is halfway between two bins,
            # then it splits up the magnitudes accordingly
            # depending on their distance away from each respective bin     .

    return hist
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
