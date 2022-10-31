import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import rgb2gray,load_image
import scipy.signal as sp

def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################
    x=[]
    y=[]
    #Convert to gray
    if len(image.shape)==3:
        gray_img = rgb2gray(image)
    else:
        gray_img=image
    #Compute derivative
    soble_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    soble_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # TODO: difference between convolve2d and filter2D?
    I_x = cv2.Sobel(gray_img, -1, 1, 0, ksize=3)
    I_y = cv2.Sobel(gray_img, -1, 0, 1, ksize=3)
    
    Ixx = I_x**2
    Iyy = I_y**2
    Ixy = I_x*I_y

    # har = np.copy(image)

    height, width = gray_img.shape

    gaussian = cv2.getGaussianKernel(ksize=9, sigma=2)


    Gxx = cv2.filter2D(Ixx, -2, gaussian)
    Gyy = cv2.filter2D(Iyy, -2, gaussian)
    Gxy = cv2.filter2D(Ixy, -2, gaussian)

    det = Gxx*Gyy - Gxy**2
    trace = Gxx+Gyy
    alpha = 0.04
    har = det - alpha*(trace**2)

    # cv2.normalize(har,har,0,1,cv2.NORM_MINMAX)

    max = np.max(har)

    for j in range(height):
        for i in range(width):
            r = har[j][i]
            if r>0.02*max:
                x.append(i)
                y.append(j)

                
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return np.array(x),np.array(y), confidences, scales, orientations


