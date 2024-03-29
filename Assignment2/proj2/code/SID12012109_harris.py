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
    responses=[]
    #Convert to gray
    if len(image.shape)==3:
        gray_img = rgb2gray(image)
    else:
        gray_img=image
    #Compute derivative
    soble_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    soble_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # TODO: difference between convolve2d and filter2D?
    I_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    I_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    
    Ixx = I_x**2
    Iyy = I_y**2
    Ixy = I_x*I_y

    # har = np.copy(image)

    height, width = gray_img.shape

    gaussian = cv2.getGaussianKernel(ksize=5, sigma=2)
    # gaussian = np.dot(gaussian,gaussian)


    Gxx = cv2.filter2D(Ixx, cv2.CV_32F, gaussian)
    Gyy = cv2.filter2D(Iyy, cv2.CV_32F, gaussian)
    Gxy = cv2.filter2D(Ixy, cv2.CV_32F, gaussian)

    Gxx = cv2.filter2D(Ixx, cv2.CV_32F, gaussian.T)
    Gyy = cv2.filter2D(Iyy, cv2.CV_32F, gaussian.T)
    Gxy = cv2.filter2D(Ixy, cv2.CV_32F, gaussian.T)


    det = Gxx*Gyy - Gxy**2
    trace = Gxx+Gyy
    alpha = 0.06
    har = det - alpha*(trace**2)

    # cv2.normalize(har,har,0,1,cv2.NORM_MINMAX)

    max = np.max(har)

    for j in range(height):
        for i in range(width):
            r = har[j][i]
            if r>0.01*max:
                x.append(i)
                y.append(j)
                responses.append(r)
                
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    #############################################################################
    image = np.transpose(image) # y,x -> x,y
    response_pairs=[]
    for i in range(len(responses)):
        response_pairs.append([x[i],y[i],responses[i]])
    response_pairs = sorted(response_pairs, key= lambda x:x[2],reverse=True) # sort by response value
    response_pairs = np.array(response_pairs)

    fig_distance = np.sqrt(image.shape[0]**2+image.shape[1]**2)
    print(fig_distance)
    radiis = np.full(len(responses),fig_distance)

    
    for i in range(1,len(response_pairs)):
        x = response_pairs[i,0]
        y = response_pairs[i,1]

        min_distance = radiis[0]
        for j in range(i): # iterate elements before i
            
            stronger_x = response_pairs[j,0]
            stronger_y = response_pairs[j,1]
            distance = np.sqrt(np.square(stronger_x-x)+np.square(stronger_y-y))
            if distance<min_distance:
                min_distance=distance
        radiis[i]=min_distance

    sorted_radiis = np.array(radiis)
    print(radiis)
    sorted_radiis = np.argsort(sorted_radiis) 
    # sorted_radiis = np.flip(sorted_radiis)
    print(sorted_radiis)
    # response_pairs = np.hstack((response_pairs,radiis))
    # response_pairs = sorted(response_pairs, key=lambda x:x[3], reverse=True)
    x=[]
    y=[]
    confidences=[]
    for i in range(1500): # n=1500

        x.append(response_pairs[sorted_radiis[len(sorted_radiis)-i-1],0])
        y.append(response_pairs[sorted_radiis[len(sorted_radiis)-i-1], 1])
        confidences.append(response_pairs[sorted_radiis[len(sorted_radiis)-i-1], 2])




    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return np.array(x),np.array(y), confidences, scales, orientations


