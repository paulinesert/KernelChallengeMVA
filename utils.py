import numpy as np 
import matplotlib.pyplot as plt

def plot_image(img):
    '''
    Reshape the flatten image and plot it.
    Input: 
        img : flatten image as an array of size (3072,)  
    '''
    r, g, b = img[0:1024].reshape(32,32), img[1024:2048].reshape(32,32), img[2048:3072].reshape(32,32)
    x = np.stack((r,g,b))
    x = np.moveaxis(x, (0,1,2),(2,0,1))
    x = (x - x.min()) / (x.max() - x.min())
    plt.imshow(x)

def compute_rgb_hist(X, n_bins):
    '''
    Computes the normalized histograms of each channels of the input images in X and output their concatenation.
    Input:
        X: n flatten images, array of shape (n, 3072)
        n_bins: number of bins to use to compute the histogram 
    Ouput:
        hist: for all the images, the concatenation of the 3 channels' histograms as an array of shape (n, 3*n_bins)
    '''
    # Fetch each channel
    R = X[:, 0:1024]
    G = X[:, 1024:2048]
    B = X[:, 2048:3072]

    # Compute the histogram for each channel and normalize it
    hist_R = (1/1024) * np.apply_along_axis(lambda x: np.histogram(x, range=(-0.5, 0.5), bins=n_bins)[0], 1, R)
    hist_G = (1/1024) * np.apply_along_axis(lambda x: np.histogram(x, range=(-0.5, 0.5), bins=n_bins)[0], 1, G)
    hist_B = (1/1024) * np.apply_along_axis(lambda x: np.histogram(x, range=(-0.5, 0.5), bins=n_bins)[0], 1, B)

    # Concatenate the 3 histograms for each image
    hist = np.concatenate([hist_R, hist_G, hist_B], axis=1)

    return hist

def compute_brightness_hist(X, n_bins):
    '''
    Computes the normalized histograms of each channels of the input images in X and output their concatenation.
    Input:
        X: n flatten images, array of shape (n, 3072)
        n_bins: number of bins to use to compute the histogram 
    Ouput:
        hist: for all the images, the histogram of the brightness (n, n_bins)
    '''
    # Fetch each channel
    R = X[:, 0:1024]
    G = X[:, 1024:2048]
    B = X[:, 2048:3072]

    # Compute brightness
    brightness = np.mean([R,G,B], axis=0)

    # Compute the histogram
    hist = (1/1024) * np.apply_along_axis(lambda x: np.histogram(x, range=(-0.5, 0.5), bins=n_bins)[0], 1, brightness)

    return hist

def gray_scale(im):
    '''
    Turns to gray scale an image. 
    '''
    return (1/3)*np.sum(im,-1)

#normalize
def normalize_image(im):
    '''
    Normalizes the image.
    '''
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def gradients(im):
    '''
    Compute the horizontal and vertical gradients.
    '''
    gx,gy = np.gradient(im)
    return gx,gy

def magnitude(dx,dy):
    '''
    Compute the magnitude.
    '''
    return np.sqrt(dy**2 + dx**2)

def orientation(dx,dy):
    '''
    Get the orientation.
    '''
    return np.rad2deg(np.arctan2(dy, dx)) % 180

def hog_cell(n_bins,mag,ori):
    '''
    Compute HOG for a given cell.
    Input:
        n_bins: number of bins for the histogram's computation
        mag: magnitude
        ori: orientation
    Output:
        HOG 
    '''
    num_ppc = mag.shape[0]*mag.shape[1]
    x,y = np.histogram(ori,bins=n_bins,weights=mag)
    return x / num_ppc

def normalize(v,eps=1e-5):
    '''
    L2 histogram normalization.
    '''
    return v / np.sqrt(np.sum(v**2) + eps**2) 

def hog_features(im, n_bins, ppc, cpb):
    '''
    Compute HOG for an image.
    ''' 
    im = gray_scale(im)
    im = normalize_image(im)

    gy,gx = gradients(im)
    mag = magnitude(gx, gy)
    ori = orientation(gx, gy)

    cx,cy = ppc
    bx,by = cpb

    sy,sx = gx.shape

    num_cells_x = int(sx / cx)
    num_cells_y = int(sy / cy)

    num_blocks_x = int(num_cells_x - bx) + 1
    num_blocks_y = int(num_cells_y - by) + 1

    # Compute HOG of each cells
    hogs = np.zeros((num_cells_x, num_cells_y,n_bins))
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            mag_cell = mag[cx*i:cx*(i+1),cy*j:cy*(j+1)]
            ori_cell = ori[cx*i:cx*(i+1),cy*j:cy*(j+1)]   
            hogs[i,j] = hog_cell(n_bins,mag_cell,ori_cell)

    # Compute block normalization
    features = np.zeros((num_blocks_x,num_blocks_y,n_bins))
    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            hog_block = hogs[i:i+bx,j:j+by].ravel()
            features[i,j] = normalize(hog_block)

    return features.ravel()

def compute_hog_features(X, n_bins, ppc, cpb):
    '''
    Reshape flatten RGB image and compute its HOG features.
    '''
    r, g, b = X[0:1024].reshape(32,32), X[1024:2048].reshape(32,32), X[2048:3072].reshape(32,32)
    x = np.stack((r,g,b))
    x = np.moveaxis(x, (0,1,2),(2,0,1))
    hog = hog_features(x, n_bins=n_bins, ppc=ppc, cpb=cpb)
    print(hog.shape)
    return hog


def labeling(Y, pos_class) :  
    '''
    Transform the labels into +1 / -1 labels.
    Input:
        Y: the n labels, array of shape (n,)
        pos_class: the label of the positive class
    Ouput:
        Y_transformed: the labels turned into +1 (positive) and -1 (negative) labels
    '''
    Y_transformed = np.where(Y==pos_class, 1, -1)  
    return Y_transformed

def find_most_voted_class(preds):
    '''
    For a list of predicted class, find the one that occurs the most. 
    Input:
        preds: list of predicted classes
    Output:
        pred: a single prediction
    '''
    values, counts = np.unique(preds, return_counts=True)
    pred = values[counts.argmax()]
    return pred 

def make_predictions(X, models, pairs):
    '''
    For each binary classifier model, select the associated pair of classes and make a prediction on the dataset.
    Input:
        X: dataset to predict, array of size (n_samples, n_features)
        models: a list of models 
        pairs: a list of tuple containing a positive and a negative class
    Output:
        predictions: the predicted classes for each sample (one for each binary classifier), list of M arrays of size (n_samples,) (where M is the number of models)
    '''
    predictions = []
    for i,model in enumerate(models):
        pos_label, neg_label = pairs[i]
        prediction = model.predict(X)
        prediction = np.where(prediction==1, pos_label, neg_label)
        predictions.append(prediction) 
    return predictions