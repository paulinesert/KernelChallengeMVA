from utils import compute_brightness_hist, compute_hog_features, compute_rgb_hist, find_most_voted_class, labeling, make_predictions 
from kernels import RBF
from sklearn.model_selection import train_test_split
from SVC import KernelSVC
from tqdm import tqdm


import numpy as np 
import os 
import pandas as pd


SEED = 0

if __name__ == "__main__":

    path = './data'
    Xtr = np.array(pd.read_csv(os.path.join(path,'Xtr.csv'),header=None,sep=',',usecols=range(3072)))
    Xte = np.array(pd.read_csv(os.path.join(path,'Xte.csv'),header=None,sep=',',usecols=range(3072)))
    Ytr = np.array(pd.read_csv(os.path.join(path,'Ytr.csv'),sep=',',usecols=[1])).squeeze()
    
    # Extract the features: brightness histogram
    Xtr_hist = compute_brightness_hist(Xtr, n_bins=256)
    Xte_hist = compute_brightness_hist(Xte, n_bins=256)

    # Extract the features: HOG
    Xtr_HOG = np.apply_along_axis(lambda x: compute_hog_features(x,n_bins=9, ppc=(8,8), cpb=(1,1)), 1, Xtr)
    Xte_HOG = np.apply_along_axis(lambda x: compute_hog_features(x,n_bins=9, ppc=(8,8), cpb=(1,1)), 1, Xte)

    # Final features: concatenation of brightness histogram and HOG features
    print(Xtr_HOG.shape, Xtr_hist.shape)
    Xtr_processed = np.concatenate([Xtr_HOG,Xtr_hist], axis=1)
    Xte_processed = np.concatenate([Xte_HOG,Xte_hist], axis=1)

    # Perform train/validation split
    X_train, X_val, y_train, y_val = train_test_split(Xtr_processed, Ytr, test_size=0.2, random_state=SEED, shuffle=True)

    # Get all the K(K-1)/2 pairs 
    n_classes = len(np.unique(Ytr))
    pairs = []
    for c1 in range(n_classes):
        for c2 in range(c1 + 1, n_classes):
            pairs.append((c1,c2))
    
    # Compute the  K(K-1)/2 classifiers: One vs One (OVO) approach
    sigma = 1.3
    C=100
    models = []
    kernel = RBF(sigma).kernel

    for pair in tqdm(pairs):
        pos_class = pair[0]
        neg_class = pair[1]
        classes = [pos_class,neg_class]
        idx_classes = [i for i in range(len(y_train)) if y_train[i] in classes]
        X, y = X_train[idx_classes], y_train[idx_classes]
        y = labeling(pos_class, y)
        model = KernelSVC(C=C, kernel=kernel)
        model.fit(X, y)
        models.append(model)
        

    # Make predictions on the validation set
    predictions = make_predictions(X_val, models, pairs)
    preds = np.vstack(predictions)
    preds_val = np.apply_along_axis(find_most_voted_class, 0, preds)
    acc = (y_val == preds_val).sum() / len(preds_val)

    print('Accuracy on the validation set: ', acc) 

    # Make predictions on the test set 
    predictions_test = make_predictions(Xte_processed, models, pairs)
    preds_test = np.vstack(predictions_test)
    preds_test = np.apply_along_axis(find_most_voted_class, 0, preds_test)
    Yte = {'Prediction' : preds_test}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv('Yte.csv',index_label='Id') 