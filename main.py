from utils import labeling, plot_image, compute_rgb_hist, hog_features
from kernels import RBF
from SVC import KernelSVC

import numpy as np 
import pandas as pd


if __name__ == "__main__":
    path = './data'
    Xtr = np.array(pd.read_csv(os.path.join(path,'Xtr.csv'),header=None,sep=',',usecols=range(3072)))
    Xte = np.array(pd.read_csv(os.path.join(path,'Xte.csv'),header=None,sep=',',usecols=range(3072)))
    Ytr = np.array(pd.read_csv(os.path.join(path,'Ytr.csv'),sep=',',usecols=[1])).squeeze()
    