# Kernel Methods for Image Classification - MVA Kaggle Challenge



## Data & task

The dataset used was a subset of 7000 images
from the Cifar-10 dataset. Each image
belonged to one of 10 different classes. 5000 images
with revealed labels were made available for
training. 2000 images with hidden labels were
used to compute the classification accuracy for
the Kaggle challenge. All images were whitened.


## Usage

To run our final script:

```
git clone https://github.com/paulinesert/KernelChallengeMVA
```

Then, install the required libraries:

```
pip install -r requirements.txt
```

#### ...

The following command will run our experiment that yield the best results:
Using HOG and the brightness' histogram features, using a One-vs-One classification approach with  Support-Vector Classifiers (SVC) and a Radial-Basis function (RBF) kernel.

```
python main.py
```
