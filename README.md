# FES-feature-selector

## Overview

This repository implements a framework for evaluating different feature selection methods. We provide a baseline **permutation importance** [2] method in our pipeline (using implementation from sklearn) as well as implementations of two more recent methods:  **Normalised Iterative Hard Thresholding** [1] and **Simultaneous Feature and Feature Group Selection through Hard Thresholding** [3]. The pipeline allows to perform evaluation on both synthetic and real data.

## Baseline method

### Permutation importance

## Implemented methods

### Normalised Iterative Hard Thresholding

The algorithm searches for ![\Large \hat{x}^{n+1}](https://latex.codecogs.com/gif.latex?\dpi{120}&space;\hat{x}^{n&plus;1}) starting from ![\Large \hat{x}^0 = 0](https://latex.codecogs.com/gif.latex?\dpi{120}&space;\hat{x}^0&space;=&space;0) by applying the iterative procedure below where ![\Large H_K()](https://latex.codecogs.com/gif.latex?\dpi{120}&space;H_K()) is a non-linear operator that sets all but the top-k largest elements by their magnitude to zero and ![\Large \mu](https://latex.codecogs.com/gif.latex?\dpi{120}&space;\mu) is computed adaptively as described in [1].

![\Large \hat{x}^{n+1} = H_K(\hat{x}^n + \mu A^\top (y - A \hat{x}^n))](https://latex.codecogs.com/gif.latex?\dpi{300}&space;\hat{x}^{n&plus;1}&space;=&space;H_K(\hat{x}^n&space;&plus;&space;\mu&space;A^\top&space;(y&space;-&space;A&space;\hat{x}^n)))

### Simultaneous Feature and Feature Group Selection through Hard Thresholding

## Evaluation protocol

### Metrics

### Synthetic data

### Real data

## References

[1]  T. Blumensath and M. E. Davies.  Normalized iterative hard thresholding:Guaranteed stability and performance.IEEE Journal of Selected Topics inSignal Processing, 4(2):298–309, 2010.

[2]  Leo Breiman.  Random forests.Machine Learning, 45(1):5–32, 2001.

[3]  Shuo Xiang, Tao Yang, and Jieping Ye.  Simultaneous feature and featuregroup selection through hard thresholding. InProceedings of the 20th ACMSIGKDD International Conference on Knowledge Discovery and Data Min-ing,  KDD  ’14,  page  532–541,  New  York,  NY,  USA,  2014.  Association  forComputing Machinery.
