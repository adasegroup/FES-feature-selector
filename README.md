# FES-feature-selector

## Overview

This repository implements a framework for evaluating different feature selection methods. We provide a baseline **permutation importance** [2] method in our pipeline (using implementation from sklearn) as well as implementations of two more recent methods:  **Normalised Iterative Hard Thresholding** [1] and **Simultaneous Feature and Feature Group Selection through Hard Thresholding** [3]. The pipeline allows to perform evaluation on both synthetic and real data.

## Baseline method

### Permutation importance

Permutation importance is an approach to compute feature importances for anyblack-box estimator by measuring the decrease of a score when a feature is notavailable.  Permutation importance is calculated after a model has been fitted (i.e., the estimator is required to be a fitted estimator compatible with scorer).

This method contains the following steps:

* A baseline metric, defined by scoring, is evaluated on a data set defined by the X, where X can be the data set used to train the estimator or ahold-out set;
* A feature column from the validation set is permuted and the metric is evaluated again;
* The difference between the baseline metric and metric from permutating the feature column is the permutation importance. 

Parameters can be described as follows [2]: 

* estimator - Fitted estimator compatible with scorer
* X - Data on which permutation importance is compute
* y - Targets (in case of supervised) or Non (in case of unsupervised)
* scoring - Scorer
* nrepeats - Number of feature permutation times
* njobs - Number of jobs for parallel run. Calculation is conducted by computing permutation score for each column, parallelized over the columns.
* randomstate - Pseudo-random number generator to control the permutations of each feature
* sampleweight - Sample weights for scoring.

The mean of feature importance shows the degree of model performance accuracy deterioration with a random shuffling, and the standard deviation shows the variation of performance from one reshuffling to the next. Cases of negative values for permutation importances are possible, especially among small datasets; however, they merely depict the insufficiency of dataset, as they point out that "noisy" data happened to be more accurate than the real data, i.e. there is some random chance distortion. 

This method is most appropriate for computing feature importances with a reasonably limited number of columns (features), as it can be resource-intensive.

## Implemented methods

### Normalised Iterative Hard Thresholding

The algorithm searches for ![\Large \hat{x}^{n+1}](https://latex.codecogs.com/gif.latex?\dpi{120}&space;\hat{x}^{n&plus;1}) starting from ![\Large \hat{x}^0 = 0](https://latex.codecogs.com/gif.latex?\dpi{120}&space;\hat{x}^0&space;=&space;0) by applying the iterative procedure below where ![\Large H_K()](https://latex.codecogs.com/gif.latex?\dpi{120}&space;H_K()) is a non-linear operator that sets all but the top-k largest elements by their magnitude to zero and ![\Large \mu](https://latex.codecogs.com/gif.latex?\dpi{120}&space;\mu) is computed adaptively as described in [1].

![\Large \hat{x}^{n+1} = H_K(\hat{x}^n + \mu A^\top (y - A \hat{x}^n))](https://latex.codecogs.com/gif.latex?\dpi{150}&space;\hat{x}^{n&plus;1}&space;=&space;H_K(\hat{x}^n&space;&plus;&space;\mu&space;A^\top&space;(y&space;-&space;A&space;\hat{x}^n)))

Pros:
* Gives theoretical guarantees on convergence to a local minimum of the cost function
* Doesn't depend on the scaling of the design matrix ![\Large A](https://latex.codecogs.com/gif.latex?A)

Cons:
* Theoretical guarantees on convergence exist only if ![\Large A](https://latex.codecogs.com/gif.latex?A) that satisfies restricted isometry property 

### Simultaneous Feature and Feature Group Selection through Hard Thresholding

The algorithm employs the iterative procedure below where ![\Large f](https://latex.codecogs.com/gif.latex?f) is the objective loss function, ![\Large L](https://latex.codecogs.com/gif.latex?L) is found by line search and ![\Large SGHT()](https://latex.codecogs.com/gif.latex?SGHT()) stands for Sparse Group Hard Thresholding that is solved by dynamic programming as described in [3].

![\Large \hat{x}^{n+1} = SGHT(\hat{x}^n - \frac{1}{L} \nabla f(\hat{x}^n))](https://latex.codecogs.com/gif.latex?\dpi{150}&space;\hat{x}^{n&plus;1}&space;=&space;SGHT(\hat{x}^n&space;-&space;\frac{1}{L}&space;\nabla&space;f(\hat{x}^n)))

Pros:
* Line search on each iteration can be significantly sped up
* Gives theoretical guarantees on convergence to a local minimum of the cost function that is at least within ![\Large c||y-Ax^*||_2](https://latex.codecogs.com/gif.latex?c||y-Ax^*||_2) radius from globally optimal solution ![\Large x^*](https://latex.codecogs.com/gif.latex?x^*) for a certain constant ![\Large c](https://latex.codecogs.com/gif.latex?c)


Cons:
* Theoretical guarantees on convergence exist only if ![\Large A](https://latex.codecogs.com/gif.latex?A) that satisfies restricted isometry property

## Evaluation protocol

### Metrics

### Synthetic data

Synthetic data used for examination of the proposed methods consists of both bi-level variable selection and group selection. The data set generation is conducted3
via the linear modely=Ax+, where the design matrixA∈R100×200and thenoise termfollow a normal distribution.  Ground truthxis partitioned into20 equally sized groups.  In this research, we intend to consider several kinds ofgrouping structure.  The goal is to obtain an accurate (in terms of least squares)estimator ofxthat preserves the grouping structure, given onlyAandy

### Real data

We intend to study the algorithms on the Boston Housing data set.  The originaldata set is used as a regression task, containing 506 samples with 13 features.Up to third-degree polynomial expansion is applied on each feature to accountfor the non-linear relationship between variables and response.  For each variablex,x2andx3are recorded and gathered in a group.  As a next step we split thedata into the training set (approximately 50%) and testing set.  The parametersettings for each method are properly scaled to fit the data set.  We intend to fita linear regression model on the training data and report the number of selectedfeatures, feature groups as well as the mean squared error (MSE) on the testingset

## Team members roles

## References

[1]  T. Blumensath and M. E. Davies.  Normalized iterative hard thresholding:Guaranteed stability and performance.IEEE Journal of Selected Topics inSignal Processing, 4(2):298–309, 2010.

[2]  Leo Breiman.  Random forests.Machine Learning, 45(1):5–32, 2001.

[3]  Shuo Xiang, Tao Yang, and Jieping Ye.  Simultaneous feature and featuregroup selection through hard thresholding. InProceedings of the 20th ACMSIGKDD International Conference on Knowledge Discovery and Data Min-ing,  KDD  ’14,  page  532–541,  New  York,  NY,  USA,  2014.  Association  forComputing Machinery.
