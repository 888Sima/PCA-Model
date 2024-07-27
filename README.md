Part One: Analysis on the Iris Dataset

Purpose: To apply PCA for dimensionality reduction on the Iris dataset and visualize the results in a two-dimensional scatter plot.

Dependencies: This project uses numpy, pandas, scikit-learn, matplotlib, and seaborn.

The Iris dataset is a well-known dataset in the field of machine learning, containing 150 samples of iris flowers, each with four features:
Sepal length
Sepal width
Petal length
Petal width

The dataset is categorized into three species:
Setosa
Versicolor
Virginica

Standardize the Data: The features are standardized to have a mean of 0 and a variance of 1.

Visualize the Results: The results are visualized using a scatter plot that displays the data points in the PCA-reduced space.


Part Two: PCA Implementation from Scratch

Objective: Implement PCA from scratch to understand its underlying principles and visualize data in a reduced-dimensional space.

Dependencies: This project uses numpy for numerical operations and matplotlib for plotting. It also uses scikit-learn to load the example dataset.
Prerequisites

Load the Dataset: The example uses the Iris dataset, which is loaded using scikit-learn.

Standardize the Data: The dataset is standardized to have zero mean and unit variance.

Compute Covariance Matrix: Calculate the covariance matrix of the standardized data.

Eigenvalue and Eigenvector Computation: Compute eigenvalues and eigenvectors from the covariance matrix.

Sort Eigenvalues and Eigenvectors: Sort eigenvalues and corresponding eigenvectors in descending order.

Project Data: Project the data onto the principal components to reduce its dimensionality.

Visualization: Plot the data in the PCA-reduced space to visualize the distribution.
