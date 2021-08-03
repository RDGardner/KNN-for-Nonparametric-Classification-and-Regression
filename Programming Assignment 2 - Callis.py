#!/usr/bin/env python
# coding: utf-8

# # Programming Project 2 
# ## 605.649 Introduction to Machine Learning
# ## Ricca Callis

# ## Directions
# 
# 
# The purpose of this assignment is to give you a chance to get some hands-on experience implementing
# a nonparametric classification or regression algorithm. Specifically, you will be implementing a k-nearest
# neighbor classifier and regressor. Be careful with how the attributes are handled. Nearest neighbor methods
# work best with numeric attributes, so some care will need to be taken to handle categorical attributes.
# In this project, and all future projects, the experimental design we will use is called 5-fold cross-validation.
# The basic idea is that you are going to take your data set and divide it into five equally-sized subsets. When
# you do this, you should select the data points randomly, but with a twist. Ultimately, you would like the
# same number of examples to be in each class in each of the five partitions. This is called “stratified” crossvalidation.
# 
# For example, if you have a data set of 100 points where 1/3 of the data is in one class and 2/3
# of the data is in another class, you will create five partitions of 20 examples each. Then for each of these
# partitions, 1/3 of the examples (around 6 or 7 points) should be from the one class, and the remaining points
# should be in the other class.
# 
# With five-fold cross-validation, you will run five experiments where you train on four of the partitions
# (so 80% of the data) and test on the remaining partition (20% of the data). You will rotate through the
# partitions so that each one serves as a test set exactly once. Then you will average the performance on these
# five test-set partitions when you report the results.
# For this assignment, you will use four datasets (two classification and two regression) that you will
# download from the UCI Machine Learning Repository, namely:
# 
# 1. Ecoli — https://archive.ics.uci.edu/ml/datasets/Ecoli
# [Classification] A data set to classify localization sites of proteins in ecoli cells. Three of the classes
# have a very small number of examples. These should be deleted from the data set.
# 
# 2. Image Segmentation — https://archive.ics.uci.edu/ml/datasets/Image+Segmentation
# [Classification] The instances were drawn randomly from a database of 7 outdoor images. The images
# were handsegmented to create a classification for every pixel.
# 
# 3. Computer Hardware — https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
# [Regression] The estimated relative performance values were estimated by the authors using a linear
# regression method. The gives you a chance to see how well you can replicate the results with these two
# models.
# 
# 4. Forest Fires — https://archive.ics.uci.edu/ml/datasets/Forest+Fires
# [Regression] This is a difficult regression task, where the aim is to predict the burned area of forest
# fires, in the northeast region of Portugal, by using meteorological and other data .
# 

# ### For this project, the following steps are required:
# 
#  Download the five (5) data sets from the UCI Machine Learning repository. You can find this repository
# at http://archive.ics.uci.edu/ml/. All of the specific URLs are also provided above.
# 
# 
#  Implement k-nearest neighbor and be prepared to find the best k value for your experiments. You
# must tune k and explain in your report how you did the tuning
# 
#  Implement edited k-nearest neighbor. See above with respect to tuning k. Note that you will not apply
# the edited nearest neighbor to the regression problems.
# 
#  Implement condensed k-nearest neighbor. See above with respect to tuning k. Note that you will not
# apply the condensed nearest neighbor to the regression problems.
# 
#  Run your algorithms on each of the data sets. These runs should be done with 5-fold cross-validation
# so you can compare your results statistically. You can use classification error or mean squared error
# (as appropriate) for your loss function.
# 
#  Write a very brief paper that incorporates the following elements, summarizing the results of your
# experiments.
# 
# 1. Title and author name
# 2. A brief, one paragraph abstract summarizing the results of the experiments
# 3. Problem statement, including hypothesis, projecting how you expect each algorithm to perform
# 4. Brief description of algorithms implemented
# 5. Brief description of your experimental approach
# 6. Presentation of the results of your experiments
# 7. A discussion of the behavior of your algorithms, combined with any conclusions you can draw
# 8. Summary
# 9. References (you should have at least one reference related to each of the algorithms implemented, a reference to the data sources, and any other references you consider to be relevant)
# 
#  Submit your fully documented code, the outputs from running your programs, and your paper. Your
# grade will be broken down as follows:
# 
# – Code structure – 10%
# – Code documentation/commenting – 10%
# – Proper functioning of your code, as illustrated by a 5 minute video – 30%
# – Summary paper – 50%

# In[1]:


# Author: Ricca Callis
# EN 605.649 Introduction to Machine Learning
# Programming Project #2
# Date Created: 6/22/2020
# File name: Programming Assignment 2 - Callis.ipynb
# Python Version: 3.7.5
# Jupyter Notebook: 6.0.1
# Description: Implementation of k-nearest neighbor classifier and regressor algorithms 
# using 4 datasets from the UCI Machine Learning Repository

"""
k-Nearest Neighbor (KNN) Algorithm: A lazy supervised learning algorithm for nonparametric data. 
Is used both for classification and regression problems. KNN algorithm stores all available cases 
and classifies new cases based on a similarity measure (e.g., distance functions).  
"""

"""
k-Nearest Neighbor Classifier: When KNN is used for classification, the output can be calculated as 
the class with the highest frequency from the K-most similar instances. Each instance in essence votes 
for their class and the class with the most votes is taken as the prediction.
"""

"""
k-Nearest Neighbor Regressor:When KNN is used for regression problems the prediction is based on the 
mean or the median of the K-most similar instances.

"""

"""
Required Data Sets:
    ecoli.data.csv
    ecoli.names.csv
    forestfires.data.csv
    forestfires.names.csv
    machine.data.csv
    machine.names.csv
    segmentation.data.csv
    segmentation.names.csv
""" 


# In[2]:


from platform import python_version
print ( python_version() )


# In[3]:


# Common standard libraries
import datetime
import time
import os
# Common external libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import sklearn #scikit-learn
import sklearn
from sklearn.model_selection import train_test_split 
import random as py_random
import numpy.random as np_random
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
import scipy.stats as stats
from toolz import pipe # pass info from one process to another (one-way communication)
from typing import Callable, Dict, Union, List
from collections import Counter, OrderedDict
import logging
from itertools import product
import warnings
import io
import requests as r

logging.basicConfig ( filename ='logfile.txt' )
logging.root.setLevel ( logging.INFO )
logger = logging.getLogger ( __name__ )


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Check current directory
currentDirectory = os.getcwd()
print ( currentDirectory )


# In[6]:


# Input data files are available in the ".../input/" directory
# Change the Current working Directory
os.chdir ( '/Users/riccacallis/Desktop/JHU/Data Science/Introduction to Machine Learning/Programming Project 2/input' )

# Get Current working Directory
currentDirectory = os.getcwd()
print ( currentDirectory )


# In[7]:


# List files in input directory
from subprocess import check_output
print ( check_output ( [ "ls", "../input" ] ).decode ( "utf8" ) )


# # Supervised Learning
# 
# Goal: Make a prediction
#    - Estimate (i.e., learn) a function (f) that maps input variables (X) to output variables (Y).
#    
# Where y can be:
#    - Real Number: Regression
#    - Categorical: Classification
#    - Complex Object: Ranking # of items, parse tree, etc.
#    
# Data is labeled:
#    - Data has many pairs { (x, y) }
#        - x: Vector of binary, categorical, real valued features/attributes
#        - y: class ( {+1, -1}, or a real number )
#        
# Method: An algorithm learns this target mapping function from training data.
#    - Create subsets of data:
#        - Training Set
#        - Testing Set
#    - Estimate y = f(x) on X, Y
#    - Assume the same f(x) generalizes (i.e., works on unseen X', Y')
#    
#    
# # Nonparametric
# 
# *Nonparametric Machine Learning Algorithms*: Algorithms that do not make strong assumptions about the form of the mapping function. By not making assumptions, they are free to learn any functional form from the training data.
#     - Thus, there is no assumed distribution
# 
# 
# ## Benefits
# 
# *Flexibility*: Capable of fitting a large number of functional forms.
# 
# *Power*: No assumptions (or weak assumptions) about the underlying function.
# 
# *Performance*: Can result in higher performance models for prediction.
# 
# ## Limitations
# 
# *More data*: Require a lot more training data to estimate the mapping function.
# 
# *Slower*: A lot slower to train as they often have far more parameters to train.
# 
# *Overfitting*: More of a risk to overfit the training data and it is harder to explain why specific predictions are made.
# 
# # Classification vs Regression
# 
# Nonparametric analyses include both classification and regression problems
# 
# **Classification**
# - Has a discrete value as its output.
# - Has predictor (or set of predictors) and a label. 
# - Determines inclusion or exclusion in group (or class) based on a predicting factor 
# - Standard practice to represent the output (label) of a classification algorithm as an integer, where, 0 = exclusion and 1 = inclusion
# - In the data set, each row is typically called an example, observation, or data point
# - In the data set, each column represents either an attribute/feature or a class/label
# 
# **Regression**
# - Has a real number (a number with a decimal point) as its output.
# - Has an independent variable (or set of independent variables) and a dependent variable (the thing we are trying to guess given our independent variables).
# - In the data set, each row is typically called an example, observation, or data point
# - In the data set, each column (not including the label/dependent variable) is often called a predictor, dimension, independent variable, or feature.

# # K-Nearest Neighbors
# 
# 
# **Overview:**
# 
# Is a supervised learning algorithm for nonparametric classification or regression problems.
# 
# The algorithm makes predictions based on the k-most similar training patterns for a new data instance. The method does not assume anything about the form of the mapping function other than patterns that are close are likely to have a similar output variable.
# 
# 
# ## KNN Learning
# 
# Assumes all instances correspond to points in the *n*-dimensional space, $\ A^n$ 
# 
# Predictions are made for a new instance (x) by searching through the entire training set for the k-most similar instances (the neighbors) and summarizing the output variable for those k-instances. For regression this might be the mean output variable, in classification this might be the mode (or most common) class value.
# 
# To determine which of the k-instances in the training dataset are most similar to a new input, a distance measure is used (usually the standard Euclidean distance).
# 
# Euclidean distance is calculated as the square root of the sum of the squared differences between a new point (x) and an existing point (xi) across all input attributes j.
# 
# EuclideanDistance(x, xi) = sqrt( sum( (xj – xij)^2 ) )
# 
# ### 3 Parts to KNN
# 
# This k-Nearest Neighbors tutorial is broken down into 3 parts:
# 
# Step 1: Calculate Euclidean Distance.
# 
# Step 2: Get Nearest Neighbors.
# 
# Step 3: Make Predictions.
# 
# ### KNN for Discrete-Valued Functions: KNN Classification
# 
# ![KNN%20Discrete%20Valued%20Target%20Functions.png](attachment:KNN%20Discrete%20Valued%20Target%20Functions.png)
# 
# ### KNN Classification Pseudo Code
# 
# - Load the data
# - Initialise the value of k
# - To predict target function's class, iterate from 1 to total number of training data points
# - Calculate the Euclidean distance between test data and each row of training data
# - Sort the calculated distances in ascending order based on distance values
# - Get top k rows from the sorted array
# - Get the most frequent class of these rows
# - Return the predicted class

# In[8]:


# Implementation of K-Nearest Neighbors Algorithm
# KNN Classifier

'''''
Class: KNearestNeighbors
    - Supervised Learning Algorithm for nonparametric classification problems 

Functions:
    - __init__: Initializes the KNearestNeighbors algorithm 
    - fit: Fits KNN by saving values of 'X' and 'y'
    - get_distance: Obtains distances between rows of X and x_0 (initial/selected row)
    - get_nearest_neighbors: Given the distances, function obtains the nearest neighbors
    - predict_probability_single_instance: Obtains the count within the k-neighbors for each class
    - predict_probability: Applies the probability obtained from function predict_probability_single_instance 
      and applies it across 1 or many instances
    - predict: Make predictions for the rows in X

'''''

class KNearestNeighbors:
    
    """
    Class to perform KNN algorithm for classification.
        Parameters
            k: Indicates the number of neighbors to take into account in the voting scheme. Type: Integer
            distance_function: A function to calculate the distance between 2 points in a feature space. Defaults to 2. Type: Callable   
    """

    def __init__ (
        # Initialize parameters
        self, # Class instance
        k: int, #  The nearest neighbor we wish to take the vote from
        distance_function: Callable = lambda x, x0: np.sum ( np.subtract ( x, x0 ) ** 2 ),
    ):

        # Initialize class instances
        self.k = k
        self.distance_function = distance_function

    def fit ( self, X: np.ndarray, y: np.ndarray ):
        
        """
        Function fits KNN by saving the values of `X` and `y`.
            Parameters
                X: Indicates the feature matrix. Type: Array (np.ndarray)
                y: Indicates the target vector. Type: Array (np.ndarray)
        """
        
        self.X = X
        self.y = y

    def get_distance ( self, X: np.ndarray, x0: np.ndarray ):
        
        """
        Function obtains distances between the rows of X and x_0.
            Parameters
                X : Indicates a matrix of observations. Type: Array (np.ndarray)
                x0 : Indicates the row to get the distances against the matrix. Type: Array (np.ndarray)
            Returns: Distance from query point to data instance across each row in the feature matrix
        """
        
        # Apply distance_function across rows of X
        return [ self.distance_function ( row, x0 ) for row in X ]
        # End get_distance()

    def get_nearest_neighbors ( self, distances: List [ float ] ):
        
        """
        Function obtains nearest neighbors, given the distances.
            Parameters
                distances: Indicates a list of distances. Type: Float List
            Returns: distance_map (a sorted key-value list mapping instances based on their k & distance)
        """
        
        return pipe (
            # Map index to distance
            dict ( enumerate ( distances ) ),
            # Sort the indices based on their value in the mapping and take the 1st k
            lambda distance_map: sorted ( distance_map, key = distance_map.get ) [: self.k ],
        ) # End get_nearest_neighbors()

    def predict_probability_single_instance ( self, row: np.ndarray ):
        
        """
        Function obtains the count within the k-neighbors for each class
            Parameters
                row : Indicates the row used for prediction. Type: Array (np.ndarray)
            Returns: Number of neighbors in a class
        """
        
        # Get the pairwise distances with X
        distances = self.get_distance ( X = self.X, x0 = row )

        # Get the k-nearest neighbors
        nearest_neighbors = self.get_nearest_neighbors ( distances = distances )

        # For each class, get the number of the neighbors that is in that class
        return {
            cls: np.sum ( self.y [ nearest_neighbors ] == cls )
            # Ensure no tie-breakers
            # Shuffle for coin flip
            # Sorted takes the 1st in ties
            for cls in np.random.permutation ( np.unique ( self.y ))
        } # End predict_probability_single_instance()

    def predict_probability ( self, X: np.ndarray ):
        
        """
        Applies the probability obtained from function predict_probability_single_instance and applies it 
        across 1 or many instances
            Parameters:
                X : Indicates the matrix to make predictions for. Type: Array (np.ndarray)
            Returns: Predicted probability of class assignment for each instance in the selected feature matrix.
        """
        
        # If 1 instance
        if X.ndim == 1:
            return [ self.predict_probability_single_instance ( X ) ]
        # Else many instances
        return [ self.predict_probability_single_instance ( row ) for row in X ]
        # End predict_probability()

    def predict ( self, X: np.ndarray ):
        
        """
        Make predictions for the rows in X
            Parameters:
                X : Indicates the matrix to make predictions for. Type: Array (np.ndarray)
            Returns: A list (key:map) of the largest probabilities for each row in X and across all classes
        """
        
        # Get the largest count across all the classes for each row in X
        return list (
            map ( lambda probs: max ( probs, key = probs.get ), self.predict_probability ( X = X ) )
        ) # End predict()
        
    # End class KNearestNeighbors

# Helper Function for KNN Classification Problems
# Standarize Input Data for KNN Classification

'''''
Class: Standardizer
    - Because distance measures are strongly affected by the scale of the input data (i.e., KNN performs much better if 
    all of the data has the same scale), this class standardizes input features.

Functions:
    - __init__: Initializes the Standardizer 
    - fit: Calculates columnwise mean and standard deviation
    - transform: Applies the columnwise mean and std transformations
    - fit_transform: Fits on the columns and then transforms X

'''''

class Standardizer:
    
    """
    Class to standardize input features.
    """

    def __init__ ( self, mean = True, std = True ):
        # Initialize parameters
        # Class instances
        self.mean = mean
        self.std = std

    def fit ( self, X ):
        
        """
        Calculates the columnwise mean and standard deviation
            Parameters:
                X: Indicates the matrix to make predictions for. Type: Array (np.ndarray)
        """
        
        if self.mean:
            self.df_means = X.mean ( axis = 0 )  # Get the colwise means
        if self.std:
            self.df_std = X.std ( axis = 0 )  # Get the colwise stds

    def transform ( self, X ):
        
        """
        Applies the columnwise mean and std transformations
            Parameters:
                X: Indicates the matrix to make predictions for. Type: Array (np.ndarray)
            Returns: df_xf a dataframe of transformed x's
        """
        
        if not self.mean and not self.std:
            return X
        if self.mean:
            df_xf = X - self.df_means  # Subtract means
        if self.std:
            is_zero = np.isclose ( self.df_std, 0 )  # If non-zero variance,
            with warnings.catch_warnings():
                warnings.simplefilter ( "ignore" )
                df_xf = np.where (
                    is_zero, X, X / self.df_std
                )  # Ensure no divide by zero issues

        return df_xf
        # End transform()

    def fit_transform ( self, X ):
        
        """
        Fits on the columns and then transforms X
            Parameters:
                X: Indicates the matrix to make predictions for. Type: Array (np.ndarray)
        """
        self.fit ( X )
        return self.transform ( X )
        # End fit_transform()
        
    #End class Standardizer


# ### KNN for Continuous-Valued Functions: KNN Regression
# 
# ![KNN%20Continuous%20Valued%20Target%20Functions.png](attachment:KNN%20Continuous%20Valued%20Target%20Functions.png)

# In[9]:


# Implementation of K-Nearest Neighbors Algorithm
# KNN Regression

'''''
Class: KNearestNeighborRegression
    - Supervised Learning Algorithm for nonparametric regression problems 

Parameter:
    - KNearestNeighbors
    
Functions:
    - __init__: Initializes the KNearestNeighborRegression algorithm 
    - predict_probability_single_instance: Obtains the mean target value within the k-neighbors
    - predict_probability: Applies the probability obtained from function predict_probability_single_instance 
      and applies it across 1 or many instances
    - predict: Make predictions for the rows in X

'''''

class KNearestNeighborRegression ( KNearestNeighbors ):
    
    """
    Class to perform KNN algorithm for classification.
        Parameters
            KNearestNeighbors: Class to perform KNN classification
            k: Indicates the number of neighbors to take into account in the voting scheme. Type: Integer
            distance_function: A function to calculate the distance between 2 points in a feature space. Defaults to 2. Type: Callable   
    """

    def __init__ (
        # Initialize parameters
        self, # Class instance
        k: int,  # The nearest neighbor we wish to take the vote from
        distance_function: Callable = lambda x, x0: np.sum ( np.subtract ( x, x0 ) ** 2 ),
    ):

        super().__init__ ( k = k, distance_function = distance_function )

    def predict_probability_single_instance ( self, row: np.ndarray ):
        
        """
        Function obtains the mean target value within the k-neighbors
            Parameters
                row : Indicates the row to do to predictions for. Type: Array (np.ndarray)
            Returns: The mean of the target vector's nearest neighbor
        """
        
        # Get distances
        distances = self.get_distance ( X = self.X, x0 = row )
        # Get neighbors
        nearest_neighbors = self.get_nearest_neighbors ( distances = distances )
        # Get mean of y of neighbors
        return np.mean ( self.y [ nearest_neighbors ] )
        # End predict_probability_single_instance

    def predict ( self, X: np.ndarray ):
        
        """
        Function make predictions for the rows in X
            Parameters:
                X : Indicates the matrix to make predictions for. Type: Array (np.ndarray)
            Returns: Predicted probability of class assignment for each instance in the selected feature matrix. 
        """
        
        return self.predict_probability ( X = X )
        # End predict()
    
    # End class KNearestNeighborRegression 


# ## Condensed KNN
# 
# Condensed Nearest Neighbors algorithm helps to reduce the dataset X for kNN classification. It constructs a subset of examples which are able to correctly classify the original data set using k=1.
# 
# Algorithm returns a subset Z of the data set X, NOT the array of misclassified points.
# 
# Similar to a forward stepwise process
# 
# Advantage: Decreases execution time, reduces space complexity
# 
# ### 3 Steps to Condensed KNN
# 
# Step 1: Scan all elements of X, looking for an element x whose nearest neighbor from Z has a different label than x
# 
# Step 2: Remove x from X and add it to Z
# 
# Step 3: Repeat the scan until no more instances are added to Z

# In[10]:


# Improve Cost of KNN
# Condensed KNN

'''''
Class: CondensedKNN
    - A more cost-efficient KNN Algorithm which reduces the number of comparisons needed in order to classify a 
      new observation. It constructs a subset of examples which are able to correctly classify the original data 
      set using a 1-NN algorithm (k = 1).

Functions:
    - __init__: Initializes the CondensedKNN algorithm 
    - get_inclusion: Determines whether a specified row (called 'row_index') should be included in the condensed set/subset set Z. 
    - fit: Fits the Condensed KNN
    - predict: Make predictions for the rows in X

'''''

class CondensedKNN:
    
    """
    Class to perform Condensed KNN
        Parameters:
            verbose: Boolean conditional. If true, will print intermediate results. Type: Boolean
            knn : Indicates the underlying KNN object that will be used for making predictions.
    """

    def __init__(
        # Initialize parameters
        self, verbose: bool = False, knn: KNearestNeighbors = KNearestNeighbors ( k = 1 )
    ):

        # Class instances
        self.knn = knn
        # Make sure k = 1 always
        self.knn.k = 1  
        self.verbose = verbose

    def get_inclusion ( self, X: np.ndarray, y: np.ndarray, row_index: int, subset_Z: List [ int ] ):
        
        """
        Determines whether a specified row (called 'row_index') should be included in the condensed set.
            Parameters:
                X: Indicates the feature matrix for training. Type: Array (np.ndarray)
                y: Indicates the labels for training. Type: Array (np.ndarray)
                row_index: Indicates the proposed row of X to include in subset_Z. Type: Integer
                subset_Z: Indicates the list of rows that have already been included; Subset Z of data set X. Type: Integer List
            Returns:
                True if row is included in subset Z (i.e., row is added to subset Z)
                False otherwise (i.e., row was not added to subset Z)
        """
        
        # Fit the KNN on just the rows of Z. 
        # Remember that k = 1.
        self.knn.fit ( X [ subset_Z ], y [ subset_Z ] )

        # If the prediction is incorrect
        if self.knn.predict ( X [ row_index ] ) != y [ row_index ]:
            
            # Add row_index to subset Z
            subset_Z.append ( row_index )

            # Return True (change to subset Z has occured; row_index was added to subset Z)
            return True

        return False
        # End get_inclusion()

    def fit ( self, X: np.ndarray, y: np.ndarray ):
        
        """
        Function to run the condensed KNN fitting procedure.
            Parameters:
                X: Indicates the feature maxtrix. Type: Array (np.ndarray)
                y: Indicates the target vector. Type: Array (np.ndarray)
        """
        
        # Track changes to subset Z
        is_changed = True

        # Z begins as empty set
        subset_Z = []
        while is_changed:
            is_changed = False
            # If subset Z is empty, add the 1st element of feature matrix X
            if not subset_Z:
                subset_Z.append ( 0 )

            # Iterate through rows of X
            for row_index in range ( 1, len ( X ) ):
                # If the row is not already in Z
                if row_index not in subset_Z:
                    # Run inclusion procedure
                    changed = self.get_inclusion ( X, y, row_index, subset_Z)
                    # Track changes to subset Z
                    is_changed = True if changed else is_changed
        # Fit model over rows of subset Z
        self.knn.fit ( X [ subset_Z ], y [ subset_Z ] )
        # End fit()

    def predict ( self, X: np.ndarray ):
        
        """
        Function make predictions for the rows in X
            Parameters:
                X: Indicates the matrix to make predictions for. Type: Array (np.ndarray)
        """
        
        return self.knn.predict ( X )
        # End predict()
    # End class CondensedKNN


# ## Edited KNN

# In[11]:


# Reduce Error Rate & Computational Cost
# Edited KNN

'''''
Class: EditedKNN
    - The edited k-nearest neighbor consists of the application of the k-nearest neighbor classifier with an edited 
    training set, in order to reduce the classification error rate. This edited training set is a subset of the 
    complete training set in which some of the training patterns are excluded.

Functions:
    - __init__: Initializes the EditedKNN algorithm 
    - get_exclusion: Determines whether a specified row (called 'row_index') should be excluded from the condensed set/subset set Z. 
    - validation_error_decreasing: Obtains validation scores
    - fit: Fits the Condensed KNN
    - predict: Make predictions for the rows in X

'''''

class EditedKNN:
    
    """
    Performs the EditedKNN algorithm for classification.
        Parameters
            k: Indicates the number of neighbors to consider in the voting scheme. Type: Integer 
            proportion_cv: Indicates the proportion of the training set that should be used for validation (i.e., when to stop training)
            verbose: Boolean conditional. If true, will log intermediate steps. Type: Boolean. 
    """

    def __init__ ( self, k, proportion_cv = 0.1, verbose = False ):

        # Initialize parameters
        # Class instances
        self.knn = KNearestNeighbors ( k = k )
        self.k = k
        self.proportion_cv = proportion_cv
        self.verbose = verbose

    def get_exclusion ( self, X, y, row_index, subset_Z ):
        
        """
        Function to determine if the specified row (row_index) of X should be excluded from the final training set
            Parameters
                X: Indicates the feature matrix. Type: Array (np.ndarray)
                y: Indicates the target vector. Type: Array (np.ndarray)
                row_index: Indicates the row of X under consideration. Type: Integer
                subset_Z: Indicates the list of rows that have already been included; Subset Z of data set X. Type: Integer List
            Returns
                False if row_index remains in subset Z (i.e., was NOT removed)
                True otherwise (i.e., row_index was removed from subset Z)
        """

        # Remove row_index from subset_Z
        subset_Z.remove ( row_index )
        # Fit the model
        self.knn.fit ( X [ subset_Z ], y [ subset_Z ] )
        # If classification is correct, put row_index back in subset_Z (i.e., don't drop it from the training set)
        # If actual & predicted are the same, do nothing
        # If actual != predicted, drop it
        if y [ row_index ] == self.knn.predict ( X [ row_index ] ):
            subset_Z.append ( row_index )
            return False
        return True
        # End get_exclusion()

    def validation_error_decreasing ( self, X_validation, y_validation, last_validation_score ):
        
        """
        Function to obtain validation scores
            Parameters
                X_validation: Indicates the feature matrix for the validation set. Type: Array (np.ndarray)
                y_validation: Indicates the target vector for the validation set. Type: Array (np.ndarray)
                last_validation_score: Indicates the previous validation score. Type: Float
            Returns: The mean error in the validation set and identifies whether error is less than the previous 
            validation score (indicating decreasing error)
        """
        
        error = np.mean ( np.array ( self.knn.predict ( X_validation ) ) != np.array ( y_validation ) )
        return error < last_validation_score, error
        # End validation_error_decreasing()

    def fit ( self, X, y ):
        
        """
        Function to run the edited fitting procedure.
            Parameters
                X: Indicates the feature matrix. Type: Array (np.ndarray)
                y: Indicates the target vector. Type: Array (np.ndarray)
        """
        
        # Split off subset for validation
        n_holdout = int ( len ( X ) * self.proportion_cv )
        X_validate = X [ :n_holdout ]
        y_validate = y [ :n_holdout ]
        X_train = X [ n_holdout :]
        y_train = y [ n_holdout: ]

        # Starting edited set with all indices in it.
        subset_Z = list ( range ( len ( X_train ) ) )

        # Tracking validation scores
        validation_decreasing = True
        last_validation_score = np.inf

        # Tracking changes to the edited set
        is_changed = True

        # While changes to edit set and validation scores decreasing...
        while is_changed and validation_decreasing:
            is_changed = False

            # For each row in X
            for row_index in range ( len ( X_train ) ):
                # Only indices that are still in Z can be eliminated
                if row_index in subset_Z:
                    # Run the exclusion
                    changed = self.get_exclusion ( X_train, y_train, row_index, subset_Z )
                    # Track if changes are made
                    is_changed = True if changed else is_changed

            # Fit the model on the edited set and get validation scores
            self.knn.fit ( X [ subset_Z ], y [ subset_Z ] )
            (
                validation_decreasing,
                last_validation_score,
            ) = self.validation_error_decreasing (
                X_validation = X_validate,
                y_validation = y_validate,
                last_validation_score = last_validation_score,
            )
            #End fit()

    def predict ( self, X ):
        
        """
        Make predictions for the rows in X
            Parameters:
                X: Indicates the matrix to make predictions for. Type: Array (np.ndarray)
        """
        
        return self.knn.predict ( X )
        # End predict()
    # End class EditedKNN


# # Model Evaluation
# 
# Loss functions are used by algorithms to learn the classification models from the data.
# 
# Classification metrics, however, evaluate the classification models themselves. 
# 
# For a binary classification task, where "1" is taken to mean "positive" or "in the class" and "0" is taken to be "negative" or "not in the class", the cases are:
# 
# 1. The true class can be "1" and the model can predict "1". This is a *true positive* or TP.
# 2. The true class can be "1" and the model can predict "0". This is a *false negative* or FN.
# 3. The true class can be "0" and the model can predict "1". This is a *false positive* or FP.
# 4. The true class can be "0" and the model can predict "0". This is a *true negative* or TN.
# 

# ## Training Learners with Cross-Validation
# 
# Fundamental assumption of machine learning:The data that you train your model on must come from the same distribution as the data you hope to apply the model to.
# 
# Cross validation is the process of training learners using one set of data and testing it using a different set.
# 
# Options:
#     - Divide your data into two sets:
#         1. The training set which you use to build the model
#         2. The test(ing) set which you use to evaluate the model. 
#     - kfolds: Yields multiple estimates of evaluation metric
# 
#     
# ### k-fold Cross-Validation
# 
# Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.
# 
# The procedure has a single parameter called k that refers to the number of groups (or folds) that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation. When a specific value for k is chosen, it may be used in place of k in the reference to the model, such as k=5 becoming 5-fold cross-validation.
# 
# 
# The general procedure is as follows:
# - Shuffle the dataset randomly.
# - Split the dataset into k groups (or folds)
# - Save first fold as the validation set & fit the method on the remaining k-1 folds
# - For each unique group:
#     - Take the group as a hold out or test data set
#     - Take the remaining groups as a training data set
# - Fit a model on the training set and evaluate it on the test set
# - Retain the evaluation score and discard the model
# - Summarize the skill of the model using the sample of model evaluation scores
#     - The average of your k recorded errors is called the cross-validation error and will serve as your performance metric for the model
# 
# Importantly, each observation in the data sample is assigned to an individual group and stays in that group for the duration of the procedure. This means that each sample is given the opportunity to be used in the hold out set 1 time and used to train the model k-1 times.
# 
# Below is the visualization of a k-fold validation when k=10.
# 
# Looks like:
# 
# |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# | **Test** | Train | Train | Train | Train | Train | Train | Train | Train | Train |
# 
# |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# | Train| **Test** | Train | Train | Train | Train | Train | Train | Train | Train |
# 
# |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# | Train| Train | **Test** | Train | Train | Train | Train | Train | Train | Train |
# 
# And finally:
# 
# |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |
# |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
# | Train| Train | Train | Train | Train | Train | Train | Train | Train | **Test** |
# 
# ### Stratified k-fold Cross-Validation
# Stratification is the process of rearranging the data so as to ensure that each fold is a good representative of the whole. For example, in a binary classification problem where each class comprises of 50% of the data, it is best to arrange the data such that in every fold, each class comprises of about half the instances.
# 
# For classification problems, one typically uses stratified k-fold cross-validation, in which the folds are selected so that each fold contains roughly the same proportions of class labels.
# 

# In[12]:


# Model Evaluation

# Teaching Learners with Cross-Validation
# k-Folds

'''''
Class: KFoldCV
    - Class to handle K-fold Cross-Validation

Functions:
    - __init__: Initializes the EditedKNN algorithm 
    - get_indices: Obtains indices of length of rows in feature matrix X
    - get_one_split: Given the split indices, obtains one of the splits
    - get_indices_split: Splits the indices by the number of folds
    - split: Creates a generator of train test splits from the feature matrix X
'''''

class KFoldCV:
    
    """
    Class to handle K-Fold Cross-Validation
        Parameters
            number_of_folds : Indicates the number of folds or splits. Type: Integer
            shuffle : If True, rows will be shuffled before the split. Type: Boolean
    """

    def __init__( self, number_of_folds: int, shuffle: bool = True ):
        # Initialize parameters
        # Class instances
        self.number_of_folds = number_of_folds
        self.shuffle = shuffle

    def get_indices ( self, X ):
    
        """
        Function obtains indices of length of rows in feature matrix X
            Parameters
                X: Indicates the matrix to make predictions for. Type: Array (np.ndarray)
            Returns: Shuffled K-Fold Indices matrix (arranged by row)
        """
       
        # Shuffle if `self.shuffle` is true.
        nrows = X.shape [ 0 ]
        return (
            np.random.permutation (
                np.arange ( nrows )
            )  # Shuffle the rows if `self.shuffle`
            if self.shuffle
            else np.arange ( nrows )
        ) # End get_indices()

    def _get_one_split ( split_indices, number_of_split ):
    
        """
        Given the split indices, function obtains one of the training splits
            Parameters
                number_of_folds: Indicates the number of folds or splits. Type: Integer
                split_indices: Indicates array of indices in the training split. Type: Integer
            Returns: number_of_split. Given the split index, obtains the number of split elememnts
        """
    
        # Given the split indices, get the `number_of_split` element of the indices.
        return ( np.delete ( np.concatenate ( split_indices ), split_indices [ number_of_split ] ),  # Drops the test from the train
            split_indices [ number_of_split ],)  # Gets the train
        # End get_one_split

    def _get_indices_split ( indices, number_of_folds ):
    
        """
        Function splits the indices by the number of folds
            Parameters
                indices: Indicates the index of the training/spilt data Type: Integer
                number_of_folds: Indicates the number of folds or splits. Type: Integer
            Returns: array split by indices
        """
        # Split the indicies by the number of folds
        return np.array_split ( indices, indices_or_sections = number_of_folds )
        # End get_indices_split()

    def split ( self, X: np.ndarray, y: np.ndarray = None ):
    
        """
        Function creates a generator of train/test splits from feature matrix X
            Parameters
                X: Indicates the matrix to make predictions for. Type: Array (np.ndarray)
            Returns: All but one split as train data (Type: Array) and one split as test data (Type: Array).
        """
        # Split the indices into `number_of_folds` subarray
        indices = self.get_indices ( X )
        split_indices = KFoldCV._get_indices_split ( indices = indices, number_of_folds = self.number_of_folds )
        for number_of_split in range ( self.number_of_folds ):
            # Return all but one split as train, and one split as test
            yield KFoldCV._get_one_split ( split_indices, number_of_split = number_of_split )
        # End split()
    # End class KFoldCV

'''''
Class: KFoldStratifiedCV
    - Class to conduct Stratified K-Fold Cross Validation. Ensures the splitting of data into folds is governed by 
    criteria such as ensuring that each fold has the same proportion of observations with a given categorical 
    value, such as the class outcome value.

Functions:
    - __init__: Initializes the KFoldStratifiedCV algorithm 
    - add_split_col: Adds new column called "split"
    - split: Takes an array of classes, and creates train/test splits with proportional examples for each group.
'''''

class KFoldStratifiedCV:
    
    """
    Class to conduct Stratified K-Fold Cross Validation.
        Parameters
            number_of_folds: Indicates the number of folds or splits. Type: Integer
            
    """

    def __init__ ( self, number_of_folds, shuffle = True ):
        # Initialize parameters
        # Class Instances
        self.number_of_folds = number_of_folds
        self.shuffle = shuffle

    def add_split_col ( self, arr ):
    
        """
        Function adds new column called "split"
            Parameters
                arr: Indicates an array
            Returns: New column in dataframe with index & split
            
        """
        arr = arr if not self.shuffle else np.random.permutation ( arr )
        n = len ( arr )
        k = int ( np.ceil ( n / self.number_of_folds ) )
        return pd.DataFrame (
            { "index": arr, "split": np.tile ( np.arange ( self.number_of_folds ), k )[ 0:n ] , }
        )

    def split ( self, y, X = None ):
    
        """
        Function takes an array of classes, and creates train/test splits with proportional examples for each group.
            Parameters
                y: Indicates the array of class labels. Type: Array (np.array)
            Returns: Dataframe with index values of not cv split & cv split train and test data
        """
        # Make sure y is an array
        y = np.array ( y ) if isinstance ( y, list ) else y

        # Groupby y and add integer indices.
        df_with_split = (
            pd.DataFrame ( { "y": y, "index": np.arange ( len ( y ) ) } )
            .groupby ( "y" ) [ "index" ]
            .apply ( self.add_split_col )  # Add col for split for instance
        )

        # For each fold, get train and test indices (based on col for split)
        for cv_split in np.arange ( self.number_of_folds - 1, -1, -1 ):
            train_bool = df_with_split [ "split" ] != cv_split
            test_bool = ~ train_bool
            # Yield index values of not cv_split and cv_split for train, test
            yield df_with_split [ "index" ].values [ train_bool.values ], df_with_split [
                "index"
            ].values [ test_bool.values ]
            # End split()
    # End class KFoldStratifiedCV


# ## Parameter Tuning
# 
# Parameter tuning is the process to selecting the values for a model’s parameters that maximize the accuracy of the model.
# 
# 
# A machine learning model has two types of parameters:
# 
#     1. Parameters learned through a machine learning model
#     
#     2. Hyper-parameters passed to the machine learning model
# 
# 
# In KNN algorithm, the hyper-parameter is the specified value of k. 
# 
# Normally we randomly set the value for these hyper parameters and see what parameters result in best performance. However randomly selecting the parameters for the algorithm can be exhaustive.
# 
# 
# ## Grid Search
# 
# Instead of randomly selecting the values of the parameters, GridSearch automatically finds the best parameters for a particular model. Grid Search is one such algorithm.
# 
# Grid Search evaluates all the combinations from a list of desired hyper-parameters and reports which combination has the best accuracy.
# 
# ### Process
# 
# Step 1: Set your hyper-parameters ("param_grid" here).
# 
# Step 2: Fit the model. Use k-fold cross-validation internally on selected hyper-parameters. Store model average & accuracy.
# 
# Step 3: Go back to step 1 changing at least 1 hyper-parameter
# 
# Step 4: Select hyperparameter which gives best performance (highest accuracy)
# 
# Note that the search is not done within each fold. Instead, cross-validation is used to evaluate the performance of the model with the current combination of hyperparameters.

# In[13]:


# Model Evaluation
# Parameter Tuning with Grid Search
            

'''''
Class: GridSearchCV
    - Grid Search evaluates all the combinations from a list of desired hyper-parameters and reports 
    which combination has the best accuracy.

Functions:
    - __init__: Initializes the GridSearchCV algorithm 
    - create_param_grid: Creates a mapping of arguments to values to grid search over.
    - get_single_fitting_iteration: Runs a model fit and validation step.
    - get_cv_scores: Runs the grid search across the parameter grid.
'''''

class GridSearchCV:
    
    """
    Class to assist with grid searching over potential parameter values.
        Parameters:
            model_callable: Function that generates a model object. Should take the keys of param_grid as arguments. Type: Callable
            param_grid: Mapping of arguments to potential values. Type: Dictionary
            scoring_func: Takes in y and yhat and returns a score to be maximized. Type: Callable
            cv_object: A CV object that will be used to make validation splits.
    """

    def __init__(
        # Initialize parameters
        self,
        model_callable: Callable, # Generates model object; takes keys of param_grid as arguments
        param_grid: Dict, # Mapped arguments to potential values
        scoring_func: Callable, # Score to be maximized
        cv_object: Union [ KFoldCV, KFoldStratifiedCV ],
    ):
        # Class instances
        self.model_callable = model_callable
        self.param_grid = param_grid
        self.scoring_func = scoring_func
        self.cv_object = cv_object

    @staticmethod
    def create_param_grid ( param_grid: Dict ):
        
        """
        Function creates a mapping of arguments to values to grid search over.
            Parameters:
                param_grid: Dictionary of key:value map (arguments to potential values). Type: Dictionary {kwarg: [values]}
        """
        
        return (
            dict ( zip ( param_grid.keys(), instance ) )
            for instance in product ( * param_grid.values() )
        ) # End create_param_grid

    def get_single_fitting_iteration ( self, X: np.ndarray, y: np.ndarray, model ):
        
        """
        Function runs a model fit and a validation step.
            Parameters:
                X: Indicates the feature matrix for training. Type: Array (np.ndarray)
                y: Indicates the arget vector for training. Type: Array (np.ndarray)
                model: Indicates model object with a fit and predict method.
            Returns: mean score
        """
        
        scores = []
        # Create train/test splits
        for train, test in self.cv_object.split ( X = X, y = y):
            # Fit the model
            model.fit ( X [ train ], y [ train ] )
            # Get the predictions
            yhat = model.predict ( X [ test ] )
            # Get the scores
            scores.append ( self.scoring_func ( y [ test ], yhat ) )
        # Get the average score.
        return np.mean ( scores )
    # End get_single_fitting_iteration()

    def get_cv_scores ( self, X: np.ndarray, y: np.ndarray ):
        
        """
        Function runs the grid search across the parameter grid.
            Parameters:
                X: Indicates the feature matrix. Type: Array (np.ndarray)
                y: Indicates the target vector. Type: Array (np.ndarray)
        """
        # Create the parameter grid
        param_grid = list ( GridSearchCV.create_param_grid ( self.param_grid ) )

        # Zip the grid to the results from a single fit
        return zip (
            param_grid,
            [
                self.get_single_fitting_iteration (
                    X, y, model = self.model_callable ( ** param_set )
                )
                for param_set in param_grid
            ],
        ) # End get_cv_scores
    # End class GridSearchCV

# Other Helpfer Functions
# Evaluation Metrics: Accuracy of Predictions

def accuracy ( actuals, predictions ):
    
    """
    Function to get classifier accuracy
    """
    return np.mean ( actuals == predictions )
    # End accuracy()

# Other Helpfer Functions
# Evaluation Metrics: MSE

def mean_squared_error ( actuals, predictions ):
    
    """
    Function to get MSE
    """
    return np.mean ( ( actuals - predictions ) ** 2 )
    # End mean_squared_error()

# Other Helpfer Functions
# Choose best value of k

def choose_k (    
    X,
    y,
    model_call,
    param_grid,
    scoring_func = accuracy,
    cv = KFoldStratifiedCV ( number_of_folds = 3 ),
):
        
    """
    Function to use cross-validation to choose a value of k
        Parameters:
            X: Indicates the feature matrix. Type: Array (np.ndarray)
            y: Indicates the target vector. Type: Array (np.ndarray)
            model_call: A function that returns a model object. Its arguments must be the keys in param_grid. Type: Callable
            param_grid: A mapping of arguments to values that we want to try. Type: Dictionary (key:value)
            scoring_func: The function that scores the results of a model. This value is maximized.Type: Callable
            cv: The validation object to use for the cross validation.
        Returns: k (the best value for the number of nearest-neighbors)
    """
    grid_search_cv = GridSearchCV (
        model_callable = model_call,
        param_grid = param_grid,
        scoring_func = scoring_func,
        cv_object = cv,
        )
    
    # Get the last sorted value and take k from that values
    return sorted ( list ( grid_search_cv.get_cv_scores ( X, y ) ), key = lambda x: x [ 1 ] ) [ -1 ][ 0 ][ "k" ]
    # End choose_k()

def run_experiment ( X, y, model_call, param_grid = None, scoring_func = accuracy,cv = KFoldStratifiedCV ( number_of_folds = 5 ),):
    
    """
    Function runs a single experiment. If a param_grid is passed, it will select `k` from the values passed.
        Parameters:
            X: Indicates the feature matrix. Type: Array (np.ndarray)
            y: Indicates the target vector. Type: Array (np.ndarray)
            model_call: A function that returns a model object. Its arguments must be the keys in param_grid. Type: Callable
            param_grid: A mapping of arguments to values that we want to try. Type: Dictionary (key:value)
            scoring_func: The function that scores the results of a model. This value is maximized.Type: Callable
            cv: The validation object to use for the cross validation.
        Returns: model
    """

    scores = []
    iteration = 0
    # Iterate through the split
    for train, test in cv.split ( y ):
        # If first iteration and k values are passed, get the best one
        if iteration == 0 and param_grid:
            k = choose_k (
                X [ train ], y [ train ], model_call, param_grid, scoring_func, cv = cv )
            logger.info ( f"Choosing k= { k } " )
        else:
            # Defaults to 1 for condensed.
            k = 1

        iteration += 1

        # Instantiate the model with the value of k
        model = model_call ( k = k )

        # Standardize the data
        standardizer = Standardizer ( mean = True, std = True )

        # Fit the model
        model.fit ( X = standardizer.fit_transform ( X [ train ] ), y = y [ train ] )

        # make test set predictions
        y_pred = model.predict ( X = standardizer.transform ( X [ test ] ) )

        # Append the score
        scores.append ( scoring_func ( y [ test ], y_pred ) )
        
    logger.info ( f"Avg Score: { np.mean ( scores ) } " )
    
    return model
    # End run_experiment()

#ETL, EDA

# Correlations
def correlations ( data, y, xs ):
    rs = [] # pearson's r
    rhos = [] # rho
    for x in xs:
        r = stats.pearsonr ( data [ y ], data [ x ] ) [ 0 ]
        rs.append ( r )
        rho = stats.spearmanr ( data [ y ], data [ x ] ) [ 0 ]
        rhos.append ( rho )
    return pd.DataFrame ( { "feature": xs, "r": rs, "rho": rhos } )
    # End correlations()

# Pair-wise Comparisons

def describe_by_category ( data, numeric, categorical, transpose = False ):
    grouped = data.groupby ( categorical )
    grouped_y = grouped [ numeric ].describe()
    if transpose:
        print( grouped_y.transpose() )
    else:
        print ( grouped_y )
    # End describe_by_category


# # Ecoli Data Set
# ## Extract, Transform, Load: Ecoli Data
# 
# 
# ### Description
# 
# A data set to classify localization sites of proteins in ecoli cells. Three of the classes
# have a very small number of examples. These should be deleted from the data set.
# 
# Data obtained from:https://archive.ics.uci.edu/ml/datasets/Ecoli
# 
# ### Attribute Information: 8 Attributes (d)
# 
# 1. Sequence Name: Accession number for the SWISS-PROT database 
# 2. mcg: McGeoch's method for signal sequence recognition. 
# 3. gvh: von Heijne's method for signal sequence recognition. 
# 4. lip: von Heijne's Signal Peptidase II consensus sequence score. Binary attribute. 
# 5. chg: Presence of charge on N-terminus of predicted lipoproteins. Binary attribute. 
# 6. aac: score of discriminant analysis of the amino acid content of outer membrane and periplasmic proteins. 
# 7. alm1: score of the ALOM membrane spanning region prediction program. 
# 8. alm2: score of ALOM program after excluding putative cleavable signal regions from the sequence.
# 
# ### One Class Label
# 9. Class

# In[14]:


# Log ETL: Ecoli Data
logger.info ( "ETL: Ecoli Data Set" )

# Read Ecoli Data
# Create dataframe
ecoli_data = pd.read_csv ( 
        io.StringIO (
            r.get (
                "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
            )
            .text.replace ( "   ", " " )
            .replace ("  ", " " ) 
        ),
        sep = " ", 
        header = None,
        #Assign labels to columns
        names = [
            "id",
            "mcg",
            "gvh",
            "lip",
            "chg",
            "aac",
            "alm1",
            "alm2",
            "class",
        ],
    )


# In[15]:


# Confirm data was properly read by examining data frame
ecoli_data.info()


# **Notes**
# 
# As expected, we see 9 columns (8 attributes and 1 class label). There are 336 entries (n = 336). We see that the instance id is an object, but all other attributes are float type variables. We know we'll want to eliminate the id column

# In[16]:


# Look at first few rows of dataframe
ecoli_data.head()


# In[17]:


np.random.seed ( 6222020 ) # Due date
# Remove id column
ecoli_data_category_variables_only = ecoli_data.drop ( "id", axis = 1 ).sample ( frac = 1 )

# Confirm
ecoli_data_category_variables_only.head()


# In[18]:


# Classification for Class Label:
ecoli_data_category_variables_only [ "class" ].astype ( "category" ).cat.codes.values


# In[19]:


# Verify whether any values are null
ecoli_data_category_variables_only.isnull().values.any()


# **Notes**
# 
# There are no missing values

# In[20]:


# Again
ecoli_data_category_variables_only.isna().any()


# **Notes** 
# 
# Again, we see no null values

# ## (Brief) Exploratory Data Analysis: Breast Cancer Data
# 
# ### Single Variables
# 
# Let's look at the summary statistics & Tukey's 5
# 

# In[21]:


# Log EDA: Ecoli Data
logger.info ( "EDA: Ecoli Data Set" )


# In[22]:


# Descriptive Statistics
ecoli_data_category_variables_only.describe()


# **Notes**
# 
# Total number of observations: 336
# 
# Data includes: 
#     - Arithmetic Mean
#     - Variance: Spread of distribution
#     - Standard Deviation:Square root of variance
#     - Minimum Observed Value
#     - Maximum Observed Value
#    
# 
# Q1: For $Attribute_{i}$ 25% of observations are between min $Attribute_{i}$ and Q1 $Attribute_{i}$
# 
# Q2: Median. Thus for $Attribute_{i}$, 50% of observations are lower (but not lower than min $Attribute_{i}$) and 50% of the observations are higher (but not higher than Q3 $Attribute_{i}$
# 
# Q4: For $Attribute_{i}$, 75% of the observations are between min $Attribute_{i}$ and Q4 $Attribute_{i}$
# 
# If we wanted, we could use this information for each attribute to calculate the following:
#    - Interquartile Range: Q3-Q1
#    - Whisker: 1.5 * IQR (Outliers lie beyond the whisker)

# ## (Brief) Exploratory Data Analysis: Ecoli Data
# 
# ### Pair-Wise: Attribute by Class

# In[23]:


# Frequency of classifications
ecoli_data_category_variables_only [ 'class' ].value_counts() # raw counts


# In[24]:


# Plot diagnosis frequencies
sns.countplot ( ecoli_data_category_variables_only [ 'class' ],label = "Count" ) # boxplot


# In[25]:


def describe_by_category ( data, numeric, categorical, transpose = False ):
    grouped = data.groupby ( categorical )
    grouped_y = grouped [ numeric ].describe()
    if transpose:
        print( grouped_y.transpose() )
    else:
        print ( grouped_y )


# In[26]:


# Descriptive Statistics: Describe each variable by class (means only)
ecoli_data_category_variables_only.groupby ( [ 'class' ] )[ 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2' ].mean()


# In[27]:


# Descriptive Statistics: Describe each variable by class
ecoli_data_category_variables_only.groupby ( [ 'class' ] )[ 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2' ].describe()


# In[28]:


boxplot = ecoli_data_category_variables_only.boxplot ( column = [ 'mcg', 'gvh'], by = [ 'class' ] )


# In[29]:


boxplot = ecoli_data_category_variables_only.boxplot ( column = [ 'lip', 'chg' ], by = [ 'class' ] )


# In[30]:


boxplot = ecoli_data_category_variables_only.boxplot ( column = [ 'aac', 'alm1' ], by = [ 'class' ] )


# In[31]:


boxplot = ecoli_data_category_variables_only.boxplot ( column = [ 'alm2' ], by = [ 'class' ] )


# In[32]:


# Descriptive Statistics: mcg by Class
describe_by_category ( ecoli_data_category_variables_only, "mcg", "class", transpose = True ) 


# In[33]:


# Descriptive Statistics: gvh Size by Class
describe_by_category ( ecoli_data_category_variables_only, "gvh", "class", transpose = True )


# In[34]:


# Descriptive Statistics: Lip by Class
describe_by_category ( ecoli_data_category_variables_only, "lip", "class", transpose = True )


# In[35]:


# Descriptive Statistics: Chg by Class
describe_by_category ( ecoli_data_category_variables_only, "chg", "class", transpose = True )


# In[36]:


# Descriptive Statistics: Aac by Class
describe_by_category ( ecoli_data_category_variables_only, "aac", "class", transpose = True )


# In[37]:


# Descriptive Statistics: Alm1 by Class
describe_by_category ( ecoli_data_category_variables_only, "alm1", "class", transpose = True )


# In[38]:


# Descriptive Statistics: Alm2 by Class
describe_by_category ( ecoli_data_category_variables_only, "alm2", "class", transpose = True )


# ## KNN: Ecoli Data
# 
# We know the Ecoli Data Set is used to classify localization sites of proteins in ecoli cells. As this is a classification problem, we know we'll need to use the KNN Classifier
# 
# ### Assign Feature Matrix & Target Vector

# In[39]:


# Assign Variables
# X = Feature Matrix; All Attributes (i.e., drop instance class column)
# y = Target Vector; Categorical instance class (i.e., doesn't include the attribute features)
X = ecoli_data_category_variables_only.drop ( axis = 1, labels = "class" ).values
y = ecoli_data_category_variables_only [ "class" ].astype( "category" ).cat.codes.values


# ### K-Nearest Neighbors Classification

# In[40]:


# Classify k-Nearest Neighbors

# Log Experiment: Standard KNN Classification
logger.info ( "Running Standard KNN Classification: Ecoli Data" )

# Run Experiment: Standardize data, fit model (split X & y; use training data set), make predictions (split X & y; use test data set)
run_experiment (
        X,
        y,
        model_call = lambda k: KNearestNeighbors ( k = k ),
        param_grid = { "k": [ 1, 2, 3, 4, 5 ] },
    )


# In[41]:


# Reduce Error & Improve Cost
# Edited KNN (reduce training set)

# Log Experiment: Edited KNN ()
logger.info ( "Running Edited KNN: Ecoli Data" )

np.random.seed ( 6222020 )

# Run Experiment: Standardize data, fit model (split X & y; use training data set), make predictions (split X & y; use test data set)
run_experiment (
        X,
        y,
        model_call = lambda k: EditedKNN ( k = k, proportion_cv = 0.1 ),
        param_grid = { "k": [ 1, 2, 3, 4, 5 ] },
    )


# In[42]:


# Improve Cost
# Condensed KNN (Reduced dataset)

# Log Experiment: Condensed KNN
logger.info ( "Running Condensed KNN: Ecoli Data" )

np.random.seed ( 6222020 )

# Run Experiment: Standardize data, fit model (split X & y; use training data set), make predictions (split X & y; use test data set)
model = run_experiment ( X, y, model_call = lambda k: CondensedKNN ( verbose = True ) )


# # Image Segmentation Data Set
# ## Extract, Transform, Load: Image Segmentation Data
# 
# ### Description
# 
# [Classification] The instances were drawn randomly from a database of 7 outdoor images. The images
# were handsegmented to create a classification for every pixel.
# 
# Data obtained from:https://archive.ics.uci.edu/ml/datasets/Image+Segmentation
# 
# ### Attribute Information: 19 Attributes (d)
# 
# 1. region-centroid-col: the column of the center pixel of the region. 
# 2. region-centroid-row: the row of the center pixel of the region. 
# 3. region-pixel-count: the number of pixels in a region = 9. 
# 4. short-line-density-5: the results of a line extractoin algorithm that counts how many lines of length 5 (any orientation) with low contrast, less than or equal to 5, go through the region. 
# 5. short-line-density-2: same as short-line-density-5 but counts lines of high contrast, greater than 5. 
# 6. vedge-mean: measure the contrast of horizontally adjacent pixels in the region. There are 6, the mean and standard deviation are given. This attribute is used as a vertical edge detector. 
# 7. vegde-sd: (see 6) 
# 8. hedge-mean: measures the contrast of vertically adjacent pixels. Used for horizontal line detection. 
# 9. hedge-sd: (see 8). 
# 10. intensity-mean: the average over the region of (R + G + B)/3 
# 11. rawred-mean: the average over the region of the R value. 
# 12. rawblue-mean: the average over the region of the B value. 
# 13. rawgreen-mean: the average over the region of the G value. 
# 14. exred-mean: measure the excess red: (2R - (G + B)) 
# 15. exblue-mean: measure the excess blue: (2B - (G + R)) 
# 16. exgreen-mean: measure the excess green: (2G - (R + B)) 
# 17. value-mean: 3-d nonlinear transformation of RGB. (Algorithm can be found in Foley and VanDam, Fundamentals of Interactive Computer Graphics) 
# 18. saturatoin-mean: (see 17) 
# 19. hue-mean: (see 17)
# 
# ### One Class Label
# 20. class (class attribute) 

# In[43]:


# Log ETL: Image Segmentation Data
logger.info ( "ETL: Image Segmentation Data Set" )

# Read Image Segmentation Data
# Create dataframe
image_segmentation_data = pipe (
        r.get (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data"
        ).text.split ( "\n" ),
        lambda lines: pd.read_csv (
            io.StringIO ( "\n".join ( lines [ 5: ] ) ), header = None, names = lines [ 3 ].split ( "," ) ),
        lambda df: df.assign (
            instance_class = lambda df: df.index.to_series().astype ( "category" ).cat.codes
        ), )


# In[44]:


# Confirm data was properly read by examining data frame
image_segmentation_data.info()


# **Notes**
# 
# As expected, we see 20 columns (19 attributes & one class instance). There are 210 entries (n = 210). We see that the attribute/feature REGION-PIXEL-COUNT is an integer, but all other attributes are float type variables. 

# In[45]:


# Look at first few rows of dataframe
image_segmentation_data.head()


# In[46]:


# Verify whether any values are null
image_segmentation_data.isnull().values.any()


# In[47]:


# Again
image_segmentation_data.isna().any()


# ## (Brief) Exploratory Data Analysis: Image Segmentation Data
# 
# ### Single Variables
# 
# Let's look at the summary statistics & Tukey's 5

# In[48]:


# Log EDA: Image Segmentation Data
logger.info ( "EDA: Image Segmentation Data Set" )

# Descriptive Statistics
image_segmentation_data.describe()


# **Notes** 
# 
# Total number of observations: 210 (i.e., n = 210)
# 
# Data includes: 
#     - Arithmetic Mean
#     - Variance: Spread of distribution
#     - Standard Deviation:Square root of variance
#     - Minimum Observed Value
#     - Maximum Observed Value
# 
# Q1: For $Attribute_{i}$ 25% of observations are between min $Attribute_{i}$ and Q1 $Attribute_{i}$
# 
# Q2: Median. Thus for $Attribute_{i}$, 50% of observations are lower (but not lower than min $Attribute_{i}$) and 50% of the observations are higher (but not higher than Q3 $Attribute_{i}$
# 
# Q4: For $Attribute_{i}$, 75% of the observations are between min $Attribute_{i}$ and Q4 $Attribute_{i}$
# 
# We'll likely want to discretize these attributes by class

# ## (Brief) Exploratory Data Analysis: Image Segmentation Data
# 
# ### Pair-Wise: Attribute by Class

# In[49]:


# Rename column
image_segmentation_data.rename ( columns = { "instance_class":"class" }, inplace = True )


# In[50]:


# Frequency of glass classifications
image_segmentation_data [ 'class' ].value_counts() # raw counts


# **Notes**
# 
# There are 7 image segmentation classifications (labeled 0, 1, 2, 3, .., 7)
# 
# Each image segmentation classification has 30 observations 
# 

# In[51]:


# Plot diagnosos frequencies
sns.countplot ( image_segmentation_data [ 'class' ],label = "Count" ) # boxplot


# **Notes**
# 
# There are 7 image segmentation classifications (labeled 0, 1, 2, 3, .., 7)
# 
# Each image segmentation classification has 30 observations 

# In[52]:


# Get column names
print ( image_segmentation_data.columns )


# In[53]:


# Descriptive Statistics: Describe each variable by class (means only)
image_segmentation_data.groupby ( [ 'class' ] )[ 'REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT',
       'SHORT-LINE-DENSITY-5', 'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN',
       'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN', 'RAWRED-MEAN',
       'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN', 'EXBLUE-MEAN',
       'EXGREEN-MEAN', 'VALUE-MEAN', 'SATURATION-MEAN', 'HUE-MEAN' ].mean()


# In[54]:


# Descriptive Statistics: Describe each variable by class (all variables)
image_segmentation_data.groupby ( [ 'class' ] ) [ 'REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT',
       'SHORT-LINE-DENSITY-5', 'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN',
       'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN', 'RAWRED-MEAN',
       'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN', 'EXBLUE-MEAN',
       'EXGREEN-MEAN', 'VALUE-MEAN', 'SATURATION-MEAN', 'HUE-MEAN' ].describe()        


# In[55]:


boxplot = image_segmentation_data.boxplot ( column = [ 'REGION-CENTROID-COL', 'REGION-CENTROID-ROW' ], by = [ 'class' ] )  


# In[56]:


boxplot = image_segmentation_data.boxplot ( column = [ 'REGION-PIXEL-COUNT','SHORT-LINE-DENSITY-5' ], by = [ 'class' ] )


# In[57]:


boxplot = image_segmentation_data.boxplot ( column = [ 'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN' ], by = [ 'class' ] )


# In[58]:


boxplot = image_segmentation_data.boxplot ( column = [ 'VEDGE-SD', 'HEDGE-MEAN' ], by = [ 'class' ] )


# In[59]:


boxplot = image_segmentation_data.boxplot ( column = [ 'HEDGE-SD', 'INTENSITY-MEAN'], by = [ 'class' ] )


# In[60]:


boxplot = image_segmentation_data.boxplot ( column = [ 'RAWRED-MEAN','RAWBLUE-MEAN' ], by = [ 'class' ] )


# In[61]:


boxplot = image_segmentation_data.boxplot ( column = [ 'RAWGREEN-MEAN' ], by = [ 'class' ] )


# In[62]:


boxplot = image_segmentation_data.boxplot ( column = [ 'EXRED-MEAN', 'EXBLUE-MEAN' ], by = [ 'class' ] )


# In[63]:


boxplot = image_segmentation_data.boxplot ( column = [ 'EXGREEN-MEAN'], by = [ 'class' ] )


# In[64]:


boxplot = image_segmentation_data.boxplot ( column = [ 'VALUE-MEAN', 'SATURATION-MEAN' ], by = [ 'class' ] )


# In[65]:


boxplot = image_segmentation_data.boxplot ( column = [ 'HUE-MEAN' ], by = [ 'class' ] )


# In[66]:


# Descriptive Statistics: Describe each variable by class
# REGION-CENTROID-COL by Class
describe_by_category ( image_segmentation_data, 'REGION-CENTROID-COL', "class", transpose = True )


# In[67]:


# Descriptive Statistics: Describe each variable by class
# REGION-CENTROID-ROW by Class
describe_by_category ( image_segmentation_data, 'REGION-CENTROID-ROW', "class", transpose = True )


# In[68]:


# Descriptive Statistics: Describe each variable by class
# REGION-PIXEL-COUNT by Class
describe_by_category ( image_segmentation_data, 'REGION-PIXEL-COUNT', "class", transpose = True )


# In[69]:


# Descriptive Statistics: Describe each variable by class
# SHORT-LINE-DENSITY-5 by Class
describe_by_category ( image_segmentation_data, 'SHORT-LINE-DENSITY-5', "class", transpose = True )


# In[70]:


# Descriptive Statistics: Describe each variable by class
# SHORT-LINE-DENSITY-2 by Class
describe_by_category ( image_segmentation_data, 'SHORT-LINE-DENSITY-2', "class", transpose = True )


# In[71]:


# Descriptive Statistics: Describe each variable by class
# VEDGE-MEAN by Class
describe_by_category ( image_segmentation_data, 'VEDGE-MEAN', "class", transpose = True )


# In[72]:


# Descriptive Statistics: Describe each variable by class
# VEDGE-SD by Class
describe_by_category ( image_segmentation_data, 'VEDGE-SD', "class", transpose = True )


# In[73]:


# Descriptive Statistics: Describe each variable by class
# 'HEDGE-MEAN' by Class
describe_by_category ( image_segmentation_data, 'HEDGE-MEAN', "class", transpose = True )


# In[74]:


# Descriptive Statistics: Describe each variable by class
# HEDGE-SD by Class
describe_by_category ( image_segmentation_data, 'HEDGE-SD', "class", transpose = True )


# In[75]:


# Descriptive Statistics: Describe each variable by class
# INTENSITY-MEAN by Class
describe_by_category ( image_segmentation_data, 'INTENSITY-MEAN', "class", transpose = True )


# In[76]:


# Descriptive Statistics: Describe each variable by class
# RAWRED-MEAN by Class
describe_by_category ( image_segmentation_data, 'RAWRED-MEAN', "class", transpose = True )


# In[77]:


# Descriptive Statistics: Describe each variable by class
# RAWBLUE-MEAN by Class
describe_by_category ( image_segmentation_data, 'RAWBLUE-MEAN', "class", transpose = True )


# In[78]:


# Descriptive Statistics: Describe each variable by class
# RAWGREEN-MEAN by Class
describe_by_category ( image_segmentation_data, 'RAWGREEN-MEAN', "class", transpose = True )


# In[79]:


# Descriptive Statistics: Describe each variable by class
# EXRED-MEAN by Class
describe_by_category ( image_segmentation_data, 'EXRED-MEAN', "class", transpose = True )


# In[80]:


# Descriptive Statistics: Describe each variable by class
# EXBLUE-MEAN by Class
describe_by_category ( image_segmentation_data, 'EXBLUE-MEAN', "class", transpose = True )


# In[81]:


# Descriptive Statistics: Describe each variable by class
# EXGREEN-MEAN by Class
describe_by_category ( image_segmentation_data, 'EXGREEN-MEAN', "class", transpose = True )


# In[82]:


# Descriptive Statistics: Describe each variable by class
# VALUE-MEAN by Class
describe_by_category ( image_segmentation_data, 'VALUE-MEAN', "class", transpose = True )


# In[83]:


# Descriptive Statistics: Describe each variable by class
# SATURATION-MEAN by Class
describe_by_category ( image_segmentation_data, 'SATURATION-MEAN', "class", transpose = True )


# In[84]:


# Descriptive Statistics: Describe each variable by class
# HUE-MEAN by Class
describe_by_category ( image_segmentation_data, 'HUE-MEAN', "class", transpose = True )


# ## KNN: Image Segmentation Data
# 
# We know that the Image Segmentation data set is used to create a classification for every pixel from a database of 7 outdoor images. Since this is a classification problem, we know we'll need to use the KNN Classifier
# 
# ### Assign Feature Matrix & Target Vector

# In[85]:


# Assign Variables
# X = Feature Matrix; All Attributes (i.e., drop instance class column)
X = image_segmentation_data.drop ( [ "class", "REGION-PIXEL-COUNT" ], axis = 1 ).values
y = image_segmentation_data [ "class" ].values


# ### K-Nearest Neighbors Classification

# In[86]:


# Classify k-Nearest Neighbors

# Log Experiment: Standard KNN Classification
logger.info ( "Running Standard KNN Classification: Image Segmentation Data" )

# Run Experiment: Standardize data, fit model (split X & y; use training data set), make predictions (split X & y; use test data set)
run_experiment (
        X,
        y,
        model_call = lambda k: KNearestNeighbors ( k = k ),
        param_grid = { "k": [ 1, 2, 3, 4, 5 ] },
    )


# In[87]:


# Reduce Error & Improve Cost
# Edited KNN (reduce training set)
logger.info ( "Running Edited KNN: Image Segmentation Data" )

np.random.seed ( 6222020 )

# Run Experiment: Standardize data, fit model (split X & y; use training data set), make predictions (split X & y; use test data set)
run_experiment (
        X,
        y,
        model_call = lambda k: EditedKNN ( k = k, proportion_cv = 0.2 ),
        param_grid = { "k": [ 1, 2, 3, 4, 5 ] },
    )


# In[88]:


# Improve Cost
# Condensed KNN (Reduced dataset)

# Log Experiment: Condensed KNN
logger.info ( "Running Condensed KNN: Image Segmentation Data" )

np.random.seed ( 6222020 )

# Run Experiment: Standardize data, fit model (split X & y; use training data set), make predictions (split X & y; use test data set)
run_experiment ( X, y, model_call = lambda k: CondensedKNN ( verbose = True ) )


# # Computer Hardware Data Set
# ## Extract, Transform, Load: Computer Hardware Data
# 
# ### Description
# 
# [Regression] The estimated relative performance values were estimated by the authors using a linear
# regression method. 
# 
# Data obtained from:https://archive.ics.uci.edu/ml/datasets/Iris
# 
# ### Attribute Information: 9 Attributes (d)
# 
# 2. Model Name: many unique symbols 
# 3. MYCT: machine cycle time in nanoseconds (integer) 
# 4. MMIN: minimum main memory in kilobytes (integer) 
# 5. MMAX: maximum main memory in kilobytes (integer) 
# 6. CACH: cache memory in kilobytes (integer) 
# 7. CHMIN: minimum channels in units (integer) 
# 8. CHMAX: maximum channels in units (integer) 
# 9. PRP: published relative performance (integer) 
# 10. ERP: estimated relative performance from the original article (integer)
# 
# ### One Class Label
# 1. Vendor Name (class):
#     - adviser
#     - amdah
#     - apollo
#     - basf
#     - bti
#     - burroughs
#     - c.r.d
#     - cambex
#     - cdc
#     - dec 
#     - dg
#     - formation
#     - four-phase
#     - gould
#     - honeywell
#     - hp
#     - ibm
#     - ipl
#     - magnuson
#     - microdata
#     - nas
#     - ncr
#     - nixdorf
#     - perkin-elmer
#     - prime
#     - siemens
#     - sperry
#     - sratus
#     - wang

# In[89]:


# Log ETL: Computer Hardware Data
logger.info ( "ETL: Computer Hardware Data Set" )

# Read Computer Hardware Data
# Create dataframe & label columns
computer_hardware_data = pd.read_csv (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data",
        header = None,
        names = [
            "vendor_name",
            "model_name",
            "MYCT",
            "MMIN",
            "MMAX",
            "CACH",
            "CHMIN",
            "CHMAX",
            "PRP",
            "ERP",
        ],
    )


# In[90]:


# Confirm data was properly read by examining data frame
computer_hardware_data.info()


# **Notes**
# 
# As expected, we see 10 columns (9 attributes and 1 class label). There are 209 entries (n = 209). We see that the instance class (vendor_name) is an object, as is the model_name, but all other attributes are integer type variables.

# In[91]:


# Verify whether any values are null
computer_hardware_data.isnull().values.any()


# **Notes**
# 
# We observe no null instances

# In[92]:


# Again
computer_hardware_data.isna().any()


# **Notes**
# 
# We observe no null instances in any of the attribute columns

# In[93]:


# Look at first few rows of dataframe
computer_hardware_data.head()


# In[94]:


# Classification for Class Label: data frame for this category
computer_hardware_data[ "vendor_name" ].astype ( "category" ).cat.codes


# ## (Brief) Exploratory Data Analysis: Computer Hardware Data
# 
# ### Single Variable

# In[95]:


# Log EDA: Computer Hardware Data
logger.info ( "EDA: Computer Hardware Data Set" )

# Descriptive Statistics
computer_hardware_data.describe()


# **Notes**
# 
# Total number of observations: 209
# 
# Data includes: 
#     - Arithmetic Mean
#     - Variance: Spread of distribution
#     - Standard Deviation:Square root of variance
#     - Minimum Observed Value
#     - Maximum Observed Value
#    
# 
# Q1: For $Attribute_{i}$ 25% of observations are between min $Attribute_{i}$ and Q1 $Attribute_{i}$
# 
# Q2: Median. Thus for $Attribute_{i}$, 50% of observations are lower (but not lower than min $Attribute_{i}$) and 50% of the observations are higher (but not higher than Q3 $Attribute_{i}$
# 
# Q4: For $Attribute_{i}$, 75% of the observations are between min $Attribute_{i}$ and Q4 $Attribute_{i}$
# 

# ## (Brief) Exploratory Data Analysis: Iris Data
# 
# ### Pair-Wise: Attribute by Class

# In[96]:


# Frequency of diagnoses classifications
computer_hardware_data [ 'vendor_name' ].value_counts() # raw counts


# In[97]:


# Plot diagnosos frequencies
sns.countplot ( computer_hardware_data [ 'vendor_name' ],label = "Count" ) # boxplot


# In[98]:


# Descriptive Statistics: Describe each variable by class (means only)
computer_hardware_data.groupby ( [ 'vendor_name' ] )[ "MYCT", "MMIN", "MMAX", "CACH","CHMIN", "CHMAX","PRP", "ERP" ].mean()


# In[99]:


# Descriptive Statistics: Describe each variable by class (means only)
computer_hardware_data.groupby ( [ 'vendor_name' ] )[ "MYCT", "MMIN", "MMAX", "CACH","CHMIN", "CHMAX","PRP", "ERP" ].describe()


# In[100]:


boxplot = computer_hardware_data.boxplot ( column = [ "MYCT" ], by = [ 'vendor_name' ] )


# In[101]:


boxplot = computer_hardware_data.boxplot ( column = [ "CACH" ], by = [ 'vendor_name' ] )


# In[102]:


boxplot = computer_hardware_data.boxplot ( column = [ "MMIN" ], by = [ 'vendor_name' ] )


# In[103]:


boxplot = computer_hardware_data.boxplot ( column = [ "MMAX" ], by = [ 'vendor_name' ] )


# In[104]:


boxplot = computer_hardware_data.boxplot ( column = [ "CHMIN" ], by = [ 'vendor_name' ] )


# In[105]:


boxplot = computer_hardware_data.boxplot ( column = [ "CHMAX" ], by = [ 'vendor_name' ] )


# In[106]:


boxplot = computer_hardware_data.boxplot ( column = [ "PRP" ], by = [ 'vendor_name' ] )


# In[107]:


boxplot = computer_hardware_data.boxplot ( column = [ "ERP" ], by = [ 'vendor_name' ] )


# In[108]:


# Descriptive Statistics: Attribute by Class
# MYCT by Class
describe_by_category ( computer_hardware_data, "MYCT", "vendor_name", transpose = True )

"MYCT", "MMIN", "MMAX", "CACH","CHMIN", "CHMAX","PRP", "ERP"


# **Notes**
# 

# In[109]:


# Descriptive Statistics: Attribute by Class
# MMIN by Class
describe_by_category ( computer_hardware_data, "MMIN", "vendor_name", transpose = True )


# **Notes**
# 
# 

# In[110]:


# Descriptive Statistics: Attribute by Class
# MMAX by Class
describe_by_category ( computer_hardware_data, "MMAX", "vendor_name", transpose = True )


# **Notes**
# 

# In[111]:


# Descriptive Statistics: Attribute by Class
# CACH by Class
describe_by_category ( computer_hardware_data, "CACH", "vendor_name", transpose = True )


# **Notes**
# 
# 

# In[112]:


# Descriptive Statistics: Attribute by Class
# CHMIN by Class
describe_by_category ( computer_hardware_data, "CHMIN", "vendor_name", transpose = True )


# In[113]:


# Descriptive Statistics: Attribute by Class
# CHMAX by Class
describe_by_category ( computer_hardware_data, "CHMAX", "vendor_name", transpose = True )


# In[114]:


# Descriptive Statistics: Attribute by Class
# PRP by Class
describe_by_category ( computer_hardware_data, "PRP", "vendor_name", transpose = True )


# In[115]:


# Descriptive Statistics: Attribute by Class
# ERP by Class
describe_by_category ( computer_hardware_data, "ERP", "vendor_name", transpose = True )


# ## KNN: Computer Hardware Data
# 
# We know this dataset is used for a regression problem, so we'll use the KNN Regressor
# 
# ### Assign Feature Matrix & Target Vector
# 

# In[116]:


# Assign Variables
# X = Feature Matrix; All Attributes (i.e., drop instance class column)
# y = Target Vector; Categorical instance class (i.e., doesn't include the attribute features)
X = computer_hardware_data.drop ( [ "vendor_name", "model_name", "PRP", "ERP" ], axis = 1 ).values
y_real = computer_hardware_data [ "PRP" ].values
y_ols = computer_hardware_data [ "ERP" ].values


# ### K-Nearest Neighbors Regression

# In[117]:


#KNN Regression

# Log Experiment: KNN Regression Computer Hardware Data
logger.info ( "Running KNN Regression on Computer Hardware Data Set" )

np.random.seed ( 6222020 )

# Run Experiment: Standardize data, fit model (split X & y; use training data set), make predictions (split X & y; use test data set)
run_experiment (
        X = X,
        y = y_real,
        model_call = lambda k: KNearestNeighborRegression ( k = k),
        param_grid = { "k": list ( range ( 1, 5 ) ) },
        scoring_func = lambda *args, **kwargs: -1
        * np.sqrt ( mean_squared_error ( *args, **kwargs ) ),
        cv = KFoldCV ( number_of_folds = 5 ),
    )


# # Forest Fires Data Set
# ## Extract, Transform, Load: Forest Fires Data
# 
# ### Description
# 
# [Regression] This is a difficult regression task, where the aim is to predict the burned area of forest
# fires, in the northeast region of Portugal, by using meteorological and other data .
# 
# Data obtained from:https://archive.ics.uci.edu/ml/datasets/Forest+Fires
# 
# ### Attribute Information: 13 Attributes (d)
# 
# 1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9 
# 2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9 
# 3. month - month of the year: 'jan' to 'dec' 
# 4. day - day of the week: 'mon' to 'sun' 
# 5. FFMC - FFMC index from the FWI system: 18.7 to 96.20 
# 6. DMC - DMC index from the FWI system: 1.1 to 291.3 
# 7. DC - DC index from the FWI system: 7.9 to 860.6 
# 8. ISI - ISI index from the FWI system: 0.0 to 56.10 
# 9. temp - temperature in Celsius degrees: 2.2 to 33.30 
# 10. RH - relative humidity in %: 15.0 to 100 
# 11. wind - wind speed in km/h: 0.40 to 9.40 
# 12. rain - outside rain in mm/m2 : 0.0 to 6.4 
# 13. area - the burned area of the forest (in ha): 0.00 to 1090.84 
# (this output variable is very skewed towards 0.0, thus it may make 
# sense to model with the logarithm transform).
# 

# In[118]:


# Log ETL: Forest Fire Data
logger.info ( "ETL: Forest Fire Data Set" )

# Read Forest Fire Data
# Create dataframe
forest_fire_data = pd.read_csv (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
    )


# In[119]:


# Confirm data was properly read by examining data frame
forest_fire_data.info()


# **Notes**
# 
# As expected, we see 13 columns (# attributes and # class label). There are 517 entries (n = 517). We see that month and day attributes are objects; X, Y, RH are integer type variables; and FFMC, DMC, DC, ISI, temp, wind, rain, area variables are all float type.

# In[120]:


# Verify whether any values are null
forest_fire_data.isnull().values.any()


# **Note**
# 
# We see there are no null instances

# In[121]:


# Again
forest_fire_data.isna().any()


# ## (Brief) Exploratory Data Analysis: Forrest Fire Data
# 
# ### Single Variable
# 
# Let's look at the summary statistics & Tukey's 5
# 

# In[122]:


# Look at first few rows of dataframe
forest_fire_data.head()


# ## (Brief) Exploratory Data Analysis: Forest Fire Data
# 
# ### Single Variable
# 
# Let's look at the summary statistics & Tukey's 5
# 

# In[123]:


# Log EDA: Forest Fire Data
logger.info ( "EDA: Forest Fire Data Set" )

# Descriptive Statistics
forest_fire_data.describe()


# **Notes**
# 
# Total number of observations: 517
# 
# Data includes: 
#     - Arithmetic Mean
#     - Variance: Spread of distribution
#     - Standard Deviation:Square root of variance
#     - Minimum Observed Value
#     - Maximum Observed Value
#    
# 
# Q1: For $Attribute_{i}$ 25% of observations are between min $Attribute_{i}$ and Q1 $Attribute_{i}$
# 
# Q2: Median. Thus for $Attribute_{i}$, 50% of observations are lower (but not lower than min $Attribute_{i}$) and 50% of the observations are higher (but not higher than Q3 $Attribute_{i}$
# 
# Q4: For $Attribute_{i}$, 75% of the observations are between min $Attribute_{i}$ and Q4 $Attribute_{i}$
# 

# ## (Brief) Exploratory Data Analysis: Forest Fire Data
# 
# ### Pair-Wise: Attribute by Class

# In[124]:


# Frequency of diagnoses classifications
forest_fire_data [ 'area' ].value_counts() # raw counts


# In[125]:


# Get Columns
list ( forest_fire_data.columns )


# In[126]:


# Descriptive Statistics: Describe each variable by class (means only)
forest_fire_data.groupby ( [ 'area' ] )[ 'X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain' ].mean()


# In[127]:


# Descriptive Statistics: Describe each variable by class (means only)
forest_fire_data.groupby ( [ 'area' ] )[ 'X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain' ].describe()


# In[128]:


boxplot = forest_fire_data.boxplot ( column = [ "X", "Y"], by = [ 'area' ] )


# In[129]:


boxplot = forest_fire_data.boxplot ( column = [ 'FFMC','DMC' ], by = [ 'area' ] )


# In[130]:


boxplot = forest_fire_data.boxplot ( column = [ "DC", "ISI" ], by = [ 'area' ] )


# In[131]:


boxplot = forest_fire_data.boxplot ( column = [ "temp", "RH" ], by = [ 'area' ] )


# In[132]:


boxplot = forest_fire_data.boxplot ( column = [ "wind", "rain" ], by = [ 'area' ] )


# ## KNN: Forest Fire Data
# 
# We know we're solving a regression problem, so we'll use KNN Regressor
# 
# ### Assign Feature Matrix & Target Vector

# In[134]:


# Assign Variables
# X = Feature Matrix; All Attributes (i.e., drop instance class column)
# y = Target Vector; Categorical instance class (i.e., doesn't include the attribute features)
X = forest_fire_data.drop ( "area", axis = 1 ).pipe (
        lambda df: pd.get_dummies ( df, columns = [ "month", "day" ], drop_first = True )
    )
y = forest_fire_data [ "area" ].values


# ### KNN Regression

# In[135]:


# Log Experiment: KNN Regression Forest Fire Data (Full Design Matrix)
logger.info ( "Running KNN Regression (Full Design Matrix): Forest Fire Data Set" )

np.random.seed ( 62202020 )

# Run Experiment: Standardize data, fit model (split X & y; use training data set), make predictions (split X & y; use test data set)
model = run_experiment (
        X = X.values,
        y = y,
        model_call = lambda k: KNearestNeighborRegression ( k = k ),
        param_grid = { "k": list ( range ( 1, 5 ) ) },
        scoring_func = lambda * args, ** kwargs: -1
        * np.sqrt ( mean_squared_error ( *args, ** kwargs ) ),
        cv = KFoldCV ( number_of_folds = 5 ),
    )


# In[136]:


# Log Experiment: KNN Regression Forest Fire Data (Partial Design Matrix)
logger.info ( "Running KNN Regression (Partial Design Matrix): Forest Fire Data Set" )

np.random.seed ( 62202020 )

# Run Experiment: Standardize data, fit model (split X & y; use training data set), make predictions (split X & y; use test data set)
model = run_experiment (
        X = X.loc ( axis = 1 )[ "temp", "RH", "wind", "rain" ].values,
        y = y,
        model_call = lambda k: KNearestNeighborRegression ( k = k ),
        param_grid = { "k": list ( range ( 1, 5 ) ) },
        scoring_func = lambda * args, ** kwargs: -1
        * np.sqrt ( mean_squared_error ( * args, ** kwargs ) ),
        cv = KFoldCV (number_of_folds = 5 ),
    )


# In[ ]:




