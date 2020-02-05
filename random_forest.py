"""
Title: Random Forest: Classification & Regression
Author: Rodrigo Rangel
Description: * This script focuses on Random Forest for the following:
                  - Classification	
                  - Regression
			  
              * Random Forest is a method that uses a large number of decision trees (in parallel), 
               where the output of each tree is ensembled to make a prediction
              * Each tree is built from a sample of data (Bootstrap Sampling: Sampling with replacement)
              * Each tree uses a top-down method with recursive (binary) splitting
              * Each classification tree is split on highest Gini Index or Information Gain (Entropy)
              * Random Forest minimizes the impurities at each set 
			  
              * Tune:
                  - Number trees
                  - Tree depth
                  - Number features per sample
                  - Size of parent/leaf nodes
              * Reduces variance (overfitting) for unstable classifiers (ensemble)
              * Random Forest's main goal is to reduce the disorder/impurity or uncertainty at each split		  
"""

#-----------------------------------------------------------------------------#
#                              Dependencies                                   #
#-----------------------------------------------------------------------------#
#Dependencies
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression

# Creating Dataset
X,y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

X2,y2 = make_regression(n_samples=1000, n_features=4,
                           n_informative=2,
                           random_state=0, shuffle=False)

#-----------------------------------------------------------------------------#
#                           Random Forest: Classification                     #
#-----------------------------------------------------------------------------#
# Creating Classifier Object Defaults
rf = RandomForestClassifier(n_estimators=100  # Number of Trees: Usually between 100-1000
                            ,criterion='gini' # Split Function: 'gini' or 'entropy'
                            ,max_depth=None   # Maximum Deptch Of Tree: 
                            ,min_samples_split=2 # Minimum number samples required for split
                            ,min_samples_leaf=1 # Minimum samples required to be a lead node
                            ,max_features='auto' # Features to consider: int, float, 'auto' = 'sqrt','log2',None
                            ,max_leaf_nodes=None 
                            ,bootstrap=True # If False, whole dataset used to build each tree
                            ,oob_score=False # Whether to use out-of-bag samples to generalize accuracy
                            ,n_jobs=None # Whether or not to parallelize: None: No, -1:Yes
                            ,random_state=None) # Controlls random states of bootstrap

# Fitting X,y (Usually want to fit Train data only)
rf.fit(X, y)

# View Importance of Variables 
print(rf.feature_importances_)

# Make Predidctions (Usually Predict on Test data onlu)
X_test = [[0, 0, 0, 0]
         ,[1, 0, 1, 0]]
print(rf.predict(X_test))

#-----------------------------------------------------------------------------#
#                           Random Forest: Regressor                          #
#-----------------------------------------------------------------------------#

"""
The above parameters can be applied to Random Forest Clasifier with the 
exception of 'criterion' parameter
In classification trees, the criterion used is 'gini' or 'entropy'
In regression trees, the criterion used is 'mse' or 'mae'
"""

rf = RandomForestRegressor(n_estimators=100
						, criterion='mse' # Measure quality of split: 'mse', 'mae'
						, max_depth=None
						, min_samples_split=2
						, min_samples_leaf=1
						, min_weight_fraction_leaf=0.0
						, max_features='auto'
						, max_leaf_nodes=None
						, bootstrap=True
						, oob_score=False
						, n_jobs=None
						, random_state=None)

