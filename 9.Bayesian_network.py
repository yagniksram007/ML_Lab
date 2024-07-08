# Construct a bayesian network to analyse the diagnosis of heart patients using heart diseases dataset

import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the dataset
data = pd.read_csv('Dataset/heart.csv')

# Select a subset of key features and the target variable
subset_data = data[['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'target']]

# Display first few rows of the dataset
print(subset_data.head())

# Define a simpler structure of the Bayesian network
model = BayesianNetwork([
    ('age', 'target'),
    ('sex', 'target'),
    ('cp', 'target'),
    ('thalach', 'target'),
    ('exang', 'target'),
    ('oldpeak', 'target')
])

# Parameter learning using Maximum Likelihood Estimation
model.fit(subset_data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

evidence = {
    'age': 63, 
    'sex': 1, 
    'cp': 1, 
    'thalach': 150, 
    'exang': 0, 
    'oldpeak': 2.3
}

result = inference.query(variables=['target'], evidence=evidence)
print(result)

# Output:
# age  sex  cp  thalach  exang  oldpeak  target
# 0   52    1   0      168      0      1.0       0
# 1   53    1   0      155      1      3.1       0
# 2   70    1   0      125      1      2.6       0
# 3   61    1   0      161      0      0.0       0
# 4   62    0   0      106      0      1.9       0
# +-----------+---------------+
# | target    |   phi(target) |
# +===========+===============+
# | target(0) |        0.5000 |
# +-----------+---------------+
# | target(1) |        0.5000 |
# +-----------+---------------+