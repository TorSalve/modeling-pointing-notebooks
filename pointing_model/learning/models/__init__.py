from .knearest import KNearestNeighbors, KNearestNeighborsRegressor
from .svm import SupportVectorMachine, SupportVectorMachineRegression,\
    SupportVectorMachineMultiOutput
from .naive_bayes import NaiveBayes
from .random_forest import RandomForest, RandomForestRegression
from .voting import Voting
from .perceptron import Perceptron
from .stochastic_gradient_descent import StochasticGradientDescent,\
    StochasticGradientDescentRegression
from .logistic_regression import LogisticRegression
from .linear_regression import LinearRegression
from .mlp import MultiLayerPerceptron
from .median import Median
from .recursive_feature_elimination import RecursiveFeatureElimination,\
    RecursiveFeatureEliminationCV
