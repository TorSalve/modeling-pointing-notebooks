from pointing_model import learning


def export(point_model, **kwargs):
    validate = kwargs.get('validate', False)
    gridsearch = kwargs.get('gridsearch', False)
    kfold = kwargs.get('kfold', True)

    print('-- Classification')
    classification_models = {
        'Naive Bayes': learning.NaiveBayes,
        'KNN': learning.KNearestNeighbors,
        'SVM': learning.SupportVectorMachine,
        'RandomForest': learning.RandomForest,
        'Voting': learning.Voting,
        'Perceptron': learning.Perceptron,
        'SGD': learning.StochasticGradientDescent,
        'MLP': learning.MultiLayerPerceptron
    }

    for model_name in classification_models:
        model = classification_models[model_name]
        print('\n--------------------\n', model_name)
        point_model.machine_learning_training(
            model, validate=validate, gridsearch=gridsearch, kfold=kfold
        )

    print()
    print()

    print('-- Regression')
    regression_models = {
        'KNN': learning.KNearestNeighborsRegressor,
        'LinearRegression': learning.LinearRegression,
        # 'LogisticRegression': learning.LogisticRegression,
        'RandomForest': learning.RandomForestRegression,
        'SVM': learning.SupportVectorMachineRegression,
        'SGD': learning.StochasticGradientDescentRegression,
    }

    for model_name in regression_models:
        model = regression_models[model_name]
        print('\n--------------------\n', model_name)
        point_model.machine_learning_training(
            model, validate=validate, gridsearch=gridsearch, kfold=kfold
        )
