from sklearn import neural_network, multioutput
from ..model_interface import SkLearnPointingMlModel


class MultiLayerPerceptron(SkLearnPointingMlModel):

    def model(self, **kwargs):
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (50, 100, 50))
        activation = kwargs.get('activation', 'tanh')
        solver = kwargs.get('solver', 'adam')
        alpha = kwargs.get('alpha', 0.05)
        learning_rate = kwargs.get('learning_rate', 'adaptive')
        return neural_network.MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, activation=activation,
            solver=solver, alpha=alpha, learning_rate=learning_rate
        )

    @property
    def name(self):
        return 'MLP'

    @property
    def gridsearch_params(self):
        return [
            {
                'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant', 'adaptive'],
            }
        ]
