from sklearn.naive_bayes import GaussianNB
from ..model_interface import SkLearnPointingMlModel


class NaiveBayes(SkLearnPointingMlModel):

    def model(self, **kwargs):
        return GaussianNB()

    @property
    def name(self):
        return 'Naive Bayes'

    @property
    def gridsearch_params(self):
        return [
            {
                'var_smoothing': [1e-9]
            }
        ]
