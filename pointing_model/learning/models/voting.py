from ..model_interface import SkLearnPointingMlModel
from sklearn import svm, neighbors, ensemble


class Voting(SkLearnPointingMlModel):

    def model(self, **kwargs):
        svm_params = {'C': 20000, 'gamma': 1e-3, 'kernel': 'rbf'}
        rfor_params = {
            'criterion': 'entropy', 'max_depth': 17, 'max_features': 'auto',
            'n_estimators': 150, 'random_state': 0
        }
        return ensemble.VotingClassifier([
            ('svm', svm.SVC(**svm_params)),
            ('rfor', ensemble.RandomForestClassifier(**rfor_params))
        ])

    @property
    def name(self):
        return 'Voting (SVM, KNN, RForest)'

    @property
    def short_name(self):
        return 'V_(SVM,KNN,RF)'
