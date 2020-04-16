from sklearn import linear_model, multioutput
from ..model_interface import SkLearnPointingMlModel


class LinearRegression(SkLearnPointingMlModel):

    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, _type='regression', **kwargs)

    def model(self, **kwargs):
        return multioutput.MultiOutputRegressor(
            linear_model.LinearRegression()
        )

    @property
    def name(self):
        return 'Linear Regression'
