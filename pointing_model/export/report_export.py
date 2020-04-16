from .stats_export import export as stats_export
from .figures_export import export as figures_export
from .ml_export import export as ml_export
import pointing_model.utils as utils


def export(point_model, **kwargs):
    figures_export(point_model, **kwargs)
    stats_export(point_model, **kwargs)
    point_model.analyze_all_features(base_pairplot=False)
    print("features:", utils.all_features())
    ml_export(point_model, **kwargs)
