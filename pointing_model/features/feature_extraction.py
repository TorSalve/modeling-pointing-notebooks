from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh import extract_relevant_features


def extract_features(X, y, config):
    return tsfresh_extraction(X, y, config)


def tsfresh_extraction(X, y, config):
    n_jobs = config['SVM-config']['n_jobs']
    extraction_settings = ComprehensiveFCParameters()
    return extract_relevant_features(
        X, y,
        n_jobs=n_jobs, fdr_level=0.01, show_warnings=False,
        column_id='id', column_sort='time',
        default_fc_parameters=extraction_settings
    )
