import os

import pandas as pd

from prediction.model_training import forecast
from prediction.models import peLassoWrapper
from sklearn.metrics import mean_squared_error, make_scorer

import pickle
current_dir = os.getcwd()

if __name__ == '__main__':
    file = open(os.path.join(current_dir, 'data', "marcro_releases_data.pickle"), 'rb')
    data = pickle.load(file)
    signals = data['signals']

    data_list = []
    for release in list(signals.keys()):
        surprise = pd.DataFrame(signals[release]['surprise'].resample("B").mean().ffill().values,
                                columns=[release],
                                index=signals[release]['surprise'].resample("B").mean().ffill().index)
        data_list.append(surprise)
    data = pd.concat(data_list, axis=1).dropna()

    pred_results = forecast(data=data,
                            init_train_size=252,
                            steps=1,
                            target_name="DGNOCHNG Index",
                            scorer=make_scorer(mean_squared_error),
                            Wrapper=peLassoWrapper,
                            n_iter=10,
                            n_splits=5,
                            n_jobs=1,
                            seed=2294,
                            verbose=1,
                            max_lag=0)

