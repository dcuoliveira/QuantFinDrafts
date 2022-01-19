import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

def hyper_params_search(df,
                        target_name,
                        scorer,
                        wrapper,
                        n_iter,
                        n_splits,
                        n_jobs,
                        verbose,
                        seed):
    """
    Use the dataframe 'df' to search for the best
    params for the model 'wrapper'.
    The CV split is performed using the TimeSeriesSplit
    class.
    We can define the size of the test set using the formula
    ``n_samples//(n_splits + 1)``,
    where ``n_samples`` is the number of samples. Hence,
    we can define
    n_splits = (n - test_size) // test_size
    :param df: train data
    :type df: pd.DataFrame
    :param wrapper: predictive model
    :type wrapper: sklearn model wrapper
    :param n_iter: number of hyperparameter searchs
    :type n_iter: int
    :param n_splits: number of splits for the cross-validation
    :type n_splits: int
    :param n_jobs: number of concurrent workers
    :type n_jobs: int
    :param verbose: param to print iteration status
    :type verbose: bool, int
    :param target_name: name of the target column in 'df'
    :type target_name: str
    :return: R2 value
    :rtype: float
    """

    X = df.drop(target_name, 1).values
    y = df[target_name].values

    time_split = TimeSeriesSplit(n_splits=n_splits)

    if wrapper.search_type == 'random':
        model_search = RandomizedSearchCV(estimator=wrapper.ModelClass,
                                          param_distributions=wrapper.param_grid,
                                          n_iter=n_iter,
                                          cv=time_split,
                                          verbose=verbose,
                                          n_jobs=n_jobs,
                                          scoring=scorer,
                                          random_state=seed)
    elif wrapper.search_type == 'grid':
        model_search = GridSearchCV(estimator=wrapper.ModelClass,
                                    param_grid=wrapper.param_grid,
                                    cv=time_split,
                                    verbose=verbose,
                                    n_jobs=n_jobs,
                                    scoring=scorer)
    else:
        raise Exception('search type method not registered')

    model_search = model_search.fit(y=y,
                                    X=X)

    return model_search


def time_series_cross_validation(data,
                                 init_train_size,
                                 steps,
                                 scorer,
                                 Wrapper,
                                 n_iter,
                                 n_splits,
                                 n_jobs,
                                 verbose,
                                 seed,
                                 target_name):
    """
     We recursively increase the training sample, periodically refitting
     the entire model once per period, and making
     out-of-sample predictions for the subsequent year.
     On each fit, to perform hyperparameter search,
     we perform cross-validation on a rolling basis.

     :param df: train and test data combined
     :type df: pd.DataFrame
     :param Wrapper: predictive model class
     :type Wrapper: sklearn model wrapper class
     :param n_iter: number of hyperparameter searchs
     :type n_iter: int
     :param n_splits: number of splits for the cross-validation
     :type n_splits: int
     :param n_jobs: number of concurrent workers
     :type n_jobs: int
     :param verbose: param to print iteration status
     :type verbose: bool, int
     :param target_name: name of the target column in 'df'
     :type target_name: str
     :return: dataframe with the date, true return
              and predicted return.
     :rtype: pd.DataFrame
     """

    all_preds = []
    model_wrapper = Wrapper()
    for t in tqdm(range(0, int((data.shape[0] - init_train_size)/steps + 1)),
                  disable=not verbose,
                  desc="annual training and prediction"):
        # train data
        train_data = data[0:init_train_size + t]

        # test data
        test_data = data[init_train_size:(init_train_size + t) + steps]

        if model_wrapper.model_name == "pelasso":
            lasso_search = hyper_params_search(df=train_data,
                                               target_name=target_name,
                                               scorer=scorer,
                                               wrapper=model_wrapper.Lasso,
                                               n_jobs=n_jobs,
                                               n_splits=n_splits,
                                               n_iter=n_iter,
                                               seed=seed,
                                               verbose=verbose)
            lasso_coefs_df = pd.DataFrame(lasso_search.best_estimator_.coef_, index=train_data.drop(target_name, axis=1).columns)
            lasso_coefs = list(lasso_coefs_df[lasso_coefs_df != 0].dropna().index)

            ridge_search = hyper_params_search(df=train_data[[target_name] + lasso_coefs],
                                               target_name=target_name,
                                               scorer=scorer,
                                               wrapper=model_wrapper.Ridge,
                                               n_jobs=n_jobs,
                                               n_splits=n_splits,
                                               n_iter=n_iter,
                                               seed=seed,
                                               verbose=verbose)

            test_pred = ridge_search.best_estimator_.predict(test_data.drop(target_name, axis=1)[lasso_coefs].values)
        else:
            model_search = hyper_params_search(df=train_data,
                                               scorer=scorer,
                                               wrapper=model_wrapper,
                                               n_jobs=n_jobs,
                                               n_splits=n_splits,
                                               n_iter=n_iter,
                                               seed=seed,
                                               verbose=verbose)
            test_pred = model_search.best_estimator_.predict_proba(test_data.values)[:, 1]

        dict_ = {"date": test_data.index,
                 target_name: test_data[target_name],
                 "prediction": test_pred}
        result = pd.DataFrame(dict_)
        all_preds.append(result)

    pred_results = pd.concat(all_preds).reset_index(drop=True)
    return pred_results


def forecast(data,
             init_train_size,
             steps,
             max_lag,
             target_name,
             scorer,
             Wrapper,
             n_iter,
             n_splits,
             n_jobs,
             seed,
             verbose=1):
    """
    Function to perform the predition using one ticker,
    one feature selection method, and one prediction model.

    :param data: data
    :type data: pd.DataFrame
    :param ticker_name: ticker name (without extension)
    :type ticker_name: str
    :param fs_method: folder with feature selection
                      results
    :type fs_method: str
    :param Wrapper: predictive model class
    :type Wrapper: sklearn model wrapper class
    :param n_iter: number of hyperparameter searchs
    :type n_iter: int
    :param n_splits: number of splits for the cross-validation
    :type n_splits: int
    :param n_jobs: number of concurrent workers
    :type n_jobs: int
    :param verbose: param to print iteration status
    :type verbose: bool, int
    :param target_name: name of the target column in 'df'
    :type target_name: str
    :param max_lag: maximun number of lags
    :type max_lag: int
    :return: dataframe with the date, true return
            and predicted return.
    :rtype: pd.DataFrame
    """

    if max_lag != 0:
        pass

    pred_results = time_series_cross_validation(data=data,
                                                init_train_size=init_train_size,
                                                steps=steps,
                                                scorer=scorer,
                                                Wrapper=Wrapper,
                                                n_iter=n_iter,
                                                n_jobs=n_jobs,
                                                n_splits=n_splits,
                                                target_name=target_name,
                                                seed=seed,
                                                verbose=verbose)

    return pred_results
