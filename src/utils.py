import pandas as pd

def add_lag(df,
            lags):
    
    lag_list = []
    for name in df.columns:
        for s in range(1, shifts + 1):
            tmp_lag_df = df[[name]].shift(shift)
            tmp_lag_df.rename(columns={name: name + "_lag" + str(s)})
            lag_list.append(tmp_lag_df)
    lag_df = pd.concat(lag_list, axis=1)
    
    df = pd.concat([df, lag_df], axis=1)
    return df

def add_diff(df,
             lags):
    
    diff_list = []
    for name in df.columns:
        for p in range(1, lags + 1):
            tmp_diff_df = df[[name]].diff(periods=p)
            tmp_diff_df.rename(columns={name: name + "_diff" + str(p)}, inplace=True)
            diff_list.append(tmp_diff_df)
    diff_df = pd.concat(diff_list, axis=1)
    
    df = pd.concat([df, diff_df], axis=1)
    return df