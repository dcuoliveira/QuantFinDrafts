import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from tqdm import tqdm
from pandas.tseries.offsets import BDay

plt.style.use("bmh")

bar_name = "Close"
vol_window = 90
resample_freq = "D"
vol_target = 0.1

capital = 1000

instruments = ["USDCAD=X", "USDJPY=X", "GBPUSD=X", "EURUSD=X", "USDNOK=X", "USDSEK=X", "AUDUSD=X", "NZDUSD=X"]

bars = []
vols = []
rets = []
for instrument in instruments:
    instrument_info = yf.Ticker(instrument)
    hist = instrument_info.history(period="max")[[bar_name]].resample(resample_freq).ffill().dropna()
    hist.index = pd.to_datetime([dtref.strftime("%Y-%m-%d") for dtref in hist.index])
    hist = hist.loc["2007-01-01":]

    bars.append(hist.rename(columns={bar_name: "{instrument} {bar_name}".format(instrument=instrument, bar_name=bar_name)}))
    rets.append(hist.pct_change().rename(columns={bar_name: "{instrument} ret%".format(instrument=instrument)}))
    vols.append((hist.pct_change().rolling(window=vol_window).std()).rename(columns={bar_name: "{instrument} daily ret % vol".format(instrument=instrument)}))

bars_df = pd.concat(bars, axis=1).resample(resample_freq).ffill().dropna()
vols_df = pd.concat(vols, axis=1).resample(resample_freq).ffill().dropna()
rets_df = pd.concat(rets, axis=1).resample(resample_freq).ffill().dropna()

signals = []
forecasts = []
for instrument in instruments:
    target_name = "{instrument} {bar_name}".format(instrument=instrument, bar_name=bar_name)

    tmp_signals = pd.DataFrame((bars_df[target_name] - bars_df[target_name].rolling(window=200).mean()).values,
                                columns=["{instrument} Signal".format(instrument=instrument)],
                                index=bars_df.index)
    signals.append(tmp_signals)
    
    tmp_forecasts = (bars_df[target_name] - bars_df[target_name].rolling(window=200).mean())
    tmp_forecasts = pd.DataFrame(np.where(tmp_forecasts > 0, 1, -1),
                                 columns=["{instrument} Forecasts".format(instrument=instrument)],
                                 index=bars_df.index)
    forecasts.append(tmp_forecasts)

signals_df = pd.concat(signals, axis=1)
forecasts_df = pd.concat(forecasts, axis=1)

backtest_dates = signals_df.dropna().index
portfolio_df = pd.DataFrame(index=backtest_dates)

first_day = True
for t in tqdm(backtest_dates, desc="Running Backtest", total=len(backtest_dates)):

    if first_day:
        portfolio_df.loc[t, "capital"] = capital
        portfolio_df.loc[t, "nominal"] = 0

    # compute vol. adjusted positions from forecasts and total nominal exposure
    nominal_total = 0
    for inst in instruments:
        
        if first_day:
            portfolio_df.loc[t, "{} position units".format(inst)] = 0
            portfolio_df.loc[t, "{} w".format(inst)] = 0
            portfolio_df.loc[t, "{} leverage".format(inst)] = 0
        else:
            # separate all inputs needed for the calculations
            price_change = bars_df.loc[t, "{} Close".format(inst)] - bars_df.loc[t - dt.timedelta(1), "{} Close".format(inst)]
            price = bars_df.loc[t, "{} Close".format(inst)] 
            previous_capital = portfolio_df.loc[t - dt.timedelta(1), "capital"]
            inst_daily_ret_vol = vols_df.loc[t, "{} daily ret % vol".format(inst)]
            forecast = forecasts_df.loc[t, "{} Forecasts".format(inst)]
            previos_positions_units = portfolio_df.loc[t - dt.timedelta(1), "{} position units".format(inst)]

            # invert price to local currency if needed
            convert_factor = (1 / price) if inst.split("=")[0][3:] != "USD" else 1

            # compute position vol. target in the local currency and the instrument daily vol. in the local currency as well
            position_vol_target = (previous_capital / len(inst)) * vol_target * (1 / np.sqrt(252))
            inst_daily_price_vol = price * inst_daily_ret_vol * convert_factor
            position_units = forecast * position_vol_target / inst_daily_price_vol 

            # save position units (e.g. cts, notional in USD etc)
            portfolio_df.loc[t, "{} position units".format(inst)] = position_units
            nominal_total += abs(position_units * bars_df.loc[t, "{} Close".format(inst)])

    if not first_day:

        # compute instrument weight exposure (we are going to use them below to compute nominal return)
        for inst in instruments:
            previous_price = bars_df.loc[t - dt.timedelta(1), "{} Close".format(inst)]
            position_units = portfolio_df.loc[t, "{} position units".format(inst)]

            nominal_inst = position_units * previous_price
            inst_w = nominal_inst / nominal_total
            portfolio_df.loc[t, "{} w".format(inst)] = inst_w

        # compute pnl of the last positions, if any
        pnl = 0
        nominal_ret = 0
        for inst in instruments:
            if previos_positions_units != 0:
                price_change = bars_df.loc[t, "{} Close".format(inst)] - bars_df.loc[t - dt.timedelta(1), "{} Close".format(inst)]
                convert_factor = (1 / bars_df.loc[t, "{} Close".format(inst)] ) if inst.split("=")[0][3:] != "USD" else 1
                local_currency_change = price_change * convert_factor
                inst_pnl = local_currency_change * portfolio_df.loc[t - dt.timedelta(1), "{} position units".format(inst)]
                pnl += inst_pnl
                nominal_ret += portfolio_df.loc[t - dt.timedelta(1), "{} w".format(inst)] * rets_df.loc[t, "{} ret%".format(inst)]

        capital_ret = nominal_ret * portfolio_df.loc[t - dt.timedelta(1), "leverage"]
        portfolio_df.loc[t, "capital"] = portfolio_df.loc[t - dt.timedelta(1), "capital"] + pnl
        portfolio_df.loc[t, "daily pnl"] = pnl
        portfolio_df.loc[t, "nominal ret"] = nominal_ret
        portfolio_df.loc[t, "capital ret"] = capital_ret 
    
    portfolio_df.loc[t, "nominal"] = nominal_total
    portfolio_df.loc[t, "leverage"] = nominal_total / portfolio_df.loc[t, "capital"]

    first_day = False

fim = 1