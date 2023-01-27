import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from tqdm import tqdm
from pandas.tseries.offsets import BDay
import pickle
import os

plt.style.use("bmh")

### funcs ###

def check_available_instruments(instruments: list,
                                current_dt: pd.DatetimeIndex,
                                bars_df: pd.DataFrame,
                                vols_df: pd.DataFrame,
                                forecasts_df: pd.DataFrame,
                                carrys_df: pd.DataFrame,
                                rets_df: pd.DataFrame):
    valid_instruments = []
    for inst in instruments:
        check1 = current_dt in bars_df.loc[:, "{} close".format(inst)].dropna().index
        check2 = current_dt - dt.timedelta(1) in bars_df.loc[:, "{} close".format(inst)].dropna().index
        check3 = current_dt in vols_df.loc[:, "{} daily ret % vol".format(inst)].dropna().index
        check4 = current_dt in forecasts_df.loc[:, "{} forecasts".format(inst)].dropna().index
        check5 = current_dt in rets_df.loc[:, "{} ret%".format(inst)].dropna().index
        check6 = current_dt in carrys_df.loc[:, "{} carry".format(inst)].dropna().index

        if check1 and check2 and check3 and check4 and check5 and check6:
            valid_instruments.append(inst)

    return valid_instruments

### funcs ###

bar_name = "Close"
vol_window = 90
resample_freq = "D"
vol_target = 0.1
capital = 1000
instruments = ['USDBRL', 'USDCLP', 'USDZAR', 'USDMXN', "USDCOP",
               "USDEUR", "USDJPY", "USDAUD", "USDNZD", "USDCAD", "USDGBP", "USDCHF",
               "USDSEK", "USDNOK",
               "USDHUF", "USDPLN", "USDCZK"]

file = open(os.path.join(os.getcwd(), "src", "data", "fxmm.pickle"), 'rb')
target_dict = pickle.load(file)
bars_info = target_dict["bars"]
carry_info = target_dict["carry"]
signals_info_fast = target_dict["signals"]["fast"]
signals_info_slow = target_dict["signals"]["slow"]

bars_info["USDBRL"] = bars_info["WDO1"]
carry_info["USDBRL"] = carry_info["WDO1"]
signals_info_fast["USDBRL"] = signals_info_fast["WDO1"]
signals_info_slow["USDBRL"] = signals_info_slow["WDO1"]

forecasts = []
bars = []
vols = []
rets = []
carrys = []
signals = []
for inst in instruments:
    tmp_bars = bars_info[inst][[bar_name]].resample("D").last().ffill()

    tmp_rets = tmp_bars.pct_change()
    tmp_vols = tmp_rets.rolling(window=vol_window).std()

    # tmp_signals = (tmp_bars - tmp_bars.rolling(window=90).mean())
    tmp_signals = (signals_info_fast[inst].resample("D").last().ffill() + signals_info_slow[inst].resample("D").last().ffill()) / 2
    tmp_forecasts = pd.DataFrame(np.where(tmp_signals.sum(axis=1) > 0, 1, -1),
                                 columns=["{} forecasts".format(inst)],
                                 index=tmp_signals.sum(axis=1).index)
    
    tmp_carry = carry_info[inst][[bar_name]].resample("D").last().ffill()

    bars.append(tmp_bars.rename(columns={"Close": "{} close".format(inst)}))
    vols.append(tmp_vols.rename(columns={"Close": "{} daily ret % vol".format(inst)}))
    rets.append(tmp_rets.rename(columns={"Close": "{} ret%".format(inst)}))
    carrys.append(tmp_carry.rename(columns={"Close": "{} carry".format(inst)}))
    signals.append(tmp_signals.rename(columns={"Close": "{} signals".format(inst)}))
    forecasts.append(tmp_forecasts)

bars_df = pd.concat(bars, axis=1)
vols_df = pd.concat(vols, axis=1)
rets_df = pd.concat(rets, axis=1)
forecasts_df = pd.concat(forecasts, axis=1)
carrys_df = pd.concat(carrys, axis=1)

backtest_dates = forecasts_df.dropna().index
portfolio_df = pd.DataFrame(index=backtest_dates)

first_day = True
for t in tqdm(backtest_dates, desc="Running Backtest", total=len(backtest_dates)):

    if first_day:
        portfolio_df.loc[t, "capital"] = capital
        portfolio_df.loc[t, "nominal"] = 0
        valid_instruments = instruments.copy()
    else:
        valid_instruments = check_available_instruments(instruments,
                                                        t,
                                                        bars_df,
                                                        vols_df,
                                                        forecasts_df,
                                                        carrys_df,
                                                        rets_df)

    if len(valid_instruments) == 0:
        portfolio_df.loc[t, "capital"] = portfolio_df.loc[t - dt.timedelta(1), "capital"]

    # compute vol. adjusted positions from forecasts and total nominal exposure
    nominal_total = 0
    for inst in valid_instruments:
        
        if first_day:
            portfolio_df.loc[t, "{} position units".format(inst)] = 0
            portfolio_df.loc[t, "{} w".format(inst)] = 0
            portfolio_df.loc[t, "{} leverage".format(inst)] = 0
        else:
            # separate all inputs needed for the calculations
            price_change = bars_df.loc[t, "{} close".format(inst)] - bars_df.loc[t - dt.timedelta(1), "{} close".format(inst)]
            price = bars_df.loc[t, "{} close".format(inst)] 
            previous_capital = portfolio_df.loc[t - dt.timedelta(1), "capital"]
            inst_daily_ret_vol = vols_df.loc[t, "{} daily ret % vol".format(inst)]
            forecast = forecasts_df.loc[t, "{} forecasts".format(inst)]

            # invert price to local currency if needed
            convert_factor = (1 / price) if inst.split("=")[0][3:] != "USD" else 1

            # compute position vol. target in the local currency and the instrument daily vol. in the local currency as well
            position_vol_target = (previous_capital / len(inst)) * vol_target * (1 / np.sqrt(252))
            inst_daily_price_vol = price * inst_daily_ret_vol * convert_factor
            position_units = forecast * position_vol_target / inst_daily_price_vol 

            # save position units (e.g. cts, notional in USD etc)
            portfolio_df.loc[t, "{} position units".format(inst)] = position_units
            nominal_total += abs(position_units * bars_df.loc[t, "{} close".format(inst)])

    if not first_day:

        # compute instrument weight exposure (we are going to use them below to compute nominal return)
        for inst in valid_instruments:
            previous_price = bars_df.loc[t - dt.timedelta(1), "{} close".format(inst)]
            position_units = portfolio_df.loc[t, "{} position units".format(inst)]

            nominal_inst = position_units * previous_price
            inst_w = nominal_inst / nominal_total
            portfolio_df.loc[t, "{} w".format(inst)] = inst_w

        # compute pnl of the last positions, if any
        pnl = 0
        nominal_ret = 0
        for inst in instruments:
            previos_positions_units = portfolio_df.loc[t - dt.timedelta(1), "{} position units".format(inst)]

            if (previos_positions_units != 0) and (not pd.isna(previos_positions_units)):
                price_change = bars_df.loc[t, "{} close".format(inst)] - bars_df.loc[t - dt.timedelta(1), "{} close".format(inst)]

                # convert price change to local currency when needed
                convert_factor = (1 / bars_df.loc[t, "{} close".format(inst)] ) if inst.split("=")[0][3:] != "USD" else 1
                local_currency_change = price_change * convert_factor

                # compute carry differential and pay/recieve it
                buy_sell_sign = np.sign(previos_positions_units)
                carry = carrys_df.loc[t, "{} carry".format(inst)]
                inst_carry_pnl = ((previos_positions_units * (1 + carry * buy_sell_sign)) - previos_positions_units) *  buy_sell_sign

                # compute pnl
                inst_pnl = local_currency_change * previos_positions_units + inst_carry_pnl
                portfolio_df.loc[t, "{} pnl".format(inst)] = inst_pnl

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