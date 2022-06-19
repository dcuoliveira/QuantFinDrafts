import os
import pickle

import numpy as np
import pandas as pd

from src.backtest import Strategy, Portfolio

class MovingAverageCrossStrategy(Strategy):
    """
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average."""

    def __init__(self,
                 symbol,
                 bars,
                 short_window=100,
                 long_window=400):
        self.symbol = symbol
        self.bars = bars

        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0

        # Create the set of short and long simple moving averages over the
        # respective periods
        signals['short_mavg'] = pd.rolling_mean(bars['Close'], self.short_window, min_periods=1)
        signals['long_mavg'] = pd.rolling_mean(bars['Close'], self.long_window, min_periods=1)

        # Create a 'signal' (invested or not invested) when the short moving average crosses the long
        # moving average, but only for the period greater than the shortest moving average window
        signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:]
            > signals['long_mavg'][self.short_window:], 1.0, 0.0)

        # Take the difference of the signals in order to generate actual trading orders
        signals['positions'] = signals['signal'].diff()

        return signals


class MarketOnClosePortfolio(Portfolio):
    """Encapsulates the notion of a portfolio of positions based
    on a set of signals as provided by a Strategy.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self,
                 bars=None,
                 vols=None,
                 signals=None,
                 carry=None,
                 position_type=None,
                 shift=0,
                 vol_target=10,
                 initial_capital=100000.0):
        self.bars = bars
        self.vols = vols
        self.signals = signals
        self.carry = carry

        self.initial_capital = float(initial_capital)
        self.positions_type = position_type
        self.shift = shift
        self.vol_target = vol_target

        self.positions = self.generate_positions()

    def generate_positions(self):
        if self.positions_type == "cs":
            self.rank = self.signals.rank(axis=1).shift(self.shift)
            upper_quantile = self.rank.quantile(q=0.8, axis=1)
            lower_quantile = self.rank.quantile(q=0.2, axis=1)

            positions = pd.DataFrame(np.where(self.rank < lower_quantile[:, None],
                                              -1,
                                              np.where(self.rank > upper_quantile[:, None],
                                                       1,
                                                       0)),
                                     columns=self.rank.columns,
                                     index=self.rank.index)

        if self.vols is not None:
            positions = positions.multiply((self.vol_target / self.vols.shift(self.shift)) * self.initial_capital, axis=0)
        else:
            positions = positions

        return positions

    def backtest_portfolio(self):

        portfolio_pnl = self.positions.multiply(self.bars.diff(), axis=0)

        return portfolio_pnl


if __name__ == "__main__":
    # # Obtain daily bars of AAPL from Yahoo Finance for the period
    # # 1st Jan 1990 to 1st Jan 2002 - This is an example from ZipLine
    # symbol = 'AAPL'
    # bars = DataReader(symbol, "yahoo", datetime.datetime(1990, 1, 1), datetime.datetime(2002, 1, 1))
    #
    # # Create a Moving Average Cross Strategy instance with a short moving
    # # average window of 100 days and a long window of 400 days
    # mac = MovingAverageCrossStrategy(symbol, bars, short_window=100, long_window=400)
    # signals = mac.generate_signals()

    file = open(os.path.join(os.getcwd(), "data/FXMM.pickle"), 'rb')
    data = signals = pickle.load(file)

    intrument_names = list(data['bars'].keys())  # [val for val in list(data['bars'].keys()) if val not in ["USDCOP", "USDCLP", "USDKRW"]]

    bars_dict = data['bars']
    bars = []
    for instrument in intrument_names:
        tmp_bar = bars_dict[instrument][['Close']]
        tmp_bar.columns = [instrument]
        bars.append(tmp_bar)
    bars = pd.concat(bars, axis=1)

    signals_dict = data['signals']['fast']
    signals = []
    for instrument in intrument_names:
        tmp_signal = pd.DataFrame(signals_dict[instrument].mean(axis=1),
                                  columns=[instrument])
        signals.append(tmp_signal)
    signals = pd.concat(signals, axis=1)

    vols_dict = data['vols']
    vols = []
    for instrument in intrument_names:
        tmp_vol = vols_dict[instrument][['Close']]
        tmp_vol.columns = [instrument]
        vols.append(tmp_vol)
    vols = pd.concat(vols, axis=1)

    carry_dict = data['vols']
    carry = []
    for instrument in intrument_names:
        tmp_carry = carry_dict[instrument][['Close']]
        tmp_carry.columns = [instrument]
        carry.append(tmp_carry)
    carry = pd.concat(carry, axis=1)

    portfolio = MarketOnClosePortfolio(bars=bars,
                                       vols=vols,
                                       signals=signals,
                                       carry=carry,
                                       position_type="cs",
                                       shift=1,
                                       initial_capital=1000000)
    returns = portfolio.backtest_portfolio()