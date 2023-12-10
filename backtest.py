from moexalgo import Market, Ticker
import numpy as np
import pandas as pd
import streamlit as st

# Нам нужна функция запускатор, в которую все будет собираться
# в ней буду вызываться более мелкие функции


def preprocess(ticker, gap_dates, dividends):

    """Функция для препроцессинга данных"""

    sber = Ticker(ticker)
    df = pd.DataFrame(sber.candles(date='2020-01-01', till_date='2023-10-18', period='1h'))
    df_ = pd.DataFrame(sber.candles(date='2022-12-25', till_date='2023-11-30', period='1h'))

    df = pd.concat([df, df_]).drop_duplicates().reset_index(drop=True)

    df['candle_time'] = df['begin'].apply(lambda x: x.hour)
    df['candle_day'] = df['begin'].apply(lambda x: x.date())
    df['candle_day_str'] = df['candle_day'].apply(lambda x: str(x))
    df = df[df['candle_time'].isin([10, 11, 12, 13, 14, 15, 16, 17, 18])].reset_index(drop=True)

    if gap_dates:
        gap_dates = [pd.to_datetime(d).date() for d in gap_dates]

    dividends = pd.Series(dividends)

    df['payments_to_subtract'] = df['candle_day'].apply(lambda x: 3 if x <= gap_dates[0]
                                                                        else 2 if x <= gap_dates[1]
                                                                        else 1 if x <= gap_dates[2]
                                                                        else 0
                                                                        )

    def adjust_close(row, dividends):

        """Функция для корректировки цены акции на дивиденды"""
        close = row['close']
        pay = row['payments_to_subtract']

        return close - dividends[len(dividends) - pay: len(dividends)].sum()

    df['adjusted_close'] = df.apply(adjust_close, axis=1, args=[dividends])
    df.drop(columns=['high', 'low', 'value', 'volume'], inplace=True)

    return df


def calc_mas(df, fast=9, slow=45, exp=True):

    """Функция для добавления двух скользящих средниx"""

    if exp:
        df[f'm{fast}'] = df[['adjusted_close']].ewm(span=fast, adjust=False).mean()
        df[f'm{slow}'] = df['adjusted_close'].ewm(span=slow, adjust=False).mean()
    else:
        df[f'm{fast}'] = df['adjusted_close'].rolling(fast).mean()
        df[f'm{slow}'] = df['adjusted_close'].rolling(slow).mean()

    return df


def get_signals(df, fast=9, slow=45):

    """Функция для получения сигналов от взаимодействия двух скользящих средних"""

    def ma_relative(row, short, long):
        short = row[short]
        long = row[long]

        if short > long:
            return 'upside'
        elif long > short:
            return 'downside'
        elif long == short:
            return 'equal'
        else:
            return 'unknown'

    df[f'ma_relative_{fast}_{slow}'] = df.apply(ma_relative, axis=1, args=[f'm{fast}', f'm{slow}'])
    df[f'ma_relative_{fast}_{slow}_shifted'] = df[f'ma_relative_{fast}_{slow}'].shift()

    def events(row, rel, rel_shifted):
        now = row[rel]
        prev = row[rel_shifted]
        if (now == 'upside') and (prev == 'downside'):
            return 'buy_signal'
        elif now == 'downside' and prev == 'upside':
            return 'sell_signal'
        return 'no signal'

    df['event'] = df.apply(events, axis=1, args=[f'ma_relative_{fast}_{slow}',
                                                 f'ma_relative_{fast}_{slow}_shifted'])
    df['event_shifted'] = df['event'].shift()
    df = df[['candle_day', 'end', 'event_shifted', 'close', 'open']].values

    return df


def imitate_trades(df, allocation=1.5, shortable_allocation=0.5, initial_value=100000, comission_rate=0.06/100,
                   transfer_cost=0.0045/100, margin_cost=0.17/365, gap_dates=None, dividends=None):


    share_position = [0]
    money = [initial_value]
    dates = [df[0][0].isoformat()]
    change_day = False
    total_cost_of_margin = [0]

    if gap_dates:
        gds = [str(gd) for gd in gap_dates]

    date_divs_dict = dict(zip(gds, dividends))

    for i in range(len(df)):
        # пропускаем первый день - у нас заданы начальные параметры
        if i == 0:
            continue

        date, candle_time, signal, close, open = df[i]
        date = date.isoformat()
        if dates[i - 1] != date:
            delta = (pd.to_datetime(date) - pd.to_datetime(dates[i - 1])).days
            # print(dates[i-1], date, delta)
            change_day = True
        else:
            change_day = False

        dates.append(date)

        # *_signal - у нас с задержкой, поэтому, когда мы его получили,
        # то сразу должны выполнить операцию по текущей цене open
        # если сигнал на покупку, то с аллокацией 150% денег покупаем акции
        if signal == 'buy_signal':

            short_cover_amount = share_position[i - 1] * open
            money_to_long = money[i - 1] + short_cover_amount

            shares = int(money_to_long * allocation / open)
            long_amount = shares * open

            comission = long_amount * comission_rate
            short_cover_comission = -short_cover_amount * comission_rate
            total_comission = comission + short_cover_comission

            share_position.append(shares)
            money.append(money_to_long - long_amount - total_comission)

        # если сигнал на продажу, то закрываем позицию по текущей цене open
        # и открываем шорт на 50% получившейся суммы по этой же цене
        elif signal == 'sell_signal':

            close_long = share_position[i - 1] * open
            money_from_long = money[i - 1] + close_long

            shortable_shares = int(money_from_long * shortable_allocation / open)
            money_to_short = shortable_shares * open

            selling_long_comission = close_long * comission_rate
            selling_short_comission = money_to_short * comission_rate
            total_comission = selling_long_comission + selling_short_comission

            shares = -shortable_shares
            share_position.append(shares)
            total_money = money_from_long + money_to_short - total_comission
            money.append(total_money)

        # если никаких сигналов нет, то сохраняем позицию
        else:
            money.append(money[i - 1])
            share_position.append(share_position[i - 1])

        if not change_day:
            total_cost_of_margin.append(0)
        else:
            if share_position[i] < 0:
                transfer = share_position[i] * close * transfer_cost
                margin = share_position[i] * close * margin_cost
                total_cost_of_margin.append(transfer + margin)
                money[i] = money[i] + transfer + margin
            elif share_position[i] > 0:
                margin = money[i] * (margin_cost + transfer_cost)
                total_cost_of_margin.append(margin)
                money[i] = money[i] + margin
            else:
                total_cost_of_margin.append(0)

        if date in gds:
            if candle_time.hour == 18:
                dividends_flow = share_position[i] * date_divs_dict[date]
                money[i] += dividends_flow * 0.87
                print(date, candle_time, share_position[i], dividends_flow, money[i - 1], money[i])

    result = pd.DataFrame(zip(df, money, share_position, total_cost_of_margin))
    result.columns = ['decision_data', 'money', 'shares', 'total_cost_of_margin']
    result['close'] = result['decision_data'].apply(lambda x: x[3])
    result['total_value'] = result['close'] * result['shares'] + result['money']
    result['date'] = result['decision_data'].apply(lambda x: x[1])

    return result.drop(columns=['decision_data'])


def geometric_mean(returns: pd.Series) -> float:
    returns = returns.fillna(0) + 1
    if np.any(returns <= 0):
        return 0
    return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1


def get_annual_return(result, annual_trading_days=252):
    day_returns = result[['total_value']].set_index(result['date']).resample('D').last().dropna().pct_change()
    gmean_day_return = geometric_mean(day_returns)
    annualized_return = (1 + gmean_day_return) ** annual_trading_days - 1
    return annualized_return


def get_annualized_volatility(result, path_to_rates, annual_trading_days=252):
    rf_data = pd.read_csv(path_to_rates, sep=';', header=None)
    rf_data.columns = ['date', 'rate']
    rf_data['date'] = pd.to_datetime(rf_data['date'])
    rf_data['rate'] = rf_data['rate'].apply(lambda x: float(x.replace(',', '.')))
    rf_data = rf_data.set_index(rf_data['date'])[['rate']]
    day_returns = result[['total_value']].set_index(result['date']).resample('D').last().dropna().pct_change()
    merged = day_returns.merge(rf_data, left_on=day_returns.index, right_on=rf_data.index, how='left')
    merged = merged.fillna(method='ffill').fillna(method='bfill')
    merged['daily_rate'] = merged['rate'] / 252 / 100
    merged['excess_return'] = merged['total_value'] - merged['daily_rate']

    sharpe_ratio = np.mean(merged['excess_return']) / np.std(merged['excess_return'])
    annual_factor = np.sqrt(252)
    sharpe_ratio_annualized = sharpe_ratio * annual_factor

    neg = merged[merged['excess_return'] < 0]
    sortino_ratio = np.mean(merged['excess_return']) / np.std(neg['excess_return'])
    sortino_ratio_annualized = sortino_ratio * annual_factor

    return np.std(merged['excess_return']) * np.sqrt(
        annual_trading_days), sharpe_ratio_annualized, sortino_ratio_annualized


def max_drawdown(result):
    dd = 1 - result['total_value'] / np.maximum.accumulate(result['total_value'])
    max_dd = -np.nan_to_num(dd.max())
    return max_dd


def calmar(max_dd, annualized_return):
    calmar_ratio = annualized_return / (-max_dd or np.nan)
    return calmar_ratio


def calc_stats(res, dividends, df, path_to_rates, annual_trading_days=252):
    dividends = pd.Series(dividends)
    metrics = {}
    metrics['Начальная дата'] = str(res.iloc[0, -1])
    metrics['Конечная дата'] = str(res.iloc[-1, -1])
    metrics['Длина периода тестирования'] = str(res.iloc[-1, -1] - res.iloc[0, -1])
    metrics['Время в позиции, %'] = float(f"{len(res[res['shares'] != 0]) / len(res) * 100:.2f}")
    metrics['Итоговая величина портфеля, руб.'] = int(res.iloc[-1, -2])
    metrics['Максимальная величина портфеля'] = int(res['total_value'].max())
    metrics['Совокупная доходность, %'] = int(res.iloc[-1, -2] / res.iloc[0, -2] * 100)
    metrics['Доходность стратегии buy and hold, %'] = int(int(res.iloc[0, -2] / df.loc[0, 'open']
                                                             ) * (df.iloc[-1, 1] + dividends.sum() * 0.87)
                                                             / res.iloc[0, -2] * 100)
    annualized_return = get_annual_return(res)[0]
    metrics['Среднегодовая доходность, %'] = np.round(annualized_return * 100, 2)
    vol, sharpe, sortino = get_annualized_volatility(res, path_to_rates)
    metrics['Среднегодовая волатильность, %'] = float(f'{vol * 100:.2f}')
    metrics['Sharpe-ratio'] = float(f'{sharpe:.2f}')
    metrics['Sortino-ratio'] = float(f'{sortino:.2f}')
    metrics['Максимальная просадка, %'] = np.round(max_drawdown(res), 4) * 100
    metrics['Calmar ratio'] = np.round(calmar(max_drawdown(res), annualized_return), 2)
    return metrics


def run_backtest(ticker,
                 gap_dates,
                 dividends,
                 path_to_rates,
                 fast=9,
                 slow=45,
                 exp=True,
                 allocation=1.5,
                 shortable_allocation=0.5,
                 initial_value=100000,
                 comission_rate=0.06 / 100,
                 transfer_cost=0.0045 / 100,
                 margin_cost=0.17 / 365
                 ):
    st.write('Скачиваем данные')
    df = preprocess(ticker, gap_dates, dividends)
    st.write('Рассчитываем скользящие средние')
    df = calc_mas(df, fast=fast, slow=slow, exp=exp)
    st.write('Получаем сигналы')
    data = get_signals(df, fast, slow)
    st.write('Имитируем сделки')
    res = imitate_trades(data,
                         allocation=allocation,
                         shortable_allocation=shortable_allocation,
                         comission_rate=comission_rate,
                         transfer_cost=transfer_cost,
                         margin_cost=margin_cost,
                         gap_dates=gap_dates,
                         dividends=dividends)
    st.write('Считаем метрики')
    met = calc_stats(res, dividends, df, path_to_rates)
    return pd.DataFrame.from_dict(met, orient='index').T
