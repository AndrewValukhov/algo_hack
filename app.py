import streamlit as st
import pandas as pd
from backtest import run_backtest

st.markdown('## Бэктест стратегий на скользящих средних')
st.markdown('### Условия экспериментов:')
st.write('Период бэктеста - с 1 января 2020 по 1 декабря 2023')
st.write('По умолчанию - экспоненциальные средние, периоды - 9/45 часов')
st.write('Сигналы от скользящих средних: когда быстрая "пробивает" снизу "медленную" - покупка')
st.write('Когда быстрая "пробивает" сверху "медленную" - продажа')
st.write('Таймфрейм - 1 час, по итогам каждого часа рассчитываем сигналы по средним')
st.write('Если есть сигнал на покупку или продажу, делаем сделку по цене открытия следующей часовой свечи')
st.write('Используем маржинальную торговлю, учёт всех комиссий и стоимости средств брокера')
st.write('Учитываем дивиденды, в зависимости от позиции на дату отсечки')
st.write('Тестирование алгоритма предполагает, что он принимает на вход даты дивидендных отсечек и выплаты на акцию')
st.write('Но с учётом того, что централизованно получить такие данные трудно, для тикеров отличных от SBER дивиденды не учитываются')
st.write('Не учитываем налоги - предположим, что торгуем на ИИС типа Б')


TICKER = 'SBER'
GAP_DATES = ['2020-10-01', '2021-05-10', '2023-05-08']
DIVIDENDS = [18.7, 18.7, 25]
path_to_rates = 'cbr_rate.csv'

ticker = st.text_input('Введите тикер интересующей акции', value="SBER", max_chars=4).upper()
exponential = st.checkbox('Использовать экспоненциальные средние', value=True)
fast = st.number_input("Введите период для быстрой средней", value=9, placeholder="Type a number...")
slow = st.number_input("Введите период для медленной средней", value=45, placeholder="Type a number...")
alloc = st.number_input("Плечо в лонг", value=1.5, help="На какую относительную величину от номинальной части портфеля мы покупаем акции в долг (1 значит используем все свои деньги, и не берем у брокера")
short_alloc = st.number_input("Плечо в шорт", value=0.5, help="На какую относительную величину от номинального портфеля мы будем шортиь акции")
comm_rate = st.number_input("Комиссия брокера за сделку, %", value=0.06, help="введенное значение разделится на 100")
margin_cost = st.number_input("Стоимость заёмных средств, %", value=17,
                              help="введенное значение разделится на 100")

if st.button('Запустить бэктест'):
    if ticker == 'SBER':
        g_dates = GAP_DATES
        divs = DIVIDENDS
    else:
        g_dates = GAP_DATES
        divs = pd.Series([0, 0, 0])
    met = run_backtest(ticker,
                       gap_dates=g_dates,
                       dividends=divs,
                       path_to_rates=path_to_rates,
                       fast=fast,
                       slow=slow,
                       exp=exponential,
                       allocation=alloc,
                       shortable_allocation=short_alloc,
                       initial_value=100000,
                       comission_rate=comm_rate / 100,
                       transfer_cost=0.0045 / 100,
                       margin_cost=margin_cost / 100 / 365)

    st.session_state.met = met.T
    st.session_state.met.columns = ['Значение метрики']


if 'met' in st.session_state:
    st.dataframe(st.session_state.met, width=450, height=530)
