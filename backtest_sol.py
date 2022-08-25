import json
import os
import random
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import csv

import numpy as np
import pandas as pd
import pandas_ta as ta

import operator
from geneticalgorithm import geneticalgorithm as ga

# from deap import base
# from deap import creator
# from deap import gp
# from deap import tools
# from deap import algorithms
import multiprocessing

# ultimo supporto riscontrato secondo metodologia Wyckoff semplificata
last_support = 0

# ultima resistenza riscontrata secondo metodologia Wyckoff semplificata
last_resistance = 0

last_order_price = 0
real_last_order_price = 0
last_order_qty = 0
last_price = 0
last_entry_price = 0

treshold_price = 0.00001
positions = []
order_qty = 0
price_change_average_lenght = 10
last_price_change = 0
fees = 0.05
current_leverage = 25
unrealizedPnlPcnt = 0
initial_balance = 100000000
available_balance = initial_balance
max_unrealized_pnl = 0
min_unrealized_pnl = 0

max_leverage = 20
short_long = 'l'
pct_new_level = 100.5
current_order = {
    "quantity": 0,
    "entry_price": 0,
    "sat_qty": 0,
    "fee_paid": 0,
}
average = 0
average_neg = 0
old_min_alpha = 0
old_min_sr = 0
old_min_fr = 0

ticks_1m = []
ticks_5m = []
calc_sr = {}
calc_alpha = {}
trades = []
funding_rates = []
id_trade = -1
# current_time = 0
month = 1
alpha = 0
alphaneg = 0
funding_rate = 0

orders = []

# n° candele da analizzare per il calcolo del gamma positivo
N_TICKS_DELTA = 8
# n° candele da analizzare per il calcolo del gamma negativo
N_TICKS_DELTA_NEG = 100
# coefficente per il take profit
PCT_VOLUME_DISTANCE = 0.4
# coefficente per il rebuy
PCT_VOLUME_DISTANCE_NEGATIVE = 4
MULT_NEG_DELTA = 32

pct_distance = 100

N_REBUY = 7
PCT_STOP_LOSS = -3
DEFAULT_PCT_STOP_LOSS = -15
step_rebuy = 2
PCT_QTY_ORDER = 10


def convertDateToTime(date):
    return datetime.strptime(date, '%Y-%m-%d %H:%M:%S')


def write_orders():
    global orders
    global treshold_price
    global N_TICKS_DELTA
    global N_TICKS_DELTA_NEG
    global PCT_VOLUME_DISTANCE
    global PCT_VOLUME_DISTANCE_NEGATIVE
    global N_REBUY
    global PCT_STOP_LOSS
    global MULT_NEG_DELTA
    global DEFAULT_PCT_STOP_LOSS
    global PCT_QTY_ORDER
    global rand_number
    global initial_balance
    ord = pd.DataFrame(orders)
    ord.to_csv("orders/" + short_long + "_" + str(rand_number) + ".csv", header=False)
    f = open("orders/" + short_long + "_" + str(rand_number) + ".csv", 'a')
    f.write("\n\n")
    f.close()

    f = open("orders/" + short_long + "_" + str(rand_number) + ".txt", 'w')
    f.write("N_TICKS_DELTA: " + str(N_TICKS_DELTA) + "\n")
    f.write("N_TICKS_DELTA_NEG: " + str(N_TICKS_DELTA_NEG) + "\n")
    f.write("PCT_VOLUME_DISTANCE: " + str(PCT_VOLUME_DISTANCE) + "\n")
    f.write("PCT_VOLUME_DISTANCE_NEGATIVE: " + str(PCT_VOLUME_DISTANCE_NEGATIVE) + "\n")
    f.write("N_REBUY: " + str(N_REBUY) + "\n")
    f.write("PCT_STOP_LOSS: " + str(PCT_STOP_LOSS) + "\n")
    f.write("treshold_price: " + str(treshold_price) + "\n")
    f.write("MULT_NEG_DELTA: " + str(MULT_NEG_DELTA) + "\n")
    f.write("DEFAULT_PCT_STOP_LOSS: " + str(DEFAULT_PCT_STOP_LOSS) + "\n")
    f.write("PCT_QTY_ORDER: " + str(PCT_QTY_ORDER) + "\n")
    f.write("\n")
    try:
        perc_profit = round((ord.iloc[-1].values.tolist()[1] - initial_balance) / initial_balance * 100, 4)
        f.write("% Profit: " + str(perc_profit) + "\n")

        max_perdita = 0
        profit_factor = 0
        n_trade = 0
        n_win_trade = 0
        n_lose_trade = 0
        avg_win_trade = 0
        avg_lose_trade = 0
        avg_n_bar_trade_win = 0
        avg_n_bar_trade_lose = 0
        sum_win = 0
        sum_lose = 0

        ord2 = ord.reset_index()
        for index, row in ord2.iterrows():
            n_trade += 1
            if row[0] > 0:
                n_win_trade += 1
                avg_win_trade += row[0]
                d1 = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S')
                d2 = datetime.strptime(row[3], '%Y-%m-%d %H:%M:%S')
                d1 = time.mktime(d1.timetuple())
                d2 = time.mktime(d2.timetuple())
                avg_n_bar_trade_win += (d2 - d1) / 60 / 5
                sum_win += row[0]
            else:
                n_lose_trade += 1
                avg_lose_trade += row[0]
                d1 = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S')
                d2 = datetime.strptime(row[3], '%Y-%m-%d %H:%M:%S')
                d1 = time.mktime(d1.timetuple())
                d2 = time.mktime(d2.timetuple())
                avg_n_bar_trade_lose += (d2 - d1) / 60 / 5
                sum_lose += row[0]
                if max_perdita > row[0]:
                    max_perdita = row[0]

        if sum_win > 0 and sum_lose > 0:
            profit_factor = round(sum_win / sum_lose, 3)

        f.write("Profit factor: " + str(profit_factor) + "\n")
        f.write("# trades: " + str(n_trade) + "\n")
        f.write("# win trades: " + str(n_win_trade) + "\n")
        f.write("# lose trades: " + str(n_lose_trade) + "\n")

        if avg_win_trade > 0 and n_win_trade > 0:
            avg_win_trade = round(avg_win_trade / n_win_trade, 3)
        f.write("AVG win trades: " + str(avg_win_trade) + "\n")

        if avg_lose_trade > 0 and n_lose_trade > 0:
            avg_lose_trade = round(avg_lose_trade / n_lose_trade, 3)
        f.write("AVG lose trades: " + str(avg_lose_trade) + "\n")

        if avg_n_bar_trade_win > 0 and n_win_trade > 0:
            avg_n_bar_trade_win = round(avg_n_bar_trade_win / n_win_trade, 3)
        f.write("AVG win trades bars: " + str(avg_n_bar_trade_win) + "\n")

        if avg_n_bar_trade_lose > 0 and n_lose_trade > 0:
            avg_n_bar_trade_lose = round(avg_n_bar_trade_lose / n_lose_trade, 3)
        f.write("AVG lose trades bars: " + str(avg_n_bar_trade_lose) + "\n")
    except Exception as e:
        f.close()
        return 0
    f.close()

    return perc_profit


def writeRebuy(qty):
    global current_order
    global current_time
    global last_price
    global unrealizedPnlPcnt
    global step_rebuy
    rebuypct = getNextRebuyPct(step_rebuy - 1)
    entry_price = calcEntryPrice(qty)
    f = open("orders_old/rebuy.txt", 'a')
    f.write("Current time:" + str(datetime.utcfromtimestamp(current_time / 1000)) + "\n")
    f.write("Current price:" + str(last_price) + "\n")
    f.write("PNL:" + str(unrealizedPnlPcnt) + "\n")
    f.write("Rebuy PCT:" + str(rebuypct) + "\n")
    f.write("Qty: " + str(qty) + "\n")
    f.write("New EP: " + str(entry_price) + "\n")
    f.write(json.dumps(current_order) + "\n\n\n")


def import_ticks():
    global ticks_1m
    global ticks_5m
    global funding_rates
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
        'ignore'
    ]
    ticks_1m = pd.read_csv('data/SOLUSDT-1m.csv', names=columns)
    ticks_5m = pd.read_csv('data/SOLUSDT-5m.csv', names=columns)
    del ticks_1m['quote_asset_volume']
    del ticks_1m['number_of_trades']
    del ticks_1m['taker_buy_base_asset_volume']
    del ticks_1m['taker_buy_quote_asset_volume']
    del ticks_1m['ignore']
    del ticks_5m['quote_asset_volume']
    del ticks_5m['number_of_trades']
    del ticks_5m['taker_buy_base_asset_volume']
    del ticks_5m['taker_buy_quote_asset_volume']
    del ticks_5m['ignore']

    funding_rates = pd.read_json('data/BTCUSDT_fundingRates.json')
    funding_rates['timestamp'] = funding_rates['timestamp'].map(pd.Timestamp.timestamp)
    pass


def calcSR():
    global ticks_5m

    if (not os.path.exists("data/support_resistance_5m_sol.csv")):
        res = 0
        sup = 0
        inside_time = 0
        c = 0
        calc_sr_inside = {}
        len_tick = len(ticks_5m)
        for index, row in ticks_5m.iterrows():
            c = c + 1
            if c % 1000 == 0:
                pass
                # print(str(c) + " : " + str(len_tick) + "\n")
            inside_time = row['close_time']
            ohlcv = ticks_5m.loc[ticks_5m['close_time'] <= inside_time][-100:].values.tolist()
            last_support_new = 0
            last_resistance_new = 0
            supports = []
            resistances = []
            if len(ohlcv) < 3:
                continue
            m = []
            for i in range(2, (len(ohlcv) - 1)):
                m.append(abs(ohlcv[i][2] - ohlcv[i][3]))
                if ohlcv[i - 2][4] > ohlcv[i - 1][4] < ohlcv[i][4]:
                    last_support_new = float(ohlcv[i - 1][4])
                    supports.append(last_support_new)
                if ohlcv[i - 2][4] < ohlcv[i - 1][4] > ohlcv[i][4]:
                    last_resistance_new = float(ohlcv[i - 1][4])
                    resistances.append(last_resistance_new)
            if last_support_new != 0 and (last_support_new < sup or
                                          sup == 0 or
                                          (sup / last_support_new * 100) > pct_new_level or
                                          (last_support_new / sup * 100) > pct_new_level):
                sup = last_support_new
            if last_resistance_new != 0 and (last_resistance_new > res or
                                             res == 0 or
                                             (last_resistance_new / res * 100) > pct_new_level or
                                             (res / last_resistance_new * 100) > pct_new_level):
                res = last_resistance_new
            calc_sr_inside[str(inside_time)] = [str(inside_time), res, sup]
        df = pd.DataFrame.from_dict(calc_sr_inside, columns=['close_time', 'resistance', 'support'], orient='index')
        df.to_csv("data/support_resistance_5m_sol.csv", header=True)


def calcAlpha():
    global ticks_1m
    global N_TICKS_DELTA
    global N_TICKS_DELTA_NEG
    if (not os.path.exists("data/alpha_" + str(int(N_TICKS_DELTA)) + "_" + str(int(N_TICKS_DELTA)) + "_sol.csv")):
        ticks_1m.ta.ema(close=ticks_1m.ta.ohlc4(), length=N_TICKS_DELTA, suffix='alpha', append=True)
        ticks_1m.ta.ema(close=ticks_1m.ta.ohlc4(), length=N_TICKS_DELTA_NEG, suffix='alpha_neg', append=True)
        df = ticks_1m[
            ['close_time', 'EMA_' + str(int(N_TICKS_DELTA)) + '_alpha', 'EMA_' + str(int(N_TICKS_DELTA_NEG)) + '_alpha_neg']].copy()
        df.to_csv("data/alpha_" + str(int(N_TICKS_DELTA)) + "_" + str(int(N_TICKS_DELTA_NEG)) + "_sol.csv", header=True)


# https://www.bitmex.com/app/fees
def get_bitmex_fee():
    global orders
    global current_time
    if len(orders) > 0:
        ord = pd.DataFrame(orders)

        ord2 = ord.reset_index()
        ord2[2] = ord2[2].apply(convertDateToTime).map(pd.Timestamp.timestamp).astype(np.float64)

        curdate = convertDateToTime(datetime.utcfromtimestamp(current_time / 1000).strftime('%Y-%m-%d 00:00:00'))
        min30days = curdate - timedelta(days=30)
        min30days = datetime.timestamp(min30days)
        curdate = datetime.timestamp(curdate)
        ord2 = ord2.loc[ord2[2] >= min30days].loc[ord2[2] < curdate]
        sum_orders = 0
        for index, row in ord2.iterrows():
            sum_orders += float(row[10])
        if sum_orders > 0:
            sum_orders = sum_orders / 100000000
        if sum_orders >= 5 and sum_orders < 10:
            return 0.04
        elif sum_orders >= 10 and sum_orders < 25:
            return 0.035
        elif sum_orders >= 25 and sum_orders < 50:
            return 0.03
        elif sum_orders >= 50:
            return 0.025
    return 0.05


def read_next_trade():
    global last_price
    global id_trade
    global trades
    global month
    global current_time
    id_trade += 1
    if id_trade == 0:
        month_string = str(month)
        if (os.path.exists("data/SOLUSDT-trades-slim-2022-" + month_string.zfill(2) + ".csv")):
            columns = ['id', 'price', 'timestamp']
            trades = pd.read_csv("data/SOLUSDT-trades-slim-2022-" + month_string.zfill(2) + ".csv", names=columns,
                                 index_col='id')
        else:
            columns = ['id', 'price', 'i1', 'i2', 'timestamp', 'i4', 'i5']
            trades = pd.read_csv("data/SOLUSDT-trades-2022-" + month_string.zfill(2) + ".csv", names=columns,
                                 index_col='id')
            trades['timestamp'] = trades['timestamp'].divide(1000).astype('int').multiply(1000)
            del trades['i1']
            del trades['i2']
            del trades['i4']
            del trades['i5']
            trades = trades.drop_duplicates(subset=['timestamp'])
            trades.to_csv("data/SOLUSDT-trades-slim-2022-" + month_string.zfill(2) + ".csv", header=False)
    try:
        last_price = trades.iloc[id_trade]['price']
        current_time = trades.iloc[id_trade]['timestamp']
        current_month = datetime.utcfromtimestamp(current_time / 1000).strftime('%m')
    except Exception as e:
        try:
            month += 1
            month_string = str(month)
            if (os.path.exists("data/SOLUSDT-trades-slim-2022-" + month_string.zfill(2) + ".csv")):
                columns = ['id', 'price', 'timestamp']
                trades = pd.read_csv("data/SOLUSDT-trades-slim-2022-" + month_string.zfill(2) + ".csv", names=columns,
                                     index_col='id')
            else:
                columns = ['id', 'price', 'i1', 'i2', 'timestamp', 'i4', 'i5']
                trades = pd.read_csv("data/SOLUSDT-trades-2022-" + month_string.zfill(2) + ".csv", names=columns,
                                     index_col='id')
                trades['timestamp'] = trades['timestamp'].divide(1000).astype('int').multiply(1000)
                del trades['i1']
                del trades['i2']
                del trades['i4']
                del trades['i5']
                trades = trades.drop_duplicates(subset=['timestamp'])
                trades.to_csv("data/SOLUSDT-trades-slim-2022-" + month_string.zfill(2) + ".csv", header=False)
            id_trade = 0
            last_price = trades.iloc[id_trade]['price']
            current_time = trades.iloc[id_trade]['timestamp']
        except Exception as e:
            last_price = -1


def getNextRebuyPct(step):
    global N_REBUY
    global PCT_STOP_LOSS

    x = (N_REBUY * PCT_STOP_LOSS) / (N_REBUY + 1)
    s = N_REBUY - (N_REBUY - step)
    return s * x / N_REBUY


# Convertitore da satoshi a usd
def sat_to_usd(sat, price):
    usd = float(price) / 100000000 * float(sat)
    return usd


# Convertitore da usd a satoshi
def usd_to_sat(usd, price):
    sat = 100000000 / float(price) * float(usd)
    return sat


# Arrontonda a 100 una quantità
def roundTo100(qty):
    if qty < 100:
        qty = 100.0
    else:
        qty = round(qty, 0)
        resto = qty % 100
        if resto >= 50:
            qty = qty - resto + 100
        else:
            qty = qty - resto
    return qty


# Funzione che guarda il balance di un determinato simbolo
# Calcola la quantità in USD dell'ordine da fare
def getAvailableBalance(exchange, symbol):
    global available_balance
    global order_qty
    global positions
    global current_leverage
    global last_price
    global PCT_QTY_ORDER
    order_qty = sat_to_usd(available_balance, last_price) / 100 * PCT_QTY_ORDER
    order_qty = roundTo100(order_qty)


def calcEntryPrice(order_qty):
    global current_order
    global last_price

    entry_price = (abs(current_order['quantity']) * current_order['entry_price'] + abs(order_qty) * last_price) / (
            abs(current_order['quantity']) + abs(order_qty))
    return round(entry_price, 2)


# Fonte https://medium.com/@zoomerjd/how-to-bitmex-calculate-liquidation-price-on-xbtusd-1feac648178d
def calcBankruptcyPrice():
    global current_order
    global available_balance

    s = abs(int(current_order['quantity'])) / 100
    ep = current_order['entry_price']
    im = available_balance / 100000000

    return ((1 / ep) + (im / s)) ** -1


# Fonte https://medium.com/@zoomerjd/how-to-bitmex-calculate-liquidation-price-on-xbtusd-1feac648178d
def calcBankruptcyValue():
    global current_order

    bp = calcBankruptcyPrice()
    s = abs(int(current_order['quantity'])) / 100
    return s * (1 / bp)


# Fonte https://medium.com/@zoomerjd/how-to-bitmex-calculate-liquidation-price-on-xbtusd-1feac648178d
def calcMainteneceMargin():
    global current_order

    fr = getFundingRate()
    ev = abs(int(current_order['sat_qty'])) / 100000000
    bp = calcBankruptcyValue()
    # https://www.bitmex.com/app/contract/XBTUSD
    return (0.35 * ev) + (0.05 * bp) + (fr * bp)


# Fonte https://medium.com/@zoomerjd/how-to-bitmex-calculate-liquidation-price-on-xbtusd-1feac648178d
def calcLiqPrice():
    global current_order

    s = abs(int(current_order['quantity'])) / 100
    ep = current_order['entry_price']
    im = available_balance / 100000000
    mm = calcMainteneceMargin()

    return ((1 / ep) + ((im - mm) / s)) ** -1


# Da prendere nello storico di BITMEX
# https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=500&reverse=false&startTime=2021-01-01%2000%3A00%3A00
def getFundingRate():
    global funding_rates
    global funding_rate
    global old_min_fr
    global current_time

    if int(current_time / (60 * 60 * 8)) > old_min_fr:
        old_min_fr = int(current_time / (60 * 60 * 8))
        funding_rate = funding_rates.loc[funding_rates['timestamp'] <= (current_time / 1000)][-1:].values.tolist()[0][3]

    return funding_rate


def calcFees(order_qty):
    global fees
    global last_price
    fees = get_bitmex_fee()
    return usd_to_sat(order_qty, last_price) * fees / 100


# Piazza un ordine in posizione SELL
# Regola automaticamente il leveraggio per poter fare l'ordine
def placeOrderSell(exchange, qty):
    global orders
    global last_price
    global current_order
    global available_balance
    global current_time
    global step_rebuy
    global unrealizedPnlPcnt
    global min_unrealized_pnl

    qty = 0 - qty
    fees = calcFees(abs(qty))
    if current_order['quantity'] == 0:
        current_order['quantity'] = qty
        current_order['sat_qty'] = usd_to_sat(qty, last_price)
        current_order['entry_price'] = last_price
        current_order['fee_paid'] = fees
        current_order['entry_time'] = str(datetime.utcfromtimestamp(current_time / 1000))
    elif current_order['quantity'] > 0:
        # chiudi ordine e salva i guadagni
        current_order['fee_paid'] += fees
        gain = abs(current_order['sat_qty']) - usd_to_sat(abs(qty), last_price)
        available_balance += gain - current_order['fee_paid']
        print(str(rand_number) + " - " + "Esco alle " + str(datetime.utcfromtimestamp(current_time / 1000)))
        print(str(rand_number) + " - " + "Closed order gain: %.1f - Balance: %.1f" % (gain - current_order['fee_paid'], available_balance,))
        orders.append((gain - current_order['fee_paid'], available_balance, current_order['entry_time'],
                       str(datetime.utcfromtimestamp(current_time / 1000)), str(current_order['quantity']),
                       str(current_order['entry_price']), str(last_price), str(step_rebuy), str(unrealizedPnlPcnt),
                       str(min_unrealized_pnl), str(current_order['sat_qty'])))
        write_orders()
        current_order = {
            "quantity": 0,
            "entry_price": 0,
            "sat_qty": 0,
            "fee_paid": 0,
        }
    elif current_order['quantity'] < 0:
        entry_price = calcEntryPrice(qty)
        current_order['quantity'] += qty
        current_order['sat_qty'] += usd_to_sat(qty, last_price)
        # available_balance -= usd_to_sat(abs(qty), last_price)
        current_order['entry_price'] = entry_price
    getOpenPositions(exchange, ["SOLUSDT"])


# Piazza un ordine in posizione BUY
# Regola automaticamente il leveraggio per poter fare l'ordine
def placeOrderBuy(exchange, qty):
    global orders
    global last_price
    global current_order
    global available_balance
    global current_time
    global step_rebuy
    global unrealizedPnlPcnt
    global min_unrealized_pnl

    fees = calcFees(abs(qty))
    if current_order['quantity'] == 0:
        current_order['quantity'] = qty
        current_order['sat_qty'] = usd_to_sat(qty, last_price)
        current_order['entry_price'] = last_price
        current_order['fee_paid'] = fees
        current_order['entry_time'] = str(datetime.utcfromtimestamp(current_time / 1000))
    elif current_order['quantity'] < 0:
        # chiudi ordine e salva i guadagni
        current_order['fee_paid'] += fees
        gain = usd_to_sat(abs(qty), last_price) - abs(current_order['sat_qty'])
        available_balance += gain - current_order['fee_paid']
        print(str(rand_number) + " - " + "Esco alle " + str(datetime.utcfromtimestamp(current_time / 1000)))
        print(str(rand_number) + " - " + "Closed order gain: %.1f - Balance: %.1f" % (gain - current_order['fee_paid'], available_balance,))
        orders.append((gain - current_order['fee_paid'], available_balance, current_order['entry_time'],
                       str(datetime.utcfromtimestamp(current_time / 1000)), str(current_order['quantity']),
                       str(current_order['entry_price']), str(last_price), str(step_rebuy), str(unrealizedPnlPcnt),
                       str(min_unrealized_pnl)))
        write_orders()
        current_order = {
            "quantity": 0,
            "entry_price": 0,
            "sat_qty": 0,
            "fee_paid": 0,
        }
    elif current_order['quantity'] > 0:
        entry_price = calcEntryPrice(qty)
        writeRebuy(qty)
        current_order['quantity'] += qty
        current_order['sat_qty'] += usd_to_sat(qty, last_price)
        # available_balance -= usd_to_sat(abs(qty), last_price)
        current_order['entry_price'] = entry_price

    getOpenPositions(exchange, ["SOLUSDT"])


# Calcola gli ultimi supporti e resistenze
# Inoltre calcola la volatilità media delle ultime "price_change_average_lenght" candele
def lastSupportResistance(exchange):
    global last_support
    global last_resistance
    global last_order_price
    global price_change_average_lenght
    global last_price_change
    global calc_sr
    global current_time

    global old_min_sr

    cur_min = int(current_time / 1000 / (60 * 5))
    if cur_min == old_min_sr:
        return
    old_min_sr = cur_min
    row = calc_sr.loc[calc_sr['close_time'] <= current_time][-1:].values.tolist()
    last_support = row[0][1]
    last_resistance = row[0][2]


# Funzione che resistuisce l'ultimo prezzo di XBTUSD basato sull'ultimo trade fatto
def getLastPrice(exchange):
    c = 0
    while c < 10:
        try:
            trades = exchange.watch_trades(symbol='SOLUSDT', since=(exchange.milliseconds() - (60 * 1000)))
            return float(trades[-1]['price'])
        except Exception as inst:
            c = c + 1
            # print(inst)
            pass


# Attiva il ws per ascoltare i prezzi
# Se il prezzo è sopra la resistenza di % "treshold_price" oppure sotto il supporto crea un ordine
def fetchPrice(exchange):
    global last_support
    global last_resistance
    global last_order_price
    global treshold_price
    global order_qty
    global last_price_change
    global short_long
    global last_entry_price
    global last_order_price
    global step_rebuy
    global last_order_qty
    global last_price
    global current_time
    global max_unrealized_pnl
    global min_unrealized_pnl
    # global alpha
    # global alphaneg
    step_rebuy = 2
    last_order_qty = 0
    max_unrealized_pnl = 0
    min_unrealized_pnl = 0
    getAvailableBalance(exchange, 'USDT')

    # fetchAlpha(None)

    # print("Last Res: {%2.f} - Last Supp: {%2.f} - Last price: {%2.f} - Last price change: {%2.f}" % (last_resistance, last_support, last_price, (last_resistance * treshold_price_alpha),))
    if last_price < last_support - (
            last_support * treshold_price) and last_support > 0 and short_long == 's' and last_support != last_order_price:
        print(str(rand_number) + " - " + "Entro alle " + str(datetime.utcfromtimestamp(current_time / 1000)))
        placeOrderSell(exchange, order_qty)
        last_order_price = last_support
    elif last_price > last_resistance + (
            last_resistance * treshold_price) and last_resistance > 0 and short_long == 'l' and last_resistance != last_order_price:
        print(str(rand_number) + " - " + "Entro alle " + str(datetime.utcfromtimestamp(current_time / 1000)))
        placeOrderBuy(exchange, order_qty)
        last_order_price = last_resistance


# Funzione per chiudere gli ordini
# chiusura basata su trailing stop della percentuale del PNL
# rebuy basato su percentuale
def fetchPositionToClosePct(exchange):
    global last_support
    global last_resistance
    global last_order_price
    global last_price
    global positions
    global fees
    global acceptable_loss
    global real_last_order_price
    global current_leverage
    global unrealizedPnlPcnt
    global available_balance
    global max_unrealized_pnl
    global min_unrealized_pnl
    global order_qty
    global PCT_STOP_LOSS
    global PCT_VOLUME_DISTANCE
    global PCT_VOLUME_DISTANCE_NEGATIVE
    global N_REBUY
    global step_rebuy
    global last_order_qty
    global current_order
    global alpha
    global alphaneg
    start = time.time()
    try:
        canClose = False
        getOpenPositions(exchange, ['SOLUSDT'])
        getAvailableBalance(exchange, 'SOL')

        if last_order_qty != abs(int(current_order['quantity'])):
            max_unrealized_pnl = 0
            last_order_qty = abs(int(current_order['quantity']))

        if max_unrealized_pnl < unrealizedPnlPcnt:
            max_unrealized_pnl = unrealizedPnlPcnt

        if min_unrealized_pnl > unrealizedPnlPcnt:
            min_unrealized_pnl = unrealizedPnlPcnt

        if unrealizedPnlPcnt > (fees * 4):
            canClose = True

        fetchAlpha(None)
        gamma_stop = -alphaneg * PCT_VOLUME_DISTANCE_NEGATIVE * (step_rebuy / 2)
        gamma_take = alpha * PCT_VOLUME_DISTANCE

        next_step_rebuy = getNextRebuyPct(step_rebuy)
        # print("Current PNL: %.4f" % (unrealizedPnlPcnt,))
        # print("PCT Refill: %.4f" % (next_step_rebuy,))
        # print("Gamma stop: %.2f \nGamma take: %.2f" % (gamma_stop, gamma_take,))
        # print("Max PNL: %.4f - PNL = %.4f\n\n" % (max_unrealized_pnl, max_unrealized_pnl - unrealizedPnlPcnt))

        mult_order_qty = order_qty * abs(next_step_rebuy) * 10 / 2
        mult_order_qty = roundTo100(mult_order_qty)
        double_rebuy = roundTo100(abs(int(current_order['quantity'])) * 1.3)

        liquidation_pct = 100 / (abs(current_order['sat_qty']) / available_balance)
        liquidation_price = calcLiqPrice()
        if int(current_order['quantity']) < 0:  # SELL
            if (last_price > liquidation_price):
                last_price = 9999999999999
                placeOrderBuy(exchange, abs(int(current_order['quantity'])))
                last_price = -1
                max_unrealized_pnl = 0
            if (unrealizedPnlPcnt < 0 and max_unrealized_pnl > 0):
                max_unrealized_pnl = 0
                last_order_price = last_support
            if unrealizedPnlPcnt <= PCT_STOP_LOSS and (N_REBUY < step_rebuy):
                print("Close position qty: %.1f" % (abs(int(current_order['quantity']))))
                placeOrderBuy(exchange, abs(int(current_order['quantity'])))
                max_unrealized_pnl = 0
            elif (unrealizedPnlPcnt <= next_step_rebuy and next_step_rebuy < gamma_stop) or (
                    unrealizedPnlPcnt <= PCT_STOP_LOSS and unrealizedPnlPcnt < gamma_stop):
                step_rebuy = step_rebuy + 1
                print("Add position qty: %.1f" % (double_rebuy))
                placeOrderSell(exchange, double_rebuy)
            elif (canClose is True and (max_unrealized_pnl - unrealizedPnlPcnt) > gamma_take):
                print("Close position qty: %.1f" % (abs(int(current_order['quantity']))))
                placeOrderBuy(exchange, abs(int(current_order['quantity'])))
                max_unrealized_pnl = 0
                last_order_price = last_support
        else:  # BUY
            if (last_price < liquidation_price):
                last_price = -1
                placeOrderSell(exchange, abs(int(current_order['quantity'])))
                max_unrealized_pnl = 0
            if (unrealizedPnlPcnt < 0 and max_unrealized_pnl > 0):
                max_unrealized_pnl = 0
                last_order_price = last_resistance
            if unrealizedPnlPcnt <= PCT_STOP_LOSS and (N_REBUY < step_rebuy):
                print("Close position qty: %.1f" % (0 - int(current_order['quantity'])))
                placeOrderSell(exchange, abs(int(current_order['quantity'])))
                max_unrealized_pnl = 0
            elif (unrealizedPnlPcnt <= next_step_rebuy and next_step_rebuy < gamma_stop) or (
                    unrealizedPnlPcnt <= PCT_STOP_LOSS and unrealizedPnlPcnt < gamma_stop):
                step_rebuy = step_rebuy + 1
                print("Add position qty: %.1f" % (double_rebuy))
                placeOrderBuy(exchange, double_rebuy)
            elif (canClose is True and (max_unrealized_pnl - unrealizedPnlPcnt) > gamma_take):
                print("Close position qty: %.1f" % (0 - int(current_order['quantity'])))
                placeOrderSell(exchange, abs(int(current_order['quantity'])))
                max_unrealized_pnl = 0
                last_order_price = last_resistance
    except Exception as inst:
        # print(inst)
        pass


# Funzione per raccogliere le seguenti info:
# - quantità corrente di ordine
# - percentuale PNL
# - entry price
# - leveraggio corrente
def getOpenPositions(exchange, symbols):
    global positions
    global current_leverage
    global unrealizedPnlPcnt
    global last_unrealized_pnl
    global max_unrealized_pnl
    global last_entry_price
    global last_price
    global current_order
    global short_long

    if current_order['quantity'] != 0:

        unrealizedPnlPcnt = round(float((last_price / current_order['entry_price'] * 100) - 100), 2)
        if short_long == 's':
            unrealizedPnlPcnt = -unrealizedPnlPcnt
        if max_unrealized_pnl == 0:
            max_unrealized_pnl = unrealizedPnlPcnt


# Funzione principale
# Setta la posizione per vedere se lo script è long o short
# A ciclo infinito fa:
# - se ci sono posizioni aperte:
#       - controlla il prezzo per uscire
# - se non ci sono posizioni aperte:
#       - verifica l'ultimo supporto e resistenza
#       - ascolta il prezzo per fare un'entrata
def calculateProfit(config={}):
    global last_support
    global last_resistance
    global positions
    global short_long
    global last_price
    global current_order
    global current_time
    global month
    global treshold_price
    global pct_new_level
    global N_TICKS_DELTA
    global N_TICKS_DELTA_NEG
    global PCT_VOLUME_DISTANCE
    global PCT_VOLUME_DISTANCE_NEGATIVE
    global MULT_NEG_DELTA
    global N_REBUY
    global DEFAULT_PCT_STOP_LOSS
    global PCT_QTY_ORDER
    global rand_number

    if 'treshold_price' in config:
        treshold_price = config['treshold_price']
    if 'pct_new_level' in config:
        pct_new_level = config['pct_new_level']
    if 'N_TICKS_DELTA' in config:
        N_TICKS_DELTA = config['N_TICKS_DELTA']
    if 'N_TICKS_DELTA_NEG' in config:
        N_TICKS_DELTA_NEG = config['N_TICKS_DELTA_NEG']
    if 'PCT_VOLUME_DISTANCE' in config:
        PCT_VOLUME_DISTANCE = config['PCT_VOLUME_DISTANCE']
    if 'PCT_VOLUME_DISTANCE_NEGATIVE' in config:
        PCT_VOLUME_DISTANCE_NEGATIVE = config['PCT_VOLUME_DISTANCE_NEGATIVE']
    if 'MULT_NEG_DELTA' in config:
        MULT_NEG_DELTA = config['MULT_NEG_DELTA']
    if 'N_REBUY' in config:
        N_REBUY = config['N_REBUY']
    if 'DEFAULT_PCT_STOP_LOSS' in config:
        DEFAULT_PCT_STOP_LOSS = config['DEFAULT_PCT_STOP_LOSS']
    if 'PCT_QTY_ORDER' in config:
        PCT_QTY_ORDER = config['PCT_QTY_ORDER']

    import_ticks()
    calcSR()
    calcAlpha()
    import_precalc()

    rand_number = random.randint(1, 1234567890)

    bitmex = ()
    start_time = time.time()
    write_orders()
    # print("Inizio " + str(rand_number))
    month = 7
    day = 1
    read_next_trade()
    while last_price >= 0:
        if current_time % 3600000 == 0:
            pass
            print(str(rand_number) + " - " + str(datetime.utcfromtimestamp(current_time / 1000)))
            print(str(rand_number) + " - " + "Elapsed time: " + str((time.time() - start_time) / 60))
            # month = int(datetime.utcfromtimestamp(current_time / 1000).strftime('%m'))
            # day = int(datetime.utcfromtimestamp(current_time / 1000).strftime('%d'))
        read_next_trade()
        if last_price == -1:
            return write_orders()
        try:
            getOpenPositions(bitmex, ["SOLUSDT"])
            if current_order['quantity'] == 0:
                lastSupportResistance(bitmex)
                # print("Last support: " + str(last_support))
                # print("Last resistance: " + str(last_resistance))
                if last_support > 0 or last_resistance > 0:
                    # print("Cerco prezzo per entrare")
                    fetchPrice(bitmex)
            else:
                print("Cerco prezzo per uscire")
                fetchPositionToClosePct(bitmex)
        except Exception as inst:
            pass
            # print(inst)

    # print("Chiudo. Elapsed time: " + str((time.time() - start_time) / 60))
    return write_orders()


def fetchAlpha(exchange, config={}):
    global N_TICKS_DELTA
    global N_TICKS_DELTA_NEG
    global current_time
    global last_price
    global alpha
    global alphaneg
    global average
    global average_neg
    global old_min_alpha
    global PCT_STOP_LOSS
    global MULT_NEG_DELTA
    global DEFAULT_PCT_STOP_LOSS
    global calc_alpha

    cur_min = int(current_time / 1000 / 60)

    if cur_min != old_min_alpha:
        row = calc_alpha.loc[calc_alpha['close_time'] <= current_time][-1:].values.tolist()
        average = row[0][1]
        average_neg = row[0][2]
        old_min_alpha = cur_min

    if last_price > average:
        delta = (last_price / average * 100) - 100
    else:
        delta = (average / last_price * 100) - 100
    delta = float(delta)
    alpha = round(delta, 2)

    if last_price > average_neg:
        delta = (last_price / average_neg * 100) - 100
    else:
        delta = (average_neg / last_price * 100) - 100
    delta = float(delta)
    alphaneg = round(delta, 2)

    PCT_STOP_LOSS = -abs(alphaneg * MULT_NEG_DELTA)
    if PCT_STOP_LOSS > DEFAULT_PCT_STOP_LOSS:
        PCT_STOP_LOSS = DEFAULT_PCT_STOP_LOSS


def import_precalc():
    global calc_alpha
    global calc_sr
    global N_TICKS_DELTA
    global N_TICKS_DELTA_NEG

    calc_sr = pd.read_csv("data/support_resistance_5m_sol.csv")
    calc_alpha = pd.read_csv("data/alpha_" + str(int(N_TICKS_DELTA)) + "_" + str(int(N_TICKS_DELTA_NEG)) + "_sol.csv")


def evaluate(individual):
    config = {
        'treshold_price': individual[0],
        'pct_new_level': individual[1],
        'N_TICKS_DELTA': individual[2],
        'N_TICKS_DELTA_NEG': individual[3],
        'PCT_VOLUME_DISTANCE': individual[4],
        'PCT_VOLUME_DISTANCE_NEGATIVE': individual[5],
        'MULT_NEG_DELTA': individual[6],
        'N_REBUY': individual[7],
        'DEFAULT_PCT_STOP_LOSS': individual[8],
        'PCT_QTY_ORDER': individual[9]
    }
    names = ['treshold_price',
             'pct_new_level',
             'N_TICKS_DELTA',
             'N_TICKS_DELTA_NEG',
             'PCT_VOLUME_DISTANCE',
             'PCT_VOLUME_DISTANCE_NEGATIVE',
             'MULT_NEG_DELTA',
             'N_REBUY',
             'DEFAULT_PCT_STOP_LOSS',
             'PCT_QTY_ORDER',
             'result'
             ]
    calc = -calculateProfit(config)
    # waittime = random.randint(2, 10)
    # calc = random.randint(1, 10000)
    # print("Waiting " + str(waittime) + " seconds")
    # time.sleep(waittime)
    config['result'] = -calc
    towrite = {"1": config}
    with open('results_sol.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in towrite.items():
            writer.writerow([key, value])
    csvfile.close()

    return calc


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


TRESHOLD_MIN, TRESHOLD_MAX = 0.000001, 0.0001
PCT_NEW_LEVEL_MIN, PCT_NEW_LEVEL_MAX = 99.5, 102
TICK_DELTA_MIN, TICK_DELTA_MAX = 4, 20
TICK_DELTA_NEG_MIN, TICK_DELTA_NEG_MAX = 50, 200
PCT_VOL_DIST_MIN, PCT_VOL_DIST_MAX = 0.1, 4
PCT_VOL_DIST_NEG_MIN, PCT_VOL_DIST_NEG_MAX = 0.1, 10
MULT_NEG_DELTA_MIN, MULT_NEG_DELTA_MAX = 1, 100
N_REBUY_MIN, N_REBUY_MAX = 2, 10
DEFAULT_PCT_STOP_LOSS_MIN, DEFAULT_PCT_STOP_LOSS_MAX = -90, -1
PCT_QTY_ORD_MIN, PCT_QTY_ORD_MAX = 1, 100


def main():
    random.seed(169)

    # pool = multiprocessing.Pool()
    varbound = np.array([[TRESHOLD_MIN, TRESHOLD_MAX], [PCT_NEW_LEVEL_MIN, PCT_NEW_LEVEL_MAX], [TICK_DELTA_MIN, TICK_DELTA_MAX],
                         [TICK_DELTA_NEG_MIN, TICK_DELTA_NEG_MAX], [PCT_VOL_DIST_MIN, PCT_VOL_DIST_MAX],
                         [PCT_VOL_DIST_NEG_MIN, PCT_VOL_DIST_NEG_MAX], [MULT_NEG_DELTA_MIN, MULT_NEG_DELTA_MAX], [N_REBUY_MIN, N_REBUY_MAX],
                         [DEFAULT_PCT_STOP_LOSS_MIN, DEFAULT_PCT_STOP_LOSS_MAX], [PCT_QTY_ORD_MIN, PCT_QTY_ORD_MAX]])
    vartype = np.array([['real'], ['real'], ['int'], ['int'], ['real'], ['real'], ['real'], ['int'], ['real'], ['real']])
    algorithm_param = {'max_num_iteration': None,
                       'population_size': 20,
                       'mutation_probability': 0.25,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'two_point',
                       'max_iteration_without_improv': None,
                       'multiprocessing_ncpus': 4,
                       'multiprocessing_engine': None
                       }
    model = ga(function=evaluate, dimension=10, variable_type_mixed=vartype, variable_boundaries=varbound, function_timeout=1000000000,
               algorithm_parameters=algorithm_param)

    model.run()


if __name__ == '__main__':
    c = {'treshold_price': 3.119274843259444e-05, 'pct_new_level': 100.2068483379926, 'N_TICKS_DELTA': 5.0, 'N_TICKS_DELTA_NEG': 10.0,
         'PCT_VOLUME_DISTANCE': 0.1, 'PCT_VOLUME_DISTANCE_NEGATIVE': 13, 'MULT_NEG_DELTA': 4,
         'N_REBUY': 8.0, 'DEFAULT_PCT_STOP_LOSS': -4.0, 'PCT_QTY_ORDER': 100}
    calculateProfit(c)
    # main()
