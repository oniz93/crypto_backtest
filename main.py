import asyncio
import time
import sys
import subprocess
import ccxtpro as ccxt

# ultimo supporto riscontrato secondo metodologia Wyckoff semplificata
# default a 0 perché
last_support = 0

# ultima resistenza riscontrata secondo metodologia Wyckoff semplificata
last_resistance = 0

last_order_price = 0
last_order_qty = 0
last_price = 0
treshold_price = 0.0005
positions = []
order_qty = 0
price_change_average_lenght = 10
trail_qty = 350
fees = 0.0028
current_leverage = 25
unrealizedPnlPcnt = 0
available_balance = 0
max_unrealized_pnl = 0
min_margin_liq_perc = 15
delta_margin_liq_perc = 45

max_leverage = 20
short_long = 'l'
pct_new_level = 100.5
last_entry_price = 0

# n° candele da analizzare per il calcolo del gamma positivo
N_TICKS_DELTA = 8
# n° candele da analizzare per il calcolo del gamma negativo
N_TICKS_DELTA_NEG = 50
# coefficente per il take profit
PCT_VOLUME_DISTANCE = 0.7
# coefficente per il rebuy
PCT_VOLUME_DISTANCE_NEGATIVE = 1.8

pct_distance = 100

N_REBUY = 5
PCT_STOP_LOSS = -1.2
step_rebuy = 2

def getNextRebuyPct(step):
    global N_REBUY
    global PCT_STOP_LOSS

    x = (N_REBUY * PCT_STOP_LOSS) / (N_REBUY + 1)
    s = N_REBUY - (N_REBUY - step)
    return s*x/N_REBUY



# Apre la posizione dell'exchange (Bitmex)
# sceglie l'account basandosi sul fatto che il software sia fatto partire
# con impostazione long o short
async def openExchange(short_long):
    if short_long == 's':
        # Omar
        bitmex = ccxt.bitmex({
            'apiKey': 'jthlVK-ZVNU0MxrMOigHILnK',
            'secret': 'f2FVLDeAJ9qloB3pBzD_k_PNSo_prwSrPswVYnFiuzdzpAfH',
            'enableRateLimit': True
        })
        print("SHORT")
    else:
        # Teo
        bitmex = ccxt.bitmex({
            'apiKey': 'MfQ26mFjrCuDG36-GrUuVouw',
            'secret': 'GamTfJxbQ2sqSj_fFUQySqw9CbkbGaPwYBtnEJZpZOfqKhEq',
            'enableRateLimit': True
        })
        # bitmex = ccxt.bitmex({
        #     'apiKey': '9hrJo8aYRtkczLCKHTunb5yR',
        #     'secret': '9Agi0e-PFX_OF-liCjuh_9d1b9oW9uJWc_hs5NxNuKYizpyo',
        #     'enableRateLimit': True
        # })
        short_long = 'l'
        print("LONG")

    return bitmex

# Funzione per scrivere i log
def writeOpen(data):
    f = open('orders_old/open.log', 'a')
    for i in range(0, len(data) - 1):
        f.write(str(data[i])+",")
    f.write("\n")
    f.close()

# Funzione per scrivere i log
def writeClose(data):
    f = open('close.log', 'a')
    for i in range(0, len(data) - 1):
        f.write(str(data[i])+",")
    f.write("\n")
    f.close()

# Convertitore da satoshi a usd
def sat_to_usd(sat, price):
    usd = float(price) / 100000000 * float(sat)
    return usd

# Convertitore da usd a satoshi
def usd_to_sat(usd, price):
    sat = 100000000 / float(price) * float(usd)
    return sat

# Cambia il leveraggio al valore scelto
async def changeLeverage(exchange, symbol, leverage):
    global current_leverage
    await exchange.request(path="position/leverage", method="POST", params={"symbol": symbol, "leverage": leverage})
    current_leverage = leverage

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
async def getAvailableBalance(exchange, symbol):
    global available_balance
    global order_qty
    global positions
    balance = await exchange.request(path="user/margin", method="GET", params={"currency": symbol})
    available_balance = float(balance['availableMargin'])
    total_balance = float(balance['amount'])
    try:
        if len(positions) > 0:
            price = float(positions[0]['markPrice'])
        else:
            price = await getLastPrice(exchange)
            price = float(price)

        order_qty = sat_to_usd(total_balance, price) / 100 * 22
        order_qty = roundTo100(order_qty)
    except Exception as inst:
        print(inst)

# Piazza un ordine in posizione SELL
# Regola automaticamente il leveraggio per poter fare l'ordine
async def placeOrderSell(exchange, qty):
    global orders
    global positions
    global max_leverage
    global current_leverage
    skip_leverage = False
    if len(positions) > 0:
        price = float(positions[0]['markPrice'])
        if int(positions[0]['currentQty']) > 0:
            skip_leverage = True
    else:
        price = await getLastPrice(exchange)
        price = float(price)
    qty = roundTo100(qty)
    sat_to_order = usd_to_sat(qty,price)
    c = 0
    if not skip_leverage:
        while c < 200 and current_leverage <= max_leverage:
            await getAvailableBalance(exchange, 'XBt')
            if available_balance < (sat_to_order + sat_to_order * 0.001) / current_leverage:
                c = c + 1
                try:
                    await changeLeverage(exchange, 'XBTUSD', current_leverage + 1)
                except Exception as inst:
                    print(inst)
                    time.sleep(2)
                    try:
                        await changeLeverage(exchange, 'XBTUSD', current_leverage + 0.5)
                    except Exception as inst:
                        print(inst)
                        time.sleep(2)
            else:
                break
    print("Order qty: %.1f" % (qty,))
    qty = 0 - qty
    await exchange.create_market_sell_order(symbol="XBTUSD", amount=qty, params={
        'orderQty': qty,
    })
    orders = await getOpenPositions(exchange, ["XBTUSD"])

# Piazza un ordine in posizione BUY
# Regola automaticamente il leveraggio per poter fare l'ordine
async def placeOrderBuy(exchange, qty):
    global orders
    global positions
    global max_leverage
    global current_leverage
    skip_leverage = False
    if len(positions) > 0:
        price = float(positions[0]['markPrice'])
        if int(positions[0]['currentQty']) < 0:
            skip_leverage = True
    else:
        price = await getLastPrice(exchange)
        price = float(price)
    qty = roundTo100(qty)
    sat_to_order = usd_to_sat(qty,price)
    c = 0
    if not skip_leverage:
        while c < 200 and current_leverage <= max_leverage:
            await getAvailableBalance(exchange, 'XBt')
            if available_balance < (sat_to_order + sat_to_order * 0.001) / current_leverage:
                c = c + 1
                try:
                    await changeLeverage(exchange, 'XBTUSD', current_leverage + 1)
                except Exception as inst:
                    print(inst)
                    time.sleep(2)
                    try:
                        await changeLeverage(exchange, 'XBTUSD', current_leverage + 0.5)
                    except Exception as inst:
                        print(inst)
                        time.sleep(2)
            else:
                break
    print("Order qty: %.1f" % (qty,))
    await exchange.create_market_buy_order(symbol="XBTUSD", amount=qty, params={
        'orderQty': qty,
    })
    await getOpenPositions(exchange, ["XBTUSD"])

# Calcola gli ultimi supporti e resistenze
# Inoltre calcola la volatilità media delle ultime "price_change_average_lenght" candele
async def lastSupportResistance(exchange):
    global last_support
    global last_resistance
    global last_order_price
    global trail_qty
    global price_change_average_lenght
    global pct_new_level
    ohlcv = await exchange.fetch_ohlcv(symbol="XBTUSD", timeframe="5m", limit=100, since=(exchange.milliseconds() - (5 * 100 * 60 * 1000)))
    last_support_new = 0
    last_resistance_new = 0
    supports = []
    resistances = []
    if len(ohlcv) < 3:
        return
    c = 0
    m = []
    for i in range(2, (len(ohlcv) - 1)):
        m.append(abs(ohlcv[i][2] - ohlcv[i][3]))
        if ohlcv[i - 2][4] > ohlcv[i - 1][4] < ohlcv[i][4]:
            last_support_new = float(ohlcv[i - 1][4])
            supports.append(last_support_new)
        if ohlcv[i - 2][4] < ohlcv[i - 1][4] > ohlcv[i][4]:
            last_resistance_new = float(ohlcv[i - 1][4])
            resistances.append(last_resistance_new)
    t = 0
    for i in range(len(m) - 1 - price_change_average_lenght, len(m) - 1):
        t = m[i] + t
        c = c + 1
    trail_qty = int(t / c / 2)

    if last_support_new < last_support or \
            last_support == 0 or \
            (last_support / last_support_new * 100) > pct_new_level or \
            (last_support_new / last_support * 100) > pct_new_level:
        last_support = last_support_new
    if last_resistance_new > last_resistance or \
            last_resistance == 0 or \
            (last_resistance_new / last_resistance * 100) > pct_new_level or \
            (last_resistance / last_resistance_new * 100) > pct_new_level:
        last_resistance = last_resistance_new

    f = open('last_support', 'w')
    f.write(str(last_support))
    f.close()
    f = open('last_resistance', 'w')
    f.write(str(last_resistance))
    f.close()

    print("Last support: " + str(last_support))
    print("Last resistance: " + str(last_resistance))

# Funzione che resistuisce l'ultimo prezzo di XBTUSD basato sull'ultimo trade fatto
async def getLastPrice(exchange):
    c = 0
    while c < 10:
        try:
            trades = await exchange.watch_trades(symbol='XBTUSD', since=(exchange.milliseconds() - (60 * 1000)))
            return float(trades[-1]['price'])
        except Exception as inst:
            c = c + 1
            print(inst)
            pass

# Attiva il ws per ascoltare i prezzi
# Se il prezzo è sopra la resistenza di % "treshold_price" oppure sotto il supporto crea un ordine
async def fetchPrice(exchange):
    global last_support
    global last_resistance
    global last_order_price
    global treshold_price
    global order_qty
    global short_long
    global last_entry_price
    global step_rebuy
    start = time.time()
    step_rebuy = 2
    await getAvailableBalance(exchange, 'XBt')
    try:
        while True:
            trades = await exchange.watch_trades(symbol='XBTUSD', since=(exchange.milliseconds() - (60 * 1000)))

            alpha = read_alpha()
            alphaneg = read_alpha_neg()
            treshold_price_alpha = 0

            print("Last Res: {%2.f} - Last Supp: {%2.f} - Last price: {%2.f} - Last price change: {%2.f}" % (
                last_resistance, last_support, trades[-1]['price'], (last_resistance * treshold_price_alpha),))
            if trades[-1]['price'] < last_support - (last_support * treshold_price_alpha) and last_support > 0 and short_long == 's' and last_support != last_order_price:
                await placeOrderSell(exchange, order_qty)
                last_order_price = last_support
                writeOpen([time.time(), trades[-1]['price'], last_resistance, last_support, "SHORT"])
                break
            elif trades[-1]['price'] > last_resistance + (last_resistance * treshold_price_alpha) and last_resistance > 0 and short_long == 'l' and last_resistance != last_order_price:
                await placeOrderBuy(exchange, order_qty)
                last_order_price = last_resistance
                writeOpen([time.time(), trades[-1]['price'], last_resistance, last_support, "LONG"])
                break

            if start + 30 <= time.time():
                break
    except Exception as inst:
        print(inst)
        pass

# Controlla se il liquidation price corrente è troppo vicino al mark price e cambia il leveraggio per allontanare o avvicinarsi al liquidation
async def checkLiquidation(exchange):
    global positions
    global current_leverage
    global min_margin_liq_perc
    global delta_margin_liq_perc
    global max_leverage
    try:
        while True:
            liq = float(positions[0]['liquidationPrice'])
            if liq <= 0:
                liq = 1
            pct = abs(
                ((liq * 100) / float(positions[0]['markPrice'])) - 100)  # distanza in % del liq. rispetto allo 0 in relazione al markPrice attuale
            if pct <= min_margin_liq_perc:
                if current_leverage > 1:
                    if current_leverage <= 5:
                        await changeLeverage(exchange, 'XBTUSD', current_leverage - 1)
                        await getOpenPositions(exchange, ['XBTUSD'])
                        continue
                    else:
                        await changeLeverage(exchange, 'XBTUSD', current_leverage - 2)
                        await getOpenPositions(exchange, ['XBTUSD'])
                        continue
            if pct > delta_margin_liq_perc + min_margin_liq_perc:
                if current_leverage < 10:
                    await changeLeverage(exchange, 'XBTUSD', current_leverage + 1)
                    await getOpenPositions(exchange, ['XBTUSD'])
                    continue
                elif current_leverage <= 25 and current_leverage <= max_leverage - 2:
                    await changeLeverage(exchange, 'XBTUSD', current_leverage + 2)
                    await getOpenPositions(exchange, ['XBTUSD'])
                    continue
            break
    except Exception as inst:
        pass

# Funzione per chiudere gli ordini
# chiusura basata su trailing stop della percentuale del PNL
# rebuy basato su percentuale
async def fetchPositionToClosePct(exchange):
    global last_support
    global last_resistance
    global last_order_price
    global last_price
    global positions
    global fees
    global acceptable_loss
    global unrealizedPnlPcnt
    global max_unrealized_pnl
    global order_qty
    global PCT_STOP_LOSS
    global PCT_VOLUME_DISTANCE
    global PCT_VOLUME_DISTANCE_NEGATIVE
    global N_REBUY
    global step_rebuy
    global last_order_qty
    start = time.time()
    try:
        while True:
            canClose = False
            await getOpenPositions(exchange, ['XBTUSD'])
            await getAvailableBalance(exchange, 'XBt')

            if last_order_qty != abs(int(positions[0]['currentQty'])):
                max_unrealized_pnl = 0
                last_order_qty = abs(int(positions[0]['currentQty']))

            if max_unrealized_pnl < unrealizedPnlPcnt:
                max_unrealized_pnl = unrealizedPnlPcnt

            if unrealizedPnlPcnt > (fees * 100):
                canClose = True

            await checkLiquidation(exchange)

            alpha = read_alpha()
            alphaneg = read_alpha_neg()
            gamma_stop = -alphaneg * PCT_VOLUME_DISTANCE_NEGATIVE * (step_rebuy / 2)
            gamma_take = alpha * PCT_VOLUME_DISTANCE

            next_step_rebuy = getNextRebuyPct(step_rebuy)
            print("Current PNL: %.4f" % (unrealizedPnlPcnt,))
            print("PCT Refill: %.4f" % (next_step_rebuy,))
            print("Gamma stop: %.2f \nGamma take: %.2f" % (gamma_stop, gamma_take,) )
            print("Max PNL: %.4f - PNL = %.4f\n\n" % (max_unrealized_pnl, max_unrealized_pnl - unrealizedPnlPcnt))

            double_rebuy = roundTo100(abs(int(positions[0]['currentQty'])) * 1.3)


            if int(positions[0]['currentQty']) < 0:  # SELL
                if (unrealizedPnlPcnt < 0 and max_unrealized_pnl > 0):
                    max_unrealized_pnl = 0
                    last_order_price = last_support
                if unrealizedPnlPcnt <= PCT_STOP_LOSS:
                    print("Close position qty: %.1f" % (abs(int(positions[0]['currentQty']))))
                    await placeOrderBuy(exchange, abs(int(positions[0]['currentQty'])))
                    max_unrealized_pnl = 0
                    writeClose([time.time(), unrealizedPnlPcnt, abs(int(positions[0]['currentQty'])), "SHORT"])
                    break
                elif unrealizedPnlPcnt <= next_step_rebuy and next_step_rebuy < gamma_stop:
                    step_rebuy = step_rebuy + 1
                    print("Add position qty: %.1f" % (double_rebuy))
                    await placeOrderSell(exchange, double_rebuy)
                    if unrealizedPnlPcnt <= PCT_STOP_LOSS:
                        await sendRebuy(abs(int(positions[0]['currentQty'])))
                    break
                elif (canClose is True and (max_unrealized_pnl - unrealizedPnlPcnt) > gamma_take):
                    print("Close position qty: %.1f" % (abs(int(positions[0]['currentQty']))))
                    await placeOrderBuy(exchange, abs(int(positions[0]['currentQty'])))
                    max_unrealized_pnl = 0
                    last_order_price = last_support
                    writeClose([time.time(), unrealizedPnlPcnt, abs(int(positions[0]['currentQty'])), "SHORT"])
                    break
            else:  # BUY
                if (unrealizedPnlPcnt < 0 and max_unrealized_pnl > 0):
                    max_unrealized_pnl = 0
                    last_order_price = last_resistance
                if unrealizedPnlPcnt <= PCT_STOP_LOSS:
                    print("Close position qty: %.1f" % (0 - int(positions[0]['currentQty'])))
                    await placeOrderSell(exchange, abs(int(positions[0]['currentQty'])))
                    max_unrealized_pnl = 0
                    writeClose([time.time(), unrealizedPnlPcnt, 0 - int(positions[0]['currentQty']), "LONG"])
                    break
                elif unrealizedPnlPcnt <= next_step_rebuy and next_step_rebuy < gamma_stop:
                    step_rebuy = step_rebuy + 1
                    print("Add position qty: %.1f" % (double_rebuy))
                    await placeOrderBuy(exchange, double_rebuy)
                    if unrealizedPnlPcnt <= PCT_STOP_LOSS:
                        await sendRebuy(abs(int(positions[0]['currentQty'])))
                    break
                elif (canClose is True and (max_unrealized_pnl - unrealizedPnlPcnt) > gamma_take):
                    print("Close position qty: %.1f" % (0 - int(positions[0]['currentQty'])))
                    await placeOrderSell(exchange, abs(int(positions[0]['currentQty'])))
                    max_unrealized_pnl = 0
                    last_order_price = last_resistance
                    writeClose([time.time(), unrealizedPnlPcnt, 0 - int(positions[0]['currentQty']), "LONG"])
                    break
            if start + 120 <= time.time():
                break
            else:
                time.sleep(2)
    except Exception as inst:
        print(inst)
        pass

# Funzione per raccogliere le seguenti info:
# - quantità corrente di ordine
# - percentuale PNL
# - entry price
# - leveraggio corrente
async def getOpenPositions(exchange, symbols):
    global positions
    global current_leverage
    global unrealizedPnlPcnt
    global last_unrealized_pnl
    global max_unrealized_pnl
    global last_entry_price
    positions = []
    p = await exchange.fetch_positions(symbols=symbols)
    for i in range(0, len(p)):
        if int(p[i]['currentQty']) != 0:
            positions.append(p[i])
            if round(float(p[i]['leverage']), 2) > 0 and p[i]['symbol'] == 'XBTUSD':
                current_leverage = round(float(p[i]['leverage']), 2)
                unrealizedPnlPcnt = float(p[i]['unrealisedRoePcnt']) * 100 / current_leverage
                if max_unrealized_pnl == 0:
                    max_unrealized_pnl = unrealizedPnlPcnt
            if short_long == 's':
                f = open('short', 'w')
            else:
                f = open('long', 'w')
            f.write(str(abs(int(p[i]['currentQty']))))
            f.close()
            last_entry_price = abs(int(float(p[i]['avgEntryPrice'])))

# Funzione principale
# Setta la posizione per vedere se lo script è long o short
# A ciclo infinito fa:
# - se ci sono posizioni aperte:
#       - controlla il prezzo per uscire
# - se non ci sono posizioni aperte:
#       - verifica l'ultimo supporto e resistenza
#       - ascolta il prezzo per fare un'entrata
async def main():
    global last_support
    global last_resistance
    global positions
    global short_long
    # from variable id
    if 'short' in sys.argv or 'SHORT' in sys.argv:
        short_long = 's'
    else:
        short_long = 'l'
    bitmex = await openExchange(short_long)

    while True:
        try:
            await getOpenPositions(bitmex, ["XBTUSD"])
            if len(positions) == 0:
                if short_long == 's':
                    f = open('short', 'w')
                else:
                    f = open('long', 'w')
                f.write("0")
                f.close()
                await lastSupportResistance(bitmex)
                print("Last support: " + str(last_support))
                print("Last resistance: " + str(last_resistance))
                if last_support > 0 or last_resistance > 0:
                    print("Cerco prezzo per entrare")
                    await fetchPrice(bitmex)
            else:
                print("Cerco prezzo per uscire")
                await fetchPositionToClosePct(bitmex)

            if len(positions) == 0:
                time.sleep(1)
            else:
                time.sleep(1)
        except Exception as inst:
            print(inst)
            time.sleep(1)

    await bitmex.close()

async def fetchAlpha(exchange):
    global N_TICKS_DELTA
    global N_TICKS_DELTA_NEG
    ohlcv = await exchange.fetch_ohlcv(symbol="XBTUSD", timeframe="1m", limit=100, since=(exchange.milliseconds() - (1 * 100 * 60 * 1000)))
    ohlcv.reverse()
    average = 0
    average_neg = 0
    if len(ohlcv) < N_TICKS_DELTA:
        return
    for i in range(0, (len(ohlcv) - 1)):
        if i >= N_TICKS_DELTA:
            break
        average += ((ohlcv[i][1] + ohlcv[i][2] + ohlcv[i][3] + ohlcv[i][4]) / 4)
    average = average / N_TICKS_DELTA

    for i in range(0, (len(ohlcv) - 1)):
        if i >= N_TICKS_DELTA_NEG:
            break
        average_neg += ((ohlcv[i][1] + ohlcv[i][2] + ohlcv[i][3] + ohlcv[i][4]) / 4)
    average_neg = average_neg / N_TICKS_DELTA_NEG

    start = time.time()
    await getAvailableBalance(exchange, 'XBt')
    try:
        while True:
            trades = await exchange.watch_trades(symbol='XBTUSD', since=(exchange.milliseconds() - (60 * 1000)))

            last_price = trades[-1]['price']

            if last_price > average:
                delta = (last_price / average * 100) - 100
            else:
                delta = (average / last_price * 100) - 100
            delta = float(delta)
            delta = round(delta, 2)
            f = open('alpha', 'w')
            f.write(str(delta))
            f.close()
            print("Alpha: " + str(delta))

            if last_price > average_neg:
                delta = (last_price / average_neg * 100) - 100
            else:
                delta = (average_neg / last_price * 100) - 100
            delta = float(delta)
            delta = round(delta, 2)
            f = open('alphaneg', 'w')
            f.write(str(delta))
            f.close()
            print("Alpha neg: " + str(delta))

            if start + 15 <= time.time():
                break
    except Exception as inst:
        print(inst)
        pass


async def alpha():
    bitmex = await openExchange(short_long)
    while True:
        try:
            await fetchAlpha(bitmex)
            time.sleep(5)
        except Exception as inst:
            print(inst)
            time.sleep(10)



# Funzione per aprire un rebuy nella posizione opposta
async def sendRebuy(qty):
    global short_long
    if short_long == 's':
        subprocess.run(["python3", "main.py", "long", "rebuy", str(qty)])
    else:
        subprocess.run(["python3", "main.py", "short", "rebuy", str(qty)])

# Funzione per il rebuy
async def rebuy(qty):
    global short_long
    global positions
    try:
        bitmex = await openExchange(short_long)
        await getOpenPositions(bitmex, ["XBTUSD"])
        await getAvailableBalance(bitmex, 'XBt')
        if len(positions) == 0:
            current_position = 0
        else:
            current_position = abs(int(positions[0]['currentQty']))
        if qty - current_position > 0:
            qty = qty - current_position
        else:
            return True

        if short_long == 's':
            await placeOrderSell(bitmex, qty)
        else:
            await placeOrderBuy(bitmex, qty)
        await bitmex.close()
    except Exception as inst:
        print(inst)
        await rebuy(qty)

# Lettura degli ultimi supporti e resistenze scritti su disco
def read_last_supp_res():
    global last_support
    global last_resistance
    try:
        f = open('last_support', 'r')
        last_support = float(f.readline())
        f.close()
    except Exception as inst:
        print(inst)
        last_support = 0
    try:
        f = open('last_resistance', 'r')
        last_resistance = float(f.readline())
        f.close()
    except Exception as inst:
        print(inst)
        last_resistance = 0

def read_alpha():
    try:
        f = open('alpha', 'r')
        alpha = float(f.readline())
        f.close()
    except Exception as inst:
        print(inst)
        alpha = 100
    return alpha

def read_alpha_neg():
    try:
        f = open('alphaneg', 'r')
        alpha = float(f.readline())
        f.close()
    except Exception as inst:
        print(inst)
        alpha = 100
    return alpha

if __name__ == '__main__':
    read_last_supp_res()
    if 'short' in sys.argv or 'SHORT' in sys.argv:
        short_long = 's'
    else:
        short_long = 'l'
    if 'rebuy' in sys.argv:
        qty = float(sys.argv[3])
        if qty > 0:
            asyncio.get_event_loop().run_until_complete(rebuy(qty))
    elif 'alpha' in sys.argv:
        asyncio.get_event_loop().run_until_complete(alpha())
    else:
        asyncio.get_event_loop().run_until_complete(main())
