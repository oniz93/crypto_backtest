import asyncio
import time
import sys
import subprocess


import ccxtpro as ccxt

last_support = 0
last_resistance = 0

last_order_price = 0
real_last_order_price = 0
last_price = 0
treshold_price = 0.0005
positions = []
order_qty = 0
price_change_average_lenght = 10
last_price_change = 0
trail_qty = 350
fees = 0.0028
acceptable_loss = 0.005
current_leverage = 25
unrealizedPnlPcnt = 0
available_balance = 0
margin_used = 0
current_order_value = 0
max_unrealized_pnl = 0
total_balance = 0
min_margin_liq_perc = 15
delta_margin_liq_perc = 45
time_last_order = 0
PCNT_REFILL = -0.4
pct_refill = PCNT_REFILL
pct_close_order = 0.15
pct_stop_loss = -1.2
last_res_sup_check = 5
max_leverage = 20
short_long = 'l'
pct_new_level = 100.5
pct_new_level_rebuy = 100.3
multi_edging = 1.3
last_entry_price = 0


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

def writeOpen(data):
    f = open('open.log', 'a')
    for i in range(0, len(data) - 1):
        f.write(str(data[i])+",")
    f.write("\n")
    f.close()

def writeClose(data):
    f = open('../close.log', 'a')
    for i in range(0, len(data) - 1):
        f.write(str(data[i])+",")
    f.write("\n")
    f.close()

def sat_to_usd(sat, price):
    usd = float(price) / 100000000 * float(sat)
    return usd

def usd_to_sat(usd, price):
    sat = 100000000 / float(price) * float(usd)
    return sat


async def printMarkets(exchange):
    market = await exchange.watch_liquidation("XBTUSD")
    print(market)


async def changeLeverage(exchange, symbol, leverage):
    global current_leverage
    await exchange.request(path="position/leverage", method="POST", params={"symbol": symbol, "leverage": leverage})
    current_leverage = leverage


async def getAvailableBalance(exchange, symbol):
    global available_balance
    global margin_used
    global current_order_value
    global total_balance
    global order_qty
    global positions
    global current_leverage
    balance = await exchange.request(path="user/margin", method="GET", params={"currency": symbol})
    available_balance = float(balance['availableMargin'])
    total_balance = float(balance['amount'])
    margin_used = float(balance['marginUsedPcnt'])
    current_order_value = float(balance['riskValue'])
    try:
        if len(positions) > 0:
            price = float(positions[0]['markPrice'])
        else:
            price = await getLastPrice(exchange)
            price = float(price)
        # if short_long == 's':
        #     f = open('long', 'r')
        # else:
        #     f = open('short', 'r')
        # order_qty = int(f.readline())
        # f.close()
        if order_qty == 0:
            order_qty = sat_to_usd(total_balance, price) / current_leverage / 100 * 10
            if order_qty < 100:
                order_qty = 100.0
            else:
                order_qty = round(order_qty, 0)
                resto = order_qty % 100
                order_qty = order_qty - resto + 100
    except Exception as inst:
        print(inst)



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
    resto = qty % 100
    qty = qty - resto
    if qty < 100:
        qty = 100
    print("Order qty: %.1f" % (qty,))
    await exchange.create_market_sell_order(symbol="XBTUSD", amount=qty, params={
        'orderQty': qty,
    })
    orders = await getOpenPositions(exchange, ["XBTUSD"])


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
    resto = qty % 100
    qty = qty - resto
    if qty < 100:
        qty = 100
    print("Order qty: %.1f" % (qty,))
    await exchange.create_market_buy_order(symbol="XBTUSD", amount=qty, params={
        'orderQty': qty,
    })
    await getOpenPositions(exchange, ["XBTUSD"])


async def lastSupportResistance(exchange):
    global last_support
    global last_resistance
    global last_order_price
    global trail_qty
    global price_change_average_lenght
    global last_price_change
    global last_res_sup_check
    global time_last_order
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
    trail_qty = int(t / c / 10)
    last_price_change = ohlcv[-1][2] / (ohlcv[-1][2] - abs(ohlcv[-1][2] - ohlcv[-1][3]) - trail_qty) * 100 - 100
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

    f = open('../last_support', 'w')
    f.write(str(last_support))
    f.close()
    f = open('../last_resistance', 'w')
    f.write(str(last_resistance))
    f.close()

    print("Last support: " + str(last_support))
    print("Last resistance: " + str(last_resistance))

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


async def fetchPrice(exchange):
    global last_support
    global last_resistance
    global treshold_price
    global order_qty
    global last_price_change
    global pct_refill
    global PCNT_REFILL
    global time_last_order
    global short_long
    global pct_new_level
    global last_order_price
    start = time.time()
    pcnt_refill = PCNT_REFILL
    await getAvailableBalance(exchange, 'XBt')
    try:
        while True:
            trades = await exchange.watch_trades(symbol='XBTUSD', since=(exchange.milliseconds() - (60 * 1000)))
            print("Last Res: {%2.f} - Last Supp: {%2.f} - Last price: {%2.f} - Last price change: {%2.f}" % (
                last_resistance, last_support, trades[-1]['price'], (last_resistance * treshold_price),))
            if trades[-1]['price'] < last_support - (last_support * treshold_price) and last_support > 0 and short_long == 's' and last_support != last_order_price:
                await placeOrderSell(exchange, order_qty)
                last_order_price = last_support
                writeOpen([time.time(), trades[-1]['price'], last_resistance, last_support, "SHORT"])
                break
            elif trades[-1]['price'] > last_resistance + (last_resistance * treshold_price) and last_resistance > 0 and short_long == 'l' and last_resistance != last_order_price:
                await placeOrderBuy(exchange, order_qty * 2)
                last_order_price = last_resistance
                writeOpen([time.time(), trades[-1]['price'], last_resistance, last_support, "LONG"])
                break

            if start + 30 <= time.time():
                break
    except Exception as inst:
        print(inst)
        pass


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


async def fetchPriceToClose(exchange):
    global last_support
    global last_resistance
    global last_order_price
    global last_price
    global positions
    global fees
    global acceptable_loss
    global real_last_order_price
    global liquidation_price
    global current_leverage
    global unrealizedPnlPcnt
    global available_balance
    global margin_used
    global current_order_value

    avgEntryPrice = float(positions[0]['avgEntryPrice'])
    sl_price_buy = avgEntryPrice - (avgEntryPrice * acceptable_loss)
    sl_price_sell = avgEntryPrice + (avgEntryPrice * acceptable_loss)
    pair_price_buy = avgEntryPrice + (avgEntryPrice * fees)
    pair_price_sell = avgEntryPrice - (avgEntryPrice * fees)

    start = time.time()
    try:
        while True:
            trades = await exchange.watch_trades(symbol='XBTUSD', since=(exchange.milliseconds() - (60 * 1000)))
            currPrice = trades[-1]['price']
            if int(positions[0]['currentQty']) < 0:  # SELL
                print("Last price: %2.f - Exit price: %2.f - SL price: %2.f - TrailQTY: %2.f - Pairprice: %2.f" % (
                    currPrice, last_price + trail_qty, sl_price_sell, trail_qty, pair_price_sell,))
                if last_price == 0:
                    last_price = currPrice
                if currPrice < last_price:
                    last_price = currPrice
                elif (currPrice > last_price + trail_qty and currPrice <= pair_price_sell) or currPrice > sl_price_sell:
                    await placeOrderBuy(exchange, abs(int(positions[0]['currentQty'])))
                    break
            else:

                print("Last price: %2.f - Exit price: %2.f - SL price: %2.f - TrailQTY: %2.f - Pairprice: %2.f" % (
                    currPrice, last_price - trail_qty, sl_price_buy, trail_qty, pair_price_buy,))
                if last_price == 0:
                    last_price = currPrice
                if currPrice > last_price:
                    last_price = currPrice
                elif (currPrice < last_price - trail_qty and currPrice >= pair_price_buy) or currPrice < sl_price_buy:
                    await placeOrderSell(exchange, 0 - int(positions[0]['currentQty']))
                    break
            if start + 60 <= time.time():
                break
    except Exception as inst:
        print(inst)
        pass


async def fetchPositionToClose(exchange):
    global last_support
    global last_resistance
    global last_order_price
    global last_price
    global positions
    global fees
    global acceptable_loss
    global real_last_order_price
    global liquidation_price
    global current_leverage
    global unrealizedPnlPcnt
    global available_balance
    global margin_used
    global current_order_value
    global max_unrealized_pnl
    global order_qty
    global pct_refill
    global pct_close_order
    global time_last_order
    global pct_new_level_rebuy
    time_last_order = time.time()
    start = time.time()
    try:
        while True:
            canClose = False

            old_support = last_support
            old_resistance = last_resistance
            await getOpenPositions(exchange, ['XBTUSD'])
            await getAvailableBalance(exchange, 'XBt')

            await lastSupportResistance(exchange)
            if max_unrealized_pnl < unrealizedPnlPcnt:
                max_unrealized_pnl = unrealizedPnlPcnt

            if unrealizedPnlPcnt > (fees * 100):
                canClose = True

            await checkLiquidation(exchange)
            print("Current PNL: %.4f" % (unrealizedPnlPcnt,))
            print("Max PNL: %.4f - PNL = %.4f" % (max_unrealized_pnl, max_unrealized_pnl - unrealizedPnlPcnt))

            mult_order_qty = abs(int(positions[0]['currentQty']))

            if int(positions[0]['currentQty']) < 0:  # SELL
                if last_resistance > old_resistance and \
                        (
                          (last_resistance / old_resistance * 100) > pct_new_level_rebuy or
                          (old_resistance / last_resistance * 100) > pct_new_level_rebuy
                        ) and \
                        old_resistance > 0 and \
                        unrealizedPnlPcnt < 0:
                    print("Add position qty: %.1f" % (mult_order_qty))
                    await placeOrderSell(exchange, mult_order_qty)
                    await sendRebuy(mult_order_qty * 2)
                    break
                if canClose == True and (max_unrealized_pnl - unrealizedPnlPcnt) > pcnt_close_order:
                    print("Close position qty: %.1f" % (abs(int(positions[0]['currentQty']))))
                    await placeOrderBuy(exchange, abs(int(positions[0]['currentQty'])))
                    max_unrealized_pnl = 0
                    writeClose([time.time(), unrealizedPnlPcnt, abs(int(positions[0]['currentQty'])), "SHORT"])
                    break
            else:  # BUY
                if last_support < old_support and \
                        (
                                (old_support / last_support * 100) > pct_new_level_rebuy or
                                (last_support / old_support * 100) > pct_new_level_rebuy
                        ) and \
                        old_support > 0 and unrealizedPnlPcnt < 0:
                    print("Add position qty: %.1f" % (mult_order_qty))
                    await placeOrderBuy(exchange, mult_order_qty)
                    await sendRebuy(mult_order_qty * 2)
                    break
                if canClose == True and (max_unrealized_pnl - unrealizedPnlPcnt) > pcnt_close_order:
                    print("Close position qty: %.1f" % (0 - int(positions[0]['currentQty'])))
                    await placeOrderSell(exchange, 0 - int(positions[0]['currentQty']))
                    max_unrealized_pnl = 0
                    writeClose([time.time(), unrealizedPnlPcnt, 0 - int(positions[0]['currentQty']), "LONG"])
                    break
            if start + 120 <= time.time():
                break
            else:
                time.sleep(10)
    except Exception as inst:
        print(inst)
        pass
async def fetchPositionToClosePct(exchange):
    global last_support
    global last_resistance
    global last_order_price
    global last_price
    global positions
    global fees
    global acceptable_loss
    global real_last_order_price
    global liquidation_price
    global current_leverage
    global unrealizedPnlPcnt
    global available_balance
    global margin_used
    global current_order_value
    global max_unrealized_pnl
    global order_qty
    global pct_refill
    global pct_close_order
    global time_last_order
    global pct_new_level_rebuy
    global pct_stop_loss
    global pcnt_increase_refill
    time_last_order = time.time()
    start = time.time()
    try:
        while True:
            canClose = False
            await getOpenPositions(exchange, ['XBTUSD'])
            await getAvailableBalance(exchange, 'XBt')

            if max_unrealized_pnl < unrealizedPnlPcnt:
                max_unrealized_pnl = unrealizedPnlPcnt

            if unrealizedPnlPcnt > (fees * 100):
                canClose = True

            await checkLiquidation(exchange)
            print("Current PNL: %.4f" % (unrealizedPnlPcnt,))
            print("Max PNL: %.4f - PNL = %.4f" % (max_unrealized_pnl, max_unrealized_pnl - unrealizedPnlPcnt))

            mult_order_qty = order_qty * abs(pcnt_refill) * 10 / 2

            if int(positions[0]['currentQty']) < 0:  # SELL
                if unrealizedPnlPcnt <= pcnt_stop_loss:
                    print("Close position qty: %.1f" % (abs(int(positions[0]['currentQty']))))
                    await placeOrderBuy(exchange, abs(int(positions[0]['currentQty'])))
                    max_unrealized_pnl = 0
                    pcnt_refill = PCNT_REFILL
                    writeClose([time.time(), unrealizedPnlPcnt, abs(int(positions[0]['currentQty'])), "SHORT"])
                    break
                elif unrealizedPnlPcnt < pcnt_refill:
                    print("Add position qty: %.1f" % (mult_order_qty))
                    await placeOrderSell(exchange, mult_order_qty)
                    pcnt_refill = pcnt_refill - 0.2
                    break
                elif canClose == True and (max_unrealized_pnl - unrealizedPnlPcnt) > pcnt_close_order:
                    print("Close position qty: %.1f" % (abs(int(positions[0]['currentQty']))))
                    await placeOrderBuy(exchange, abs(int(positions[0]['currentQty'])))
                    max_unrealized_pnl = 0
                    writeClose([time.time(), unrealizedPnlPcnt, abs(int(positions[0]['currentQty'])), "SHORT"])
                    break
            else: # BUY
                if unrealizedPnlPcnt <= pcnt_stop_loss:
                    print("Close position qty: %.1f" % (0 - int(positions[0]['currentQty'])))
                    await placeOrderSell(exchange, 0 - int(positions[0]['currentQty']))
                    max_unrealized_pnl = 0
                    pcnt_refill = PCNT_REFILL
                    writeClose([time.time(), unrealizedPnlPcnt, 0 - int(positions[0]['currentQty']), "LONG"])
                    break
                elif unrealizedPnlPcnt < pcnt_refill:
                    print("Add position qty: %.1f" % (mult_order_qty))
                    await placeOrderBuy(exchange, mult_order_qty)
                    pcnt_refill = pcnt_refill - 0.2
                    break
                elif canClose == True and (max_unrealized_pnl - unrealizedPnlPcnt) > pcnt_close_order:
                    print("Close position qty: %.1f" % (0 - int(positions[0]['currentQty'])))
                    await placeOrderSell(exchange, 0 - int(positions[0]['currentQty']))
                    max_unrealized_pnl = 0
                    pcnt_refill = PCNT_REFILL
                    writeClose([time.time(), unrealizedPnlPcnt, 0 - int(positions[0]['currentQty']), "LONG"])
                    break
            if start + 120 <= time.time():
                break
            else:
                time.sleep(10)
    except Exception as inst:
        print(inst)
        pass


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
                f = open('../short', 'w')
            else:
                f = open('../long', 'w')
            f.write(str(abs(int(p[i]['currentQty']))))
            f.close()
            last_entry_price = abs(int(float(p[i]['avgEntryPrice'])))

async def main():
    global last_support
    global last_resistance
    global last_order_price
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
                    f = open('../short', 'w')
                else:
                    f = open('../long', 'w')
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
                time.sleep(10)
            else:
                time.sleep(1)
        except Exception as inst:
            print(inst)
            time.sleep(10)

    await bitmex.close()

# Funzione per aprire un rebuy nella posizione opposta
async def sendRebuy(qty):
    global short_long
    if short_long == 's':
        subprocess.run(["python3", "main.py", "long", "rebuy", str(qty)])
    else:
        subprocess.run(["python3", "main.py", "short", "rebuy", str(qty)])

async def rebuy(qty):
    global short_long
    global positions
    global current_leverage
    try:
        bitmex = await openExchange(short_long)
        await getOpenPositions(bitmex, ["XBTUSD"])
        await getAvailableBalance(bitmex, 'XBt')
        if len(positions) == 0:
            await bitmex.close()
            return True
        current_position = abs(int(positions[0]['currentQty']))
        if qty - current_position > 0:
            qty = qty - current_position

        if short_long == 's':
            await placeOrderSell(bitmex, qty)
        else:
            await placeOrderBuy(bitmex, qty)
        await bitmex.close()
    except Exception as inst:
        print(inst)
        await rebuy(qty)

def read_last_supp_res():
    global last_support
    global last_resistance
    try:
        f = open('../last_support', 'r')
        last_support = float(f.readline())
        f.close()
    except Exception as inst:
        print(inst)
        last_support = 0
    try:
        f = open('../last_resistance', 'r')
        last_resistance = float(f.readline())
        f.close()
    except Exception as inst:
        print(inst)
        last_resistance = 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    read_last_supp_res()
    if 'test' in sys.argv:
        asyncio.get_event_loop().run_until_complete(test())
        exit()
    if 'short' in sys.argv or 'SHORT' in sys.argv:
        short_long = 's'
    else:
        short_long = 'l'
    if 'rebuy' in sys.argv:
        qty = float(sys.argv[3])
        if qty > 0:
            asyncio.get_event_loop().run_until_complete(rebuy(qty))
    else:
        asyncio.get_event_loop().run_until_complete(main())
