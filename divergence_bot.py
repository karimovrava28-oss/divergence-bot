"""
Divergence Signal Bot (Binance → Telegram)
Индикаторы: RSI + MACD
Таймфреймы: 15m, 1h, 4h
"""

import asyncio
import logging
from datetime import datetime
import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from scipy.signal import argrelextrema

# ─────────────────────────────────────────
#  КОНФИГ
# ─────────────────────────────────────────
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID        = "YOUR_CHAT_ID"

SYMBOLS = [SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "SOL/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "TON/USDT",
    "TRX/USDT",
    "AVAX/USDT",
    "SHIB/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "MATIC/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "UNI/USDT",
    "SUI/USDT",
    "APT/USDT",
    "NEAR/USDT",
    "OP/USDT",
    "ARB/USDT",
    "FIL/USDT",
    "ATOM/USDT",
    "INJ/USDT",
    "MKR/USDT",
    "AAVE/USDT",
    "FTM/USDT",
    "ALGO/USDT",
    "VET/USDT",
    "HBAR/USDT",
    "ICP/USDT",
    "GRT/USDT",
    "SAND/USDT",
    "MANA/USDT",
    "AXS/USDT",
    "CHZ/USDT",
    "EOS/USDT",
    "XLM/USDT",
    "THETA/USDT",
    "EGLD/USDT",
    "XTZ/USDT",
    "FLOW/USDT",
    "ROSE/USDT",
    "ZEC/USDT",
    "ENJ/USDT",
    "1INCH/USDT",
    "COMP/USDT",
    "SNX/USDT",
    "CRV/USDT",
    "WLD/USDT",
    "JUP/USDT",
    "SEI/USDT",
    "TIA/USDT",
    "PEPE/USDT",
    "FLOKI/USDT",
    "BONK/USDT",
    "WIF/USDT",
    "BOME/USDT",
    "ORDI/USDT",
    "SATS/USDT",
    "STX/USDT",
    "CFX/USDT",
    "BLUR/USDT",
    "ID/USDT",
    "ARK/USDT",
    "ACE/USDT",
    "JTO/USDT",
    "PYTH/USDT",
    "DYM/USDT",
    "ALT/USDT",
    "MANTA/USDT",
    "PIXEL/USDT",
    "PORTAL/USDT",
    "STRK/USDT",
    "ETHFI/USDT",
    "ENA/USDT",
    "W/USDT",
    "OMNI/USDT",
    "REZ/USDT",
    "BB/USDT",
    "IO/USDT",
    "ZK/USDT",
    "LISTA/USDT",
    "ZRO/USDT",
    "RENDER/USDT",
    "ONDO/USDT",
    "NOT/USDT",
    "DOGS/USDT",
    "HMSTR/USDT",
    "CATI/USDT",
    "NEIRO/USDT",
    "EIGEN/USDT",
    "SCR/USDT",
    "GRASS/USDT",
    "ACT/USDT",
    "PNUT/USDT",
    "HIPPO/USDT",
    "MOVE/USDT",
    "ME/USDT",
    "PENGU/USDT",
    "TRUMP/USDT",
    "MELANIA/USDT",
    "VINE/USDT",
]
    "

TIMEFRAMES = ["15m", "1h", "4h"]

# RSI
RSI_PERIOD  = 14
RSI_OB      = 70   # перекупленность
RSI_OS      = 30   # перепроданность

# MACD
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

# Поиск экстремумов — окно (свечей с каждой стороны)
PIVOT_ORDER = 5

# Интервал сканирования в секундах
SCAN_INTERVAL = {
    "15m": 60 * 15,
    "1h":  60 * 60,
    "4h":  60 * 60 * 4,
}

# Антиспам — не слать один и тот же сигнал дважды
sent_signals: set = set()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  ПОЛУЧЕНИЕ ДАННЫХ
# ─────────────────────────────────────────
async def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ─────────────────────────────────────────
#  ИНДИКАТОРЫ
# ─────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # RSI
    df["rsi"] = ta.rsi(df["close"], length=RSI_PERIOD)

    # MACD
    macd = ta.macd(df["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df["macd"]        = macd[f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]
    df["macd_signal"] = macd[f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]
    df["macd_hist"]   = macd[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]

    return df.dropna().reset_index(drop=True)


# ─────────────────────────────────────────
#  ЛОКАЛЬНЫЕ ЭКСТРЕМУМЫ
# ─────────────────────────────────────────
def get_pivots(series: pd.Series, order: int = PIVOT_ORDER):
    """Возвращает индексы локальных минимумов и максимумов."""
    lows  = argrelextrema(series.values, np.less_equal,    order=order)[0]
    highs = argrelextrema(series.values, np.greater_equal, order=order)[0]
    return lows, highs


# ─────────────────────────────────────────
#  ПОИСК ДИВЕРГЕНЦИЙ
# ─────────────────────────────────────────
def find_divergences(df: pd.DataFrame) -> list[dict]:
    signals = []
    close = df["close"]

    for indicator in ("rsi", "macd_hist"):
        ind = df[indicator]
        price_lows,  price_highs  = get_pivots(close)
        ind_lows,    ind_highs    = get_pivots(ind)

        # ── БЫЧЬЯ дивергенция: цена ↓↓, индикатор ↑↑ ──
        common_lows = sorted(set(price_lows) & set(ind_lows))
        for i in range(1, len(common_lows)):
            i1, i2 = common_lows[i - 1], common_lows[i]
            if i2 - i1 < PIVOT_ORDER * 2:
                continue
            price_down = close.iloc[i2] < close.iloc[i1]
            ind_up     = ind.iloc[i2]   > ind.iloc[i1]
            if price_down and ind_up:
                # Фильтр RSI: желательно быть ниже 45
                rsi_ok = df["rsi"].iloc[i2] < 45 if indicator == "rsi" else True
                if rsi_ok:
                    signals.append({
                        "type":      "🟢 БЫЧЬЯ",
                        "indicator": indicator.upper(),
                        "bar_idx":   i2,
                        "price":     round(close.iloc[i2], 6),
                        "rsi":       round(df["rsi"].iloc[i2], 1),
                    })

        # ── МЕДВЕЖЬЯ дивергенция: цена ↑↑, индикатор ↓↓ ──
        common_highs = sorted(set(price_highs) & set(ind_highs))
        for i in range(1, len(common_highs)):
            i1, i2 = common_highs[i - 1], common_highs[i]
            if i2 - i1 < PIVOT_ORDER * 2:
                continue
            price_up  = close.iloc[i2] > close.iloc[i1]
            ind_down  = ind.iloc[i2]   < ind.iloc[i1]
            if price_up and ind_down:
                rsi_ok = df["rsi"].iloc[i2] > 55 if indicator == "rsi" else True
                if rsi_ok:
                    signals.append({
                        "type":      "🔴 МЕДВЕЖЬЯ",
                        "indicator": indicator.upper(),
                        "bar_idx":   i2,
                        "price":     round(close.iloc[i2], 6),
                        "rsi":       round(df["rsi"].iloc[i2], 1),
                    })

    return signals


# ─────────────────────────────────────────
#  ФОРМАТИРОВАНИЕ СООБЩЕНИЯ
# ─────────────────────────────────────────
def format_signal(symbol: str, tf: str, signal: dict, df: pd.DataFrame) -> str:
    ts  = df["timestamp"].iloc[signal["bar_idx"]].strftime("%d.%m %H:%M")
    vol = round(df["volume"].iloc[signal["bar_idx"]], 2)

    macd_val  = round(df["macd"].iloc[signal["bar_idx"]], 4)
    macd_hist = round(df["macd_hist"].iloc[signal["bar_idx"]], 4)

    return (
        f"{signal['type']} ДИВЕРГЕНЦИЯ\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"📌 {symbol}  |  {tf}\n"
        f"🕒 {ts}\n"
        f"💰 Цена: {signal['price']}\n"
        f"📊 Индикатор: {signal['indicator']}\n"
        f"─────────────────\n"
        f"RSI:       {signal['rsi']}\n"
        f"MACD:      {macd_val}\n"
        f"MACD hist: {macd_hist}\n"
        f"Объём:     {vol}\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"⚠️ Не является финансовым советом"
    )


# ─────────────────────────────────────────
#  СКАНЕР
# ─────────────────────────────────────────
async def scan(bot: Bot):
    exchange = ccxt.binance({"enableRateLimit": True})

    try:
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                try:
                    df  = await fetch_ohlcv(exchange, symbol, tf)
                    df  = add_indicators(df)
                    sigs = find_divergences(df)

                    for sig in sigs:
                        # Антиспам ключ
                        key = f"{symbol}_{tf}_{sig['type']}_{sig['indicator']}_{sig['bar_idx']}"
                        if key in sent_signals:
                            continue

                        # Отправляем только если дивер на последних 3 барах
                        if sig["bar_idx"] >= len(df) - 3:
                            text = format_signal(symbol, tf, sig, df)
                            await bot.send_message(chat_id=CHAT_ID, text=text)
                            sent_signals.add(key)
                            log.info(f"Сигнал отправлен: {symbol} {tf} {sig['type']}")

                except Exception as e:
                    log.warning(f"Ошибка {symbol} {tf}: {e}")

                await asyncio.sleep(0.3)  # rate limit

    finally:
        await exchange.close()


# ─────────────────────────────────────────
#  КОМАНДЫ БОТА
# ─────────────────────────────────────────
async def cmd_start(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Divergence Bot запущен!\n\n"
        "/status — текущий статус\n"
        "/scan — запустить скан вручную\n"
        "/symbols — список монет"
    )

async def cmd_status(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"✅ Бот работает\n"
        f"Монеты: {', '.join(SYMBOLS)}\n"
        f"ТФ: {', '.join(TIMEFRAMES)}\n"
        f"Сигналов отправлено: {len(sent_signals)}"
    )

async def cmd_scan(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Запускаю скан...")
    await scan(context.bot)
    await update.message.reply_text("✅ Скан завершён")

async def cmd_symbols(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📋 Монеты:\n" + "\n".join(SYMBOLS))


# ─────────────────────────────────────────
#  ФОНОВЫЙ ЦИКЛ СКАНИРОВАНИЯ
# ─────────────────────────────────────────
async def background_scanner(app: Application):
    while True:
        log.info("🔍 Скан...")
        await scan(app.bot)
        await asyncio.sleep(60 * 15)  # каждые 15 минут


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("scan",    cmd_scan))
    app.add_handler(CommandHandler("symbols", cmd_symbols))

    loop = asyncio.get_event_loop()
    loop.create_task(background_scanner(app))

    log.info("🚀 Бот запущен")
    app.run_polling()


if __name__ == "__main__":
    main()
