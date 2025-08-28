from typing import List, Optional, Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime
import math

app = FastAPI(title="Trade Signal API", version="1.0.0", description="Generates BUY/SELL/HOLD signals from OHLCV candles using EMA cross + RSI filter and suggests SL/TP via ATR.")

# --------- Data models ---------
class Candle(BaseModel):
    timestamp: int = Field(..., description="Unix timestamp (seconds) for the candle close")
    open: float
    high: float
    low: float
    close: float
    volume: float

class StrategyParams(BaseModel):
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_buy: float = 30.0
    rsi_sell: float = 70.0
    atr_period: int = 14
    atr_tp_mult: float = 2.0
    atr_sl_mult: float = 1.0
    min_volume: float = 0.0
    require_rsi_filter: bool = False  # if True, apply RSI thresholds as a filter

class SignalResponse(BaseModel):
    signal: Literal["BUY","SELL","HOLD"]
    timestamp: int
    iso_time: str
    price: float
    reason: str
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None

class BacktestResponse(BaseModel):
    signals: List[SignalResponse]
    count_buy: int
    count_sell: int

# --------- Indicator helpers (pure Python, no pandas) ---------
def ema(values: List[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be > 0")
    k = 2 / (period + 1)
    out: List[Optional[float]] = [None] * len(values)
    if len(values) == 0:
        return out
    # seed with SMA of first 'period' values if available, else first value
    if len(values) >= period:
        seed = sum(values[:period]) / period
        out[period-1] = seed
        prev = seed
        for i in range(period, len(values)):
            prev = values[i] * k + prev * (1 - k)
            out[i] = prev
    else:
        # not enough values, use progressive EMA
        prev = values[0]
        out[0] = prev
        for i in range(1, len(values)):
            prev = values[i] * k + prev * (1 - k)
            out[i] = prev
    return out

def rsi(values: List[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be > 0")
    rsi_vals: List[Optional[float]] = [None] * len(values)
    gains = [0.0] * len(values)
    losses = [0.0] * len(values)
    for i in range(1, len(values)):
        chg = values[i] - values[i-1]
        gains[i] = max(chg, 0.0)
        losses[i] = max(-chg, 0.0)
    # average gains/losses
    if len(values) > period:
        avg_gain = sum(gains[1:period+1]) / period
        avg_loss = sum(losses[1:period+1]) / period
        rs = (avg_gain / avg_loss) if avg_loss != 0 else math.inf
        rsi_vals[period] = 100 - (100 / (1 + rs))
        for i in range(period+1, len(values)):
            avg_gain = (avg_gain * (period-1) + gains[i]) / period
            avg_loss = (avg_loss * (period-1) + losses[i]) / period
            rs = (avg_gain / avg_loss) if avg_loss != 0 else math.inf
            rsi_vals[i] = 100 - (100 / (1 + rs))
    return rsi_vals

def atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> List[Optional[float]]:
    trs: List[float] = [0.0] * len(highs)
    for i in range(len(highs)):
        if i == 0:
            trs[i] = highs[i] - lows[i]
        else:
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            trs[i] = max(tr1, tr2, tr3)
    out: List[Optional[float]] = [None] * len(highs)
    if len(trs) >= period:
        # seed with SMA (Wilder's smoothing can also be used)
        seed = sum(trs[1:period+1]) / period
        out[period] = seed
        prev = seed
        for i in range(period+1, len(trs)):
            prev = (prev * (period - 1) + trs[i]) / period
            out[i] = prev
    return out

# --------- Strategy logic ---------
def generate_signal(candles: List[Candle], params: StrategyParams) -> SignalResponse:
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]

    ema_fast = ema(closes, params.ema_fast)
    ema_slow = ema(closes, params.ema_slow)
    rsi_vals = rsi(closes, params.rsi_period)
    atr_vals = atr(highs, lows, closes, params.atr_period)

    i = len(candles) - 1  # last closed candle
    # basic checks
    if i < 1 or ema_fast[i] is None or ema_slow[i] is None:
        return SignalResponse(signal="HOLD",
                              timestamp=candles[i].timestamp,
                              iso_time=datetime.utcfromtimestamp(candles[i].timestamp).isoformat() + "Z",
                              price=candles[i].close,
                              reason="Insufficient data for EMA crossover.")

    # volume filter
    if volumes[i] < params.min_volume:
        return SignalResponse(signal="HOLD",
                              timestamp=candles[i].timestamp,
                              iso_time=datetime.utcfromtimestamp(candles[i].timestamp).isoformat() + "Z",
                              price=candles[i].close,
                              reason=f"Volume {volumes[i]:.2f} < min_volume ({params.min_volume}).")

    diff_now = ema_fast[i] - ema_slow[i]
    diff_prev = (ema_fast[i-1] - ema_slow[i-1]) if (ema_fast[i-1] is not None and ema_slow[i-1] is not None) else None

    signal: Literal["BUY","SELL","HOLD"] = "HOLD"
    reason_parts = []

    # crossover detection
    if diff_prev is not None:
        if diff_prev <= 0 and diff_now > 0:
            signal = "BUY"
            reason_parts.append(f"EMA{params.ema_fast} cruzou acima da EMA{params.ema_slow}.")
        elif diff_prev >= 0 and diff_now < 0:
            signal = "SELL"
            reason_parts.append(f"EMA{params.ema_fast} cruzou abaixo da EMA{params.ema_slow}.")
        else:
            reason_parts.append("Sem crossover no candle atual.")

    # optional RSI filter
    if params.require_rsi_filter and rsi_vals[i] is not None:
        r = rsi_vals[i]
        if signal == "BUY" and r > params.rsi_sell:
            signal = "HOLD"
            reason_parts.append(f"RSI {r:.1f} acima de {params.rsi_sell} (filtro bloqueou compra).")
        elif signal == "SELL" and r < params.rsi_buy:
            signal = "HOLD"
            reason_parts.append(f"RSI {r:.1f} abaixo de {params.rsi_buy} (filtro bloqueou venda).")
        else:
            reason_parts.append(f"RSI {r:.1f} aceito no filtro.")

    # SL/TP suggestions via ATR
    tp = sl = None
    if atr_vals[i] is not None and atr_vals[i] > 0:
        a = atr_vals[i]
        price = closes[i]
        if signal == "BUY":
            sl = price - params.atr_sl_mult * a
            tp = price + params.atr_tp_mult * a
        elif signal == "SELL":
            sl = price + params.atr_sl_mult * a
            tp = price - params.atr_tp_mult * a
        if signal != "HOLD":
            reason_parts.append(f"ATR ~ {a:.4f} → TP {params.atr_tp_mult}x, SL {params.atr_sl_mult}x.")

    reason = " ".join(reason_parts) if reason_parts else "Nenhum motivo específico."
    c = candles[i]
    return SignalResponse(signal=signal,
                          timestamp=c.timestamp,
                          iso_time=datetime.utcfromtimestamp(c.timestamp).isoformat() + "Z",
                          price=c.close,
                          reason=reason,
                          take_profit=tp,
                          stop_loss=sl)

def generate_backtest(candles: List[Candle], params: StrategyParams) -> BacktestResponse:
    signals: List[SignalResponse] = []
    for end in range(2, len(candles)+1):
        sig = generate_signal(candles[:end], params)
        if sig.signal in ("BUY","SELL"):
            signals.append(sig)
    buys = sum(1 for s in signals if s.signal == "BUY")
    sells = sum(1 for s in signals if s.signal == "SELL")
    return BacktestResponse(signals=signals, count_buy=buys, count_sell=sells)

# --------- API endpoints ---------
@app.get("/health")
def health():
    return {"status": "ok"}

class SignalRequest(BaseModel):
    candles: List[Candle]
    params: Optional[StrategyParams] = None

@app.post("/signal", response_model=SignalResponse)
def signal(req: SignalRequest):
    if not req.candles or len(req.candles) < 5:
        last_ts = req.candles[-1].timestamp if req.candles else int(datetime.utcnow().timestamp())
        return SignalResponse(signal="HOLD",
                              timestamp=last_ts,
                              iso_time=datetime.utcfromtimestamp(last_ts).isoformat() + "Z",
                              price=req.candles[-1].close if req.candles else 0.0,
                              reason="Envie pelo menos 5 candles para estabilidade.")
    params = req.params or StrategyParams()
    return generate_signal(req.candles, params)

@app.post("/backtest", response_model=BacktestResponse)
def backtest(req: SignalRequest):
    if not req.candles or len(req.candles) < 5:
        return BacktestResponse(signals=[], count_buy=0, count_sell=0)
    params = req.params or StrategyParams()
    return generate_backtest(req.candles, params)
