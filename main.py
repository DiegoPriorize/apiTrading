from typing import List, Optional, Literal, Tuple, Dict
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from datetime import datetime
import math
import statistics

app = FastAPI(
    title="Trade Signal API",
    version="2.0.0",
    description=(
        "Gera sinais BUY/SELL/HOLD a partir de candles OHLCV. "
        "Agora aceita formato pagestate (T/O/C/H/I/V), inclui MACD, Bollinger, Stochastic, CCI, ADX, Parabolic SAR, "
        "estima precisão e sugere timestamps futuros mais prováveis."
    ),
)

# =============== MODELOS ===============
class Candle(BaseModel):
    timestamp: int = Field(..., description="Unix timestamp (seconds) para o fechamento do candle")
    open: float
    high: float
    low: float
    close: float
    volume: float

class StrategyParams(BaseModel):
    # Base
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    rsi_buy: float = 30.0
    rsi_sell: float = 70.0
    atr_period: int = 14
    atr_tp_mult: float = 2.0
    atr_sl_mult: float = 1.0
    min_volume: float = 0.0
    require_rsi_filter: bool = False  # se True, aplicar thresholds do RSI como filtro

    # Novos indicadores (padrões comuns)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    bb_period: int = 20
    bb_std_mult: float = 2.0

    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth_k: int = 3

    cci_period: int = 20

    adx_period: int = 14
    adx_threshold: float = 25.0  # força de tendência

    psar_step: float = 0.02
    psar_max: float = 0.2

    # Confirmações/consenso
    use_ema_cross: bool = True
    use_macd_cross: bool = True
    use_bb_touch: bool = True
    use_stochastic_cross: bool = True
    use_cci_levels: bool = True
    use_adx_trend: bool = True
    use_psar_trend: bool = True

    min_confirmations: int = 2  # mínimo de votos para emitir BUY/SELL (além de filtros)

    # Avaliação/precisão
    eval_max_bars: int = 30  # janelas para verificar TP/SL após o sinal
    # Projeção futura
    future_horizon_bars: int = 20
    future_top_k: int = 5
    analog_min_score: int = 4  # similaridade mínima de estados técnicos (0-8)

class FutureSignal(BaseModel):
    signal: Literal["BUY", "SELL"]
    timestamp: int
    iso_time: str
    confidence: float

class SignalResponse(BaseModel):
    signal: Literal["BUY", "SELL", "HOLD"]
    timestamp: int
    iso_time: str
    price: float
    reason: str
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence: Optional[float] = None  # 0-100 (%)
    future_signals: Optional[List[FutureSignal]] = None
    precision_buy: Optional[float] = None  # % de acerto (histórico/estimado)
    precision_sell: Optional[float] = None

class BacktestResponse(BaseModel):
    signals: List[SignalResponse]
    count_buy: int
    count_sell: int
    precision_buy: float
    precision_sell: float
    precision_overall: float

# --------- Novo payload: pagestate ---------
class PageStatePayload(BaseModel):
    # T: inteiros (timestamp)
    T: List[int] = Field(..., description="Lista de timestamps (segundos)")
    # O/C/H/I/V: strings numéricas (aceita '1,23' ou '1.23')
    O: List[str]
    C: List[str]
    H: List[str]
    I: List[str]
    V: List[str]
    params: Optional[StrategyParams] = None

    @validator("O", "C", "H", "I", "V", pre=True)
    def ensure_list_str(cls, v):
        if not isinstance(v, list):
            raise ValueError("Campo deve ser lista de strings")
        return v

def _to_float(x: str) -> float:
    # aceita vírgula decimal
    return float(x.replace(",", "."))

def pagestate_to_candles(ps: PageStatePayload) -> List[Candle]:
    n = len(ps.T)
    if not all(len(lst) == n for lst in [ps.O, ps.C, ps.H, ps.I, ps.V]):
        raise ValueError("Listas T/O/C/H/I/V devem ter o mesmo comprimento.")
    candles: List[Candle] = []
    for t, o, c, h, i, v in zip(ps.T, ps.O, ps.C, ps.H, ps.I, ps.V):
        candles.append(
            Candle(
                timestamp=int(t),
                open=_to_float(o),
                close=_to_float(c),
                high=_to_float(h),
                low=_to_float(i),
                volume=_to_float(v),
            )
        )
    return candles

# =============== INDICADORES (Python puro) ===============
def ema(values: List[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be > 0")
    k = 2 / (period + 1)
    out: List[Optional[float]] = [None] * len(values)
    if len(values) == 0:
        return out
    if len(values) >= period:
        seed = sum(values[:period]) / period
        out[period - 1] = seed
        prev = seed
        for i in range(period, len(values)):
            prev = values[i] * k + prev * (1 - k)
            out[i] = prev
    else:
        prev = values[0]
        out[0] = prev
        for i in range(1, len(values)):
            prev = values[i] * k + prev * (1 - k)
            out[i] = prev
    return out

def sma(values: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(values) < period:
        return out
    window = sum(values[:period])
    out[period - 1] = window / period
    for i in range(period, len(values)):
        window += values[i] - values[i - period]
        out[i] = window / period
    return out

def stddev(values: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return out
    # cálculo simples, custo O(n*period) — ok para tamanhos comuns
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        m = sum(window) / period
        var = sum((x - m) ** 2 for x in window) / period
        out[i] = var ** 0.5
    return out

def rsi(values: List[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be > 0")
    rsi_vals: List[Optional[float]] = [None] * len(values)
    gains = [0.0] * len(values)
    losses = [0.0] * len(values)
    for i in range(1, len(values)):
        chg = values[i] - values[i - 1]
        gains[i] = max(chg, 0.0)
        losses[i] = max(-chg, 0.0)
    if len(values) > period:
        avg_gain = sum(gains[1 : period + 1]) / period
        avg_loss = sum(losses[1 : period + 1]) / period
        rs = (avg_gain / avg_loss) if avg_loss != 0 else math.inf
        rsi_vals[period] = 100 - (100 / (1 + rs))
        for i in range(period + 1, len(values)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
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
            tr2 = abs(highs[i] - closes[i - 1])
            tr3 = abs(lows[i] - closes[i - 1])
            trs[i] = max(tr1, tr2, tr3)
    out: List[Optional[float]] = [None] * len(highs)
    if len(trs) >= period:
        seed = sum(trs[1 : period + 1]) / period
        out[period] = seed
        prev = seed
        for i in range(period + 1, len(trs)):
            prev = (prev * (period - 1) + trs[i]) / period  # Wilder
            out[i] = prev
    return out

# ---- MACD ----
def macd(values: List[float], fast: int, slow: int, signal: int) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line: List[Optional[float]] = [None] * len(values)
    for i in range(len(values)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]
    signal_line = ema([x if x is not None else 0.0 for x in macd_line], signal)
    hist: List[Optional[float]] = [None if macd_line[i] is None or signal_line[i] is None else macd_line[i] - signal_line[i] for i in range(len(values))]
    return macd_line, signal_line, hist

# ---- Bollinger Bands ----
def bollinger(values: List[float], period: int, std_mult: float) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    mid = sma(values, period)
    sd = stddev(values, period)
    upper = [None if mid[i] is None or sd[i] is None else mid[i] + std_mult * sd[i] for i in range(len(values))]
    lower = [None if mid[i] is None or sd[i] is None else mid[i] - std_mult * sd[i] for i in range(len(values))]
    return lower, mid, upper

# ---- Stochastic Oscillator ----
def stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int, d_period: int, smooth_k: int) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    n = len(closes)
    raw_k: List[Optional[float]] = [None] * n
    for i in range(n):
        if i < k_period - 1:
            continue
        hh = max(highs[i - k_period + 1 : i + 1])
        ll = min(lows[i - k_period + 1 : i + 1])
        denom = (hh - ll) if (hh - ll) != 0 else 1e-12
        raw_k[i] = 100.0 * ((closes[i] - ll) / denom)
    # smooth %K
    def _ma(arr: List[Optional[float]], p: int) -> List[Optional[float]]:
        out: List[Optional[float]] = [None] * len(arr)
        buf: List[float] = []
        for i, v in enumerate(arr):
            if v is None:
                buf.append(float('nan'))
            else:
                buf.append(v)
            if i >= p - 1:
                window = [x for x in buf[i - p + 1 : i + 1] if not math.isnan(x)]
                out[i] = sum(window) / len(window) if window else None
        return out
    k_smooth = _ma(raw_k, smooth_k) if smooth_k > 1 else raw_k
    d_line = _ma(k_smooth, d_period)
    return k_smooth, d_line

# ---- CCI ----
def cci(highs: List[float], lows: List[float], closes: List[float], period: int) -> List[Optional[float]]:
    tp = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(len(closes))]
    sma_tp = sma(tp, period)
    out: List[Optional[float]] = [None] * len(tp)
    for i in range(len(tp)):
        if i < period - 1 or sma_tp[i] is None:
            continue
        mean_dev = sum(abs(tp[j] - sma_tp[i]) for j in range(i - period + 1, i + 1)) / period
        denom = 0.015 * mean_dev if mean_dev != 0 else 1e-12
        out[i] = (tp[i] - sma_tp[i]) / denom
    return out

# ---- ADX (+DI, -DI) ----
def adx(highs: List[float], lows: List[float], closes: List[float], period: int) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    n = len(highs)
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i - 1])
        tr3 = abs(lows[i] - closes[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    def wilder_smooth(arr: List[float], p: int) -> List[Optional[float]]:
        out: List[Optional[float]] = [None] * len(arr)
        if len(arr) <= p:
            return out
        seed = sum(arr[1 : p + 1])
        out[p] = seed
        prev = seed
        for i in range(p + 1, len(arr)):
            prev = prev - (prev / p) + arr[i]
            out[i] = prev
        return out
    tr_s = wilder_smooth(tr, period)
    plus_s = wilder_smooth(plus_dm, period)
    minus_s = wilder_smooth(minus_dm, period)

    plus_di: List[Optional[float]] = [None] * n
    minus_di: List[Optional[float]] = [None] * n
    dx: List[Optional[float]] = [None] * n
    for i in range(n):
        if tr_s[i] is None or tr_s[i] == 0:
            continue
        plus_di[i] = 100.0 * (plus_s[i] / tr_s[i]) if plus_s[i] is not None else None
        minus_di[i] = 100.0 * (minus_s[i] / tr_s[i]) if minus_s[i] is not None else None
        if plus_di[i] is not None and minus_di[i] is not None:
            denom = (plus_di[i] + minus_di[i])
            dx[i] = 100.0 * (abs(plus_di[i] - minus_di[i]) / denom) if denom != 0 else 0.0

    # ADX (média móvel de DX)
    adx_vals = [None] * n
    if n > 2 * period:
        # seed
        seed = sum(x for x in dx[period + 1 : 2 * period + 1] if x is not None) / period
        adx_vals[2 * period] = seed
        prev = seed
        for i in range(2 * period + 1, n):
            if dx[i] is None:
                continue
            prev = (prev * (period - 1) + dx[i]) / period
            adx_vals[i] = prev
    return adx_vals, plus_di, minus_di

# ---- Parabolic SAR ----
def parabolic_sar(highs: List[float], lows: List[float], step: float = 0.02, max_step: float = 0.2) -> List[Optional[float]]:
    n = len(highs)
    out: List[Optional[float]] = [None] * n
    if n < 2:
        return out
    # Inicialização: trend up/down baseado nos 2 primeiros candles
    up_trend = highs[1] > highs[0]
    ep = highs[1] if up_trend else lows[1]  # extreme point
    sar = lows[0] if up_trend else highs[0]
    af = step
    out[1] = sar
    for i in range(2, n):
        sar = sar + af * (ep - sar)
        # Ajustes para não furar o candle
        if up_trend:
            sar = min(sar, lows[i - 1], lows[i - 2])
        else:
            sar = max(sar, highs[i - 1], highs[i - 2])
        # Verificar reversão
        if (up_trend and lows[i] < sar) or (not up_trend and highs[i] > sar):
            up_trend = not up_trend
            sar = ep
            ep = highs[i] if up_trend else lows[i]
            af = step
        else:
            if up_trend:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + step, max_step)
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + step, max_step)
        out[i] = sar
    return out

# =============== FUNÇÕES AUXILIARES ===============
def bar_seconds(candles: List[Candle]) -> Optional[int]:
    if len(candles) < 2:
        return None
    diffs = [candles[i].timestamp - candles[i - 1].timestamp for i in range(1, len(candles))]
    diffs = [d for d in diffs if d > 0]
    if not diffs:
        return None
    try:
        return int(statistics.median(diffs))
    except Exception:
        return diffs[-1]

def _vote(score: int) -> Literal["BUY","SELL","HOLD"]:
    if score > 0:
        return "BUY"
    elif score < 0:
        return "SELL"
    return "HOLD"

def _pct(x: float) -> float:
    return max(0.0, min(100.0, x))

# =============== LÓGICA DE ESTRATÉGIA ===============
def generate_signal(candles: List[Candle], params: StrategyParams) -> SignalResponse:
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]

    i = len(candles) - 1  # último candle fechado
    if i < 1:
        ts = candles[i].timestamp if candles else int(datetime.utcnow().timestamp())
        return SignalResponse(signal="HOLD",
                              timestamp=ts,
                              iso_time=datetime.utcfromtimestamp(ts).isoformat() + "Z",
                              price=candles[i].close if candles else 0.0,
                              reason="Insuficiente para cálculo.")

    # Indicadores
    ema_fast_vals = ema(closes, params.ema_fast)
    ema_slow_vals = ema(closes, params.ema_slow)
    rsi_vals = rsi(closes, params.rsi_period)
    atr_vals = atr(highs, lows, closes, params.atr_period)
    macd_line, macd_sig, macd_hist = macd(closes, params.macd_fast, params.macd_slow, params.macd_signal)
    bb_lower, bb_mid, bb_upper = bollinger(closes, params.bb_period, params.bb_std_mult)
    stoch_k, stoch_d = stochastic(highs, lows, closes, params.stoch_k, params.stoch_d, params.stoch_smooth_k)
    cci_vals = cci(highs, lows, closes, params.cci_period)
    adx_vals, plus_di, minus_di = adx(highs, lows, closes, params.adx_period)
    psar_vals = parabolic_sar(highs, lows, params.psar_step, params.psar_max)

    # Filtros básicos
    if ema_fast_vals[i] is None or ema_slow_vals[i] is None:
        return SignalResponse(signal="HOLD",
                              timestamp=candles[i].timestamp,
                              iso_time=datetime.utcfromtimestamp(candles[i].timestamp).isoformat() + "Z",
                              price=candles[i].close,
                              reason="Insuficiente para EMA.")
    if volumes[i] < params.min_volume:
        return SignalResponse(signal="HOLD",
                              timestamp=candles[i].timestamp,
                              iso_time=datetime.utcfromtimestamp(candles[i].timestamp).isoformat() + "Z",
                              price=candles[i].close,
                              reason=f"Volume {volumes[i]:.2f} < min_volume ({params.min_volume}).")

    reason_parts: List[str] = []
    votes = 0
    used = 0

    # --- EMA crossover (principal) ---
    if params.use_ema_cross:
        diff_now = ema_fast_vals[i] - ema_slow_vals[i]
        diff_prev = None
        if ema_fast_vals[i - 1] is not None and ema_slow_vals[i - 1] is not None:
            diff_prev = ema_fast_vals[i - 1] - ema_slow_vals[i - 1]
        if diff_prev is not None:
            used += 1
            if diff_prev <= 0 and diff_now > 0:
                votes += 1
                reason_parts.append(f"EMA{params.ema_fast} cruzou acima da EMA{params.ema_slow}.")
            elif diff_prev >= 0 and diff_now < 0:
                votes -= 1
                reason_parts.append(f"EMA{params.ema_fast} cruzou abaixo da EMA{params.ema_slow}.")
            else:
                reason_parts.append("EMA: sem crossover no candle atual.")

    # --- MACD crossover ---
    if params.use_macd_cross and macd_line[i] is not None and macd_sig[i] is not None and macd_line[i-1] is not None and macd_sig[i-1] is not None:
        used += 1
        if macd_line[i - 1] <= macd_sig[i - 1] and macd_line[i] > macd_sig[i]:
            votes += 1
            reason_parts.append("MACD cruzou acima da linha de sinal.")
        elif macd_line[i - 1] >= macd_sig[i - 1] and macd_line[i] < macd_sig[i]:
            votes -= 1
            reason_parts.append("MACD cruzou abaixo da linha de sinal.")
        else:
            reason_parts.append("MACD: sem crossover.")

    # --- Bollinger touch/break + reversão simples ---
    if params.use_bb_touch and bb_lower[i] is not None and bb_upper[i] is not None:
        used += 1
        if closes[i] <= bb_lower[i]:
            votes += 1
            reason_parts.append("Preço tocou/rompeu banda inferior (potencial compra).")
        elif closes[i] >= bb_upper[i]:
            votes -= 1
            reason_parts.append("Preço tocou/rompeu banda superior (potencial venda).")
        else:
            reason_parts.append("Bollinger: dentro das bandas.")

    # --- Stochastic cross em zonas ---
    if params.use_stochastic_cross and stoch_k[i] is not None and stoch_d[i] is not None and stoch_k[i-1] is not None and stoch_d[i-1] is not None:
        used += 1
        if stoch_k[i - 1] <= stoch_d[i - 1] and stoch_k[i] > stoch_d[i] and min(stoch_k[i], stoch_d[i]) < 20:
            votes += 1
            reason_parts.append("Stochastic: %K cruzou acima de %D em zona de sobrevenda (<20).")
        elif stoch_k[i - 1] >= stoch_d[i - 1] and stoch_k[i] < stoch_d[i] and max(stoch_k[i], stoch_d[i]) > 80:
            votes -= 1
            reason_parts.append("Stochastic: %K cruzou abaixo de %D em zona de sobrecompra (>80).")
        else:
            reason_parts.append("Stochastic: sem sinal forte.")

    # --- CCI níveis ---
    if params.use_cci_levels and cci_vals[i] is not None:
        used += 1
        if cci_vals[i] > 100:
            votes += 1
            reason_parts.append("CCI acima de +100 (tendência de alta).")
        elif cci_vals[i] < -100:
            votes -= 1
            reason_parts.append("CCI abaixo de -100 (tendência de baixa).")
        else:
            reason_parts.append("CCI neutro.")

    # --- ADX (+DI/-DI) ---
    if params.use_adx_trend and adx_vals[i] is not None and plus_di[i] is not None and minus_di[i] is not None:
        used += 1
        if adx_vals[i] >= params.adx_threshold:
            if plus_di[i] > minus_di[i]:
                votes += 1
                reason_parts.append(f"ADX {adx_vals[i]:.1f} forte; +DI > -DI (alta).")
            elif minus_di[i] > plus_di[i]:
                votes -= 1
                reason_parts.append(f"ADX {adx_vals[i]:.1f} forte; -DI > +DI (baixa).")
        else:
            reason_parts.append(f"ADX {adx_vals[i]:.1f} fraco.")

    # --- Parabolic SAR posição relativa ---
    if params.use_psar_trend and psar_vals[i] is not None:
        used += 1
        if psar_vals[i] < closes[i]:
            votes += 1
            reason_parts.append("Parabolic SAR abaixo do preço (tendência de alta).")
        elif psar_vals[i] > closes[i]:
            votes -= 1
            reason_parts.append("Parabolic SAR acima do preço (tendência de baixa).")
        else:
            reason_parts.append("Parabolic SAR neutro.")

    # Resultado por votos mínimos
    signal: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    if abs(votes) >= params.min_confirmations:
        signal = _vote(votes)
    else:
        reason_parts.append(f"Confirmações insuficientes (votos={votes}, min={params.min_confirmations}).")

    # Filtro RSI opcional
    if params.require_rsi_filter and rsi_vals[i] is not None:
        r = rsi_vals[i]
        if signal == "BUY" and r > params.rsi_sell:
            signal = "HOLD"
            reason_parts.append(f"RSI {r:.1f} > {params.rsi_sell} — compra bloqueada pelo filtro.")
        elif signal == "SELL" and r < params.rsi_buy:
            signal = "HOLD"
            reason_parts.append(f"RSI {r:.1f} < {params.rsi_buy} — venda bloqueada pelo filtro.")
        else:
            reason_parts.append(f"RSI {r:.1f} aceito no filtro.")

    # SL/TP via ATR
    tp = sl = None
    if signal != "HOLD" and atr_vals[i] is not None and atr_vals[i] > 0:
        a = atr_vals[i]
        price = closes[i]
        if signal == "BUY":
            sl = price - params.atr_sl_mult * a
            tp = price + params.atr_tp_mult * a
        elif signal == "SELL":
            sl = price + params.atr_sl_mult * a
            tp = price - params.atr_tp_mult * a
        reason_parts.append(f"ATR ~ {a:.4f} → TP {params.atr_tp_mult}x, SL {params.atr_sl_mult}x.")

    # Confiança (heurística): proporção de votos + força de ADX normalizada
    confidence = None
    if used > 0:
        base = abs(votes) / max(params.min_confirmations, 1)
        base = min(base, 1.0)
        adx_boost = 0.0
        if adx_vals[i] is not None:
            adx_boost = min(max((adx_vals[i] - 20) / 40.0, 0.0), 1.0) * 0.3  # 0 a 0.3
        confidence = _pct((base * 0.7 + adx_boost) * 100.0)

    c = candles[i]
    resp = SignalResponse(
        signal=signal,
        timestamp=c.timestamp,
        iso_time=datetime.utcfromtimestamp(c.timestamp).isoformat() + "Z",
        price=c.close,
        reason=" ".join(reason_parts) if reason_parts else "Sem motivo específico.",
        take_profit=tp,
        stop_loss=sl,
        confidence=confidence,
    )

    # Métricas históricas rápidas (precisão BUY/SELL) com base nos candles fornecidos
    prec_buy, prec_sell = quick_precision(candles, params)
    resp.precision_buy = prec_buy
    resp.precision_sell = prec_sell

    # Projeções futuras
    resp.future_signals = predict_future_signals(candles, params)
    return resp

def quick_precision(candles: List[Candle], params: StrategyParams) -> Tuple[float, float]:
    """Backtest rápido: cada sinal abre trade com TP/SL de ATR; acerto = bate TP antes do SL dentro de eval_max_bars."""
    signals: List[Tuple[int, str, float, float]] = []  # (idx, signal, tp, sl)
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    atr_vals = atr(highs, lows, closes, params.atr_period)

    # gerar sinais históricos simples (reutiliza votos sem heavy calc completo por performance? aqui chamaremos generate_signal parcial)
    buys_total = sells_total = 0
    buys_hit = sells_hit = 0

    for end in range(max(5, params.ema_slow + 2), len(candles) - 1):
        sub = candles[: end + 1]
        s = generate_signal_light(sub, params, precomputed_atr=atr_vals[: end + 1])
        if s.signal in ("BUY", "SELL") and s.take_profit is not None and s.stop_loss is not None:
            if s.signal == "BUY":
                buys_total += 1
            else:
                sells_total += 1
            # avaliar até eval_max_bars
            tp = s.take_profit
            sl = s.stop_loss
            win = evaluate_trade(candles, end, s.signal, tp, sl, params.eval_max_bars)
            if s.signal == "BUY" and win:
                buys_hit += 1
            if s.signal == "SELL" and win:
                sells_hit += 1

    prec_buy = _pct(100.0 * buys_hit / buys_total) if buys_total > 0 else 0.0
    prec_sell = _pct(100.0 * sells_hit / sells_total) if sells_total > 0 else 0.0
    return prec_buy, prec_sell

def evaluate_trade(candles: List[Candle], idx: int, side: str, tp: float, sl: float, max_bars: int) -> bool:
    """Retorna True se TP atinge antes do SL em até max_bars após idx."""
    end = min(len(candles) - 1, idx + max_bars)
    if side == "BUY":
        for j in range(idx + 1, end + 1):
            if candles[j].low <= sl:
                return False
            if candles[j].high >= tp:
                return True
    else:  # SELL
        for j in range(idx + 1, end + 1):
            if candles[j].high >= sl:
                return False
            if candles[j].low <= tp:
                return True
    return False

def generate_signal_light(candles: List[Candle], params: StrategyParams, precomputed_atr: Optional[List[Optional[float]]] = None) -> SignalResponse:
    """Versão leve para backtest/precisão (mesma lógica de votos)."""
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]
    i = len(candles) - 1

    ema_fast_vals = ema(closes, params.ema_fast)
    ema_slow_vals = ema(closes, params.ema_slow)
    macd_line, macd_sig, _ = macd(closes, params.macd_fast, params.macd_slow, params.macd_signal)
    bb_lower, bb_mid, bb_upper = bollinger(closes, params.bb_period, params.bb_std_mult)
    stoch_k, stoch_d = stochastic(highs, lows, closes, params.stoch_k, params.stoch_d, params.stoch_smooth_k)
    cci_vals = cci(highs, lows, closes, params.cci_period)
    adx_vals, plus_di, minus_di = adx(highs, lows, closes, params.adx_period)
    psar_vals = parabolic_sar(highs, lows, params.psar_step, params.psar_max)
    rsi_vals = rsi(closes, params.rsi_period)

    if ema_fast_vals[i] is None or ema_slow_vals[i] is None or volumes[i] < params.min_volume:
        return SignalResponse(signal="HOLD", timestamp=candles[i].timestamp, iso_time=datetime.utcfromtimestamp(candles[i].timestamp).isoformat()+"Z", price=candles[i].close, reason="")

    votes = 0
    used = 0
    if params.use_ema_cross:
        diff_now = ema_fast_vals[i] - ema_slow_vals[i]
        if ema_fast_vals[i-1] is not None and ema_slow_vals[i-1] is not None:
            diff_prev = ema_fast_vals[i-1] - ema_slow_vals[i-1]
            used += 1
            if diff_prev <= 0 and diff_now > 0:
                votes += 1
            elif diff_prev >= 0 and diff_now < 0:
                votes -= 1
    if params.use_macd_cross and macd_line[i] is not None and macd_sig[i] is not None and macd_line[i-1] is not None and macd_sig[i-1] is not None:
        used += 1
        if macd_line[i - 1] <= macd_sig[i - 1] and macd_line[i] > macd_sig[i]:
            votes += 1
        elif macd_line[i - 1] >= macd_sig[i - 1] and macd_line[i] < macd_sig[i]:
            votes -= 1
    if params.use_bb_touch and bb_lower[i] is not None and bb_upper[i] is not None:
        used += 1
        if closes[i] <= bb_lower[i]:
            votes += 1
        elif closes[i] >= bb_upper[i]:
            votes -= 1
    if params.use_stochastic_cross and stoch_k[i] is not None and stoch_d[i] is not None and stoch_k[i-1] is not None and stoch_d[i-1] is not None:
        used += 1
        if stoch_k[i - 1] <= stoch_d[i - 1] and stoch_k[i] > stoch_d[i] and min(stoch_k[i], stoch_d[i]) < 20:
            votes += 1
        elif stoch_k[i - 1] >= stoch_d[i - 1] and stoch_k[i] < stoch_d[i] and max(stoch_k[i], stoch_d[i]) > 80:
            votes -= 1
    if params.use_cci_levels and cci_vals[i] is not None:
        used += 1
        if cci_vals[i] > 100:
            votes += 1
        elif cci_vals[i] < -100:
            votes -= 1
    if params.use_adx_trend and adx_vals[i] is not None and plus_di[i] is not None and minus_di[i] is not None and adx_vals[i] >= params.adx_threshold:
        used += 1
        if plus_di[i] > minus_di[i]:
            votes += 1
        elif minus_di[i] > plus_di[i]:
            votes -= 1
    if params.use_psar_trend and psar_vals[i] is not None:
        used += 1
        if psar_vals[i] < closes[i]:
            votes += 1
        elif psar_vals[i] > closes[i]:
            votes -= 1

    signal: Literal["BUY","SELL","HOLD"] = "HOLD"
    if abs(votes) >= params.min_confirmations:
        signal = "BUY" if votes > 0 else "SELL"

    # filtro RSI
    if params.require_rsi_filter and rsi_vals[i] is not None:
        r = rsi_vals[i]
        if signal == "BUY" and r > params.rsi_sell:
            signal = "HOLD"
        elif signal == "SELL" and r < params.rsi_buy:
            signal = "HOLD"

    tp = sl = None
    a = None
    if precomputed_atr and len(precomputed_atr) == len(candles):
        a = precomputed_atr[i]
    else:
        a = atr(highs, lows, closes, params.atr_period)[i]
    if signal != "HOLD" and a is not None and a > 0:
        price = closes[i]
        if signal == "BUY":
            sl = price - params.atr_sl_mult * a
            tp = price + params.atr_tp_mult * a
        else:
            sl = price + params.atr_sl_mult * a
            tp = price - params.atr_tp_mult * a

    return SignalResponse(
        signal=signal,
        timestamp=candles[i].timestamp,
        iso_time=datetime.utcfromtimestamp(candles[i].timestamp).isoformat()+"Z",
        price=candles[i].close,
        reason="",
        take_profit=tp,
        stop_loss=sl
    )

def predict_future_signals(candles: List[Candle], params: StrategyParams) -> List[FutureSignal]:
    """Projeção por análogos: busca estados técnicos similares no histórico e estima distribuição do tempo até o próximo sinal.
    Retorna até future_top_k sugestões com confiança percentual."""
    if len(candles) < max(60, params.ema_slow + params.future_horizon_bars + 5):
        return []  # pouco histórico

    # Pré-cálculo leve
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    ema_f = ema(closes, params.ema_fast)
    ema_s = ema(closes, params.ema_slow)
    macd_line, macd_sig, _ = macd(closes, params.macd_fast, params.macd_slow, params.macd_signal)
    bb_lower, _, bb_upper = bollinger(closes, params.bb_period, params.bb_std_mult)
    stoch_k, stoch_d = stochastic(highs, lows, closes, params.stoch_k, params.stoch_d, params.stoch_smooth_k)
    cci_vals = cci(highs, lows, closes, params.cci_period)
    adx_vals, plus_di, minus_di = adx(highs, lows, closes, params.adx_period)
    psar_vals = parabolic_sar(highs, lows, params.psar_step, params.psar_max)

    def feat(idx: int) -> Optional[Tuple[int, int, int, int, int, int, int, int]]:
        if any(x is None for x in [ema_f[idx], ema_s[idx], macd_line[idx], macd_sig[idx], bb_lower[idx], bb_upper[idx], stoch_k[idx], stoch_d[idx], cci_vals[idx], adx_vals[idx], plus_di[idx], minus_di[idx], psar_vals[idx]]):
            return None
        ema_sign = 1 if (ema_f[idx] - ema_s[idx]) > 0 else -1
        macd_sign = 1 if (macd_line[idx] - macd_sig[idx]) > 0 else -1
        bb_zone = -1 if closes[idx] <= bb_lower[idx] else (1 if closes[idx] >= bb_upper[idx] else 0)
        st_cross = 1 if (stoch_k[idx] > stoch_d[idx] and min(stoch_k[idx], stoch_d[idx]) < 50) else (-1 if (stoch_k[idx] < stoch_d[idx] and max(stoch_k[idx], stoch_d[idx]) > 50) else 0)
        cci_zone = 1 if cci_vals[idx] > 100 else (-1 if cci_vals[idx] < -100 else 0)
        adx_trend = 0
        if adx_vals[idx] >= params.adx_threshold:
            adx_trend = 1 if (plus_di[idx] > minus_di[idx]) else -1
        psar_trend = 1 if psar_vals[idx] < closes[idx] else -1
        rsi_zone = 0  # opcionalmente use RSI, mas para manter leve vamos omitir aqui
        return (ema_sign, macd_sign, bb_zone, st_cross, cci_zone, adx_trend, psar_trend, rsi_zone)

    last = len(candles) - 1
    f_last = feat(last)
    if f_last is None:
        return []

    # Percorre histórico (exclui últimos future_horizon_bars para ter espaço de projeção)
    horizon = params.future_horizon_bars
    matches: List[Tuple[int, int]] = []  # (steps_a_frente, +1 BUY / -1 SELL)
    for idx in range(params.ema_slow + 2, len(candles) - horizon - 1):
        f_i = feat(idx)
        if f_i is None:
            continue
        # similaridade por contagem de componentes iguais
        score = sum(1 for a, b in zip(f_last, f_i) if a == b)
        if score >= params.analog_min_score:
            # encontre o próximo sinal real após idx usando a lógica leve
            for ahead in range(1, horizon + 1):
                sub = candles[: idx + ahead + 1]
                s = generate_signal_light(sub, params)
                if s.signal in ("BUY", "SELL"):
                    matches.append((ahead, 1 if s.signal == "BUY" else -1))
                    break

    if not matches:
        return []

    # Estatística: prob por (ahead, side)
    buckets: Dict[Tuple[int, int], int] = {}
    side_totals = {1: 0, -1: 0}
    for a, side in matches:
        buckets[(a, side)] = buckets.get((a, side), 0) + 1
        side_totals[side] += 1

    # Ordena por frequência (maior primeiro), depois por antecedência menor
    ordered = sorted(buckets.items(), key=lambda kv: (-kv[1], kv[0][0]))

    sec = bar_seconds(candles) or 60
    out: List[FutureSignal] = []
    used_pairs = 0
    for (ahead, side), freq in ordered:
        if used_pairs >= params.future_top_k:
            break
        # confiança relativa = freq / total do lado + pequeno ajuste por preferência do lado predominante
        base_conf = freq / max(1, side_totals[side])
        # normaliza para 50–95%
        conf = 0.5 + 0.45 * base_conf
        ts = candles[last].timestamp + ahead * sec
        out.append(
            FutureSignal(
                signal="BUY" if side == 1 else "SELL",
                timestamp=ts,
                iso_time=datetime.utcfromtimestamp(ts).isoformat() + "Z",
                confidence=_pct(conf * 100.0),
            )
        )
        used_pairs += 1
    return out

def generate_backtest(candles: List[Candle], params: StrategyParams) -> BacktestResponse:
    signals: List[SignalResponse] = []
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    atr_vals = atr(highs, lows, closes, params.atr_period)

    buys_total = sells_total = 0
    buys_hit = sells_hit = 0

    for end in range(max(5, params.ema_slow + 2), len(candles)):
        sub = candles[: end + 1]
        sig = generate_signal_light(sub, params, precomputed_atr=atr_vals[: end + 1])
        if sig.signal in ("BUY", "SELL"):
            signals.append(sig)
            if sig.take_profit is not None and sig.stop_loss is not None:
                if sig.signal == "BUY":
                    buys_total += 1
                else:
                    sells_total += 1
                win = evaluate_trade(candles, end, sig.signal, sig.take_profit, sig.stop_loss, params.eval_max_bars)
                if sig.signal == "BUY" and win:
                    buys_hit += 1
                if sig.signal == "SELL" and win:
                    sells_hit += 1

    prec_buy = _pct(100.0 * buys_hit / buys_total) if buys_total > 0 else 0.0
    prec_sell = _pct(100.0 * sells_hit / sells_total) if sells_total > 0 else 0.0
    total = buys_total + sells_total
    overall = _pct(100.0 * (buys_hit + sells_hit) / total) if total > 0 else 0.0

    return BacktestResponse(
        signals=signals,
        count_buy=sum(1 for s in signals if s.signal == "BUY"),
        count_sell=sum(1 for s in signals if s.signal == "SELL"),
        precision_buy=prec_buy,
        precision_sell=prec_sell,
        precision_overall=overall,
    )

# =============== ENDPOINTS ===============
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
        return SignalResponse(
            signal="HOLD",
            timestamp=last_ts,
            iso_time=datetime.utcfromtimestamp(last_ts).isoformat() + "Z",
            price=req.candles[-1].close if req.candles else 0.0,
            reason="Envie pelo menos 5 candles para estabilidade."
        )
    params = req.params or StrategyParams()
    return generate_signal(req.candles, params)

@app.post("/backtest", response_model=BacktestResponse)
def backtest(req: SignalRequest):
    if not req.candles or len(req.candles) < 5:
        return BacktestResponse(signals=[], count_buy=0, count_sell=0, precision_buy=0.0, precision_sell=0.0, precision_overall=0.0)
    params = req.params or StrategyParams()
    return generate_backtest(req.candles, params)

# ---- Endpoints para formato pagestate (FlutterFlow) ----
@app.post("/signal_pagestate", response_model=SignalResponse)
def signal_pagestate(ps: PageStatePayload):
    candles = pagestate_to_candles(ps)
    if len(candles) < 5:
        last_ts = candles[-1].timestamp if candles else int(datetime.utcnow().timestamp())
        return SignalResponse(
            signal="HOLD",
            timestamp=last_ts,
            iso_time=datetime.utcfromtimestamp(last_ts).isoformat() + "Z",
            price=candles[-1].close if candles else 0.0,
            reason="Envie pelo menos 5 candles para estabilidade (pagestate)."
        )
    params = ps.params or StrategyParams()
    return generate_signal(candles, params)

@app.post("/backtest_pagestate", response_model=BacktestResponse)
def backtest_pagestate(ps: PageStatePayload):
    candles = pagestate_to_candles(ps)
    if len(candles) < 5:
        return BacktestResponse(signals=[], count_buy=0, count_sell=0, precision_buy=0.0, precision_sell=0.0, precision_overall=0.0)
    params = ps.params or StrategyParams()
    return generate_backtest(candles, params)
