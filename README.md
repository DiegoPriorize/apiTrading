
# Trade Signal API

A **Trade Signal API** é uma aplicação FastAPI que gera sinais de BUY, SELL ou HOLD baseados em análises de candles OHLCV (Open, High, Low, Close, Volume). Utiliza indicadores técnicos como **EMA**, **RSI**, **MACD**, **Bollinger Bands**, **Stochastic Oscillator**, **CCI**, **ADX**, e **Parabolic SAR**. Além disso, fornece sugestões de **Take Profit (TP)** e **Stop Loss (SL)** com base na **ATR (Average True Range)**, estima a precisão de acerto e pode projetar sinais futuros com base em padrões históricos.

## Índice

- [Configuração](#configuração)
- [Uso](#uso)
- [Estratégia](#estratégia)
- [Perfis de Estratégia](#perfis-de-estratégia)
- [Explicação dos Parâmetros](#explicação-dos-parâmetros)
- [Dockerfile](#dockerfile)
- [Exemplo de Requisição](#exemplo-de-requisição)
- [Configuração Dependências](#configuração-dependências)

## Configuração

### Dependências

O projeto requer **Python 3.9+** e as seguintes dependências:

- `fastapi` — Framework web para criar APIs.
- `uvicorn` — Servidor ASGI para executar a aplicação FastAPI.
- `pydantic` — Biblioteca para validação de dados e modelos.
- `numpy` — Utilizado para otimização de cálculos numéricos (opcional).
- `matplotlib` — Para geração de gráficos de backtest (opcional).

Para instalar as dependências, use:

```bash
pip install -r requirements.txt
```

### Docker

Se você preferir rodar a aplicação em um ambiente **Docker**, utilize o Dockerfile para criar o contêiner.

1. Construa a imagem Docker:

```bash
docker build -t trade-signal-api .
```

2. Rode o contêiner:

```bash
docker run -p 8000:8000 trade-signal-api
```

Isso irá expor a API na porta 8000.

## Uso

A API é composta por dois principais endpoints:

- `/signal`: Recebe candles OHLCV e parâmetros de estratégia e retorna um sinal de **BUY**, **SELL** ou **HOLD**.
- `/backtest`: Realiza um backtest sobre os candles enviados e retorna a performance da estratégia com base nos sinais gerados.

### Exemplos de uso:

#### POST /signal

Envia candles e parâmetros de estratégia para gerar um sinal em tempo real.

#### POST /backtest

Envia candles e parâmetros para realizar o backtest da estratégia.

### Como Enviar Dados:

O formato de entrada deve ser o seguinte:

```json
{
    "T": [1622494800, 1622495100, 1622495400],
    "O": ["100.50", "101.00", "100.80"],
    "C": ["101.00", "100.90", "101.10"],
    "H": ["101.50", "101.30", "101.20"],
    "I": ["100.40", "100.60", "100.50"],
    "V": ["1200", "1500", "1400"],
    "params": { ... }
}
```

Onde:
- `T` é o timestamp.
- `O` é o preço de abertura.
- `C` é o preço de fechamento.
- `H` é a máxima do período.
- `I` é a mínima do período.
- `V` é o volume.

### Formato da Resposta:

A resposta será uma estrutura com os detalhes do sinal gerado:

```json
{
    "signal": "BUY",
    "timestamp": 1622495100,
    "iso_time": "2021-05-31T10:05:00Z",
    "price": 101.10,
    "reason": "Sinal baseado no crossover de EMAs.",
    "take_profit": 102.50,
    "stop_loss": 100.00,
    "confidence": 85.0,
    "future_signals": [
        {
            "signal": "BUY",
            "timestamp": 1622495400,
            "iso_time": "2021-05-31T10:10:00Z",
            "confidence": 75.0
        }
    ]
}
```

## Estratégia

A estratégia de sinais utiliza indicadores técnicos para gerar sinais de compra, venda ou manutenção. A lógica é baseada em **crossover de EMAs** (Exponential Moving Averages) e outros filtros, como **RSI**, **MACD**, **Bollinger Bands**, **Stochastic Oscillator**, **CCI**, **ADX** e **Parabolic SAR**.

### EMA Crossovers:
- Sinal de **BUY** quando a **EMA rápida** cruza acima da **EMA lenta**; 
- **SELL** quando ocorre o inverso.

### RSI:
- Se RSI for maior que um valor de venda ou menor que um valor de compra, o sinal é **HOLD**.

### MACD:
- Um crossover entre o **MACD** e sua linha de sinal.

### Bollinger Bands:
- Sinal de **BUY** quando o preço toca ou ultrapassa a **banda inferior** e começa a subir, e **SELL** quando o preço toca ou ultrapassa a **banda superior** e começa a cair.

### Stochastic Oscillator:
- Sinal de **BUY** quando o %K cruza acima do %D em uma zona de **sobrevenda** (<20) e **SELL** em uma zona de **sobrecompra** (>80).

### CCI:
- **BUY** quando o CCI cruza acima de 100 e **SELL** quando cruza abaixo de -100.

### ADX:
- Determina a força da tendência. **BUY** se o +DI for maior que o -DI e o **ADX** for maior que o limiar de 25. **SELL** se o -DI for maior que o +DI e o **ADX** for maior que 25.

### Parabolic SAR:
- O sinal de **BUY** ocorre quando o **SAR** está abaixo do preço e **SELL** quando está acima.

## Perfis de Estratégia

Os perfis são configurações prontas para ajustar o comportamento da estratégia para diferentes tipos de trader. Cada perfil ajusta a quantidade mínima de confirmações necessárias, o limiar do ADX e o horizonte futuro para sinais projetados.

### Perfil Agressivo

- `min_confirmations`: 1
- `adx_threshold`: 20
- `future_horizon_bars`: 15

Ideal para traders que desejam sinais mais rápidos e frequentes, aceitando mais risco.

### Perfil Balanceado

- `min_confirmations`: 2
- `adx_threshold`: 25
- `future_horizon_bars`: 20

Adequado para traders que buscam um equilíbrio entre rapidez e confiabilidade nos sinais.

### Perfil Conservador

- `min_confirmations`: 3
- `adx_threshold`: 30
- `future_horizon_bars`: 30

Destinado a traders que priorizam a confiabilidade e estão dispostos a aceitar menos sinais, mas com maior precisão.

## Explicação dos Parâmetros

Aqui está uma explicação detalhada do que cada parâmetro faz e como afeta a estratégia:

- `ema_fast (int)`: Período da EMA rápida. Controla a sensibilidade para detecção de movimentos rápidos.
- `ema_slow (int)`: Período da EMA lenta. Controla a suavização para detecção de tendências de longo prazo.
- `rsi_period (int)`: Período do RSI. Determina a quantidade de dados usados para calcular o RSI.
- `rsi_buy (float)`: Valor abaixo do qual o RSI indica que o mercado está sobrevendido (potencial compra).
- `rsi_sell (float)`: Valor acima do qual o RSI indica que o mercado está sobrecomprado (potencial venda).
- `atr_period (int)`: Período do ATR (Average True Range), usado para calcular a volatilidade.
- `atr_tp_mult (float)`: Multiplicador para calcular o Take Profit a partir da ATR.
- `atr_sl_mult (float)`: Multiplicador para calcular o Stop Loss a partir da ATR.
- `macd_fast, macd_slow, macd_signal (int)`: Períodos utilizados para o cálculo do MACD e sua linha de sinal.
- `bb_period (int)`: Período para o cálculo das Bandas de Bollinger.
- `bb_std_mult (float)`: Multiplicador do desvio padrão para as Bandas de Bollinger.
- `stoch_k (int)`: Período para o cálculo da linha %K no Oscilador Estocástico.
- `stoch_d (int)`: Período para o cálculo da linha %D no Oscilador Estocástico.
- `cci_period (int)`: Período para o cálculo do CCI (Commodity Channel Index).
- `adx_period (int)`: Período para o cálculo do ADX (Average Directional Index).
- `adx_threshold (float)`: Limiar do ADX para considerar uma tendência forte.
- `psar_step (float)`: Passo para o cálculo do Parabolic SAR.
- `psar_max (float)`: Valor máximo para o Parabolic SAR.
- `min_confirmations (int)`: Número mínimo de confirmações de diferentes indicadores necessários para gerar um sinal de BUY ou SELL.
- `eval_max_bars (int)`: Número máximo de barras após o sinal para verificar se Take Profit (TP) ou Stop Loss (SL) foram atingidos.
- `future_horizon_bars (int)`: Número de barras futuras a serem analisadas para projetar sinais.
- `analog_min_score (int)`: Pontuação mínima de similaridade para considerar sinais futuros como válidos.
- `future_top_k (int)`: Número máximo de sinais futuros projetados para serem retornados.

## Dockerfile

A aplicação já vem com um Dockerfile configurado para facilitar o uso em ambiente de produção. Ele utiliza o `python:3.9-slim` e o `uvicorn` para rodar a API.

Para construir e rodar o Docker:

```bash
docker build -t trade-signal-api .
docker run -p 8000:8000 trade-signal-api
```

Isso expõe a API na porta 8000.
