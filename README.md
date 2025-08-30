
# Trade Signal API

A **Trade Signal API** é uma aplicação que gera sinais de compra, venda e manutenção para ativos financeiros com base em candles OHLCV (Open, High, Low, Close, Volume). A API utiliza uma combinação de médias móveis exponenciais (EMA), índice de força relativa (RSI), média de verdade (ATR), entre outros indicadores técnicos para gerar esses sinais.

## Estratégias e Indicadores Usados

A API usa uma combinação de indicadores técnicos para gerar sinais de compra, venda ou manutenção:

### 1. **EMA (Exponential Moving Average)**
   - **Descrição**: A EMA é uma média móvel ponderada que dá mais peso aos preços mais recentes. Utilizada para identificar a direção da tendência e possíveis reversões.
   - **Estratégia**: O sinal de compra é gerado quando a EMA de curto prazo (rápida) cruza para cima a EMA de longo prazo (lenta), indicando um possível movimento de alta. O sinal de venda ocorre quando a EMA rápida cruza para baixo a EMA lenta, indicando um possível movimento de baixa.

### 2. **RSI (Relative Strength Index)**
   - **Descrição**: O RSI mede a força e a velocidade de um movimento de preços. Variando de 0 a 100, valores acima de 70 indicam que o ativo está sobrecomprado (potencial venda), e valores abaixo de 30 indicam que está sobrevendido (potencial compra).
   - **Estratégia**: A estratégia utiliza o RSI para filtrar sinais de compra e venda. Quando o RSI está acima de 70, a compra é evitada, e quando está abaixo de 30, a venda é evitada.

### 3. **ATR (Average True Range)**
   - **Descrição**: O ATR mede a volatilidade do mercado, calculando a média do intervalo verdadeiro (True Range) de preços durante um período.
   - **Estratégia**: Usado para calcular os níveis de **Stop Loss** (SL) e **Take Profit** (TP). O ATR é multiplicado por fatores definidos para definir essas distâncias em relação ao preço de entrada.

### 4. **Perfis de Investidor**:
   A aplicação oferece diferentes configurações de parâmetros para diferentes perfis de investidores. Cada perfil usa valores diferentes para os parâmetros da estratégia, ajustando o risco e a volatilidade:

   - **Conservador**: Foca em proteção e baixo risco. Gera sinais mais estáveis e evita movimentos agressivos.
   - **Balanceado**: Mistura proteção com busca por retorno. Aceita um pouco mais de risco, mas ainda tem um controle razoável.
   - **Agressivo**: Foca em maximizar os retornos, aceitando maior volatilidade e risco. Esse perfil gera sinais mais rápidos e com maior chance de lucro, mas também pode resultar em perdas maiores.

## Parâmetros da Estratégia

### 1. **`ema_fast` (9)**:
   - **Descrição**: Período da EMA rápida. Um valor menor faz a média reagir mais rápido aos movimentos do mercado.
   - **Exemplo**: Um valor de **9** significa que a média será calculada considerando os últimos 9 períodos.
   - **Exemplo de Uso**:
     - **Conservador**: `ema_fast = 9`
     - **Agressivo**: `ema_fast = 5`

### 2. **`ema_slow` (21)**:
   - **Descrição**: Período da EMA lenta. Um valor maior faz a média reagir de maneira mais suave, filtrando mais os sinais.
   - **Exemplo**: Um valor de **21** significa que a média será calculada considerando os últimos 21 períodos.
   - **Exemplo de Uso**:
     - **Conservador**: `ema_slow = 30`
     - **Agressivo**: `ema_slow = 15`

### 3. **`rsi_period` (14)**:
   - **Descrição**: Período para o cálculo do RSI, que ajuda a identificar condições de sobrecompra ou sobrevenda no ativo.
   - **Exemplo**: Um valor de **14** significa que o RSI será calculado com base nos últimos 14 candles.
   - **Exemplo de Uso**:
     - **Balanceado**: `rsi_period = 14`
     - **Agressivo**: `rsi_period = 10`

### 4. **`rsi_buy` (30.0)**:
   - **Descrição**: Valor abaixo do qual um ativo é considerado sobrevendido e um sinal de compra pode ser gerado.
   - **Exemplo**: Um valor de **30** significa que se o RSI estiver abaixo de 30, o ativo está sobrevendido e um sinal de compra pode ser gerado.
   - **Exemplo de Uso**:
     - **Conservador**: `rsi_buy = 40.0`
     - **Agressivo**: `rsi_buy = 25.0`

### 5. **`rsi_sell` (70.0)**:
   - **Descrição**: Valor acima do qual um ativo é considerado sobrecomprado e um sinal de venda pode ser gerado.
   - **Exemplo**: Um valor de **70** significa que se o RSI estiver acima de 70, o ativo está sobrecomprado e um sinal de venda pode ser gerado.
   - **Exemplo de Uso**:
     - **Conservador**: `rsi_sell = 65.0`
     - **Agressivo**: `rsi_sell = 75.0`

### 6. **`atr_period` (14)**:
   - **Descrição**: Período utilizado para o cálculo do ATR, que ajuda a definir a volatilidade do mercado e ajustar o Stop Loss (SL) e Take Profit (TP).
   - **Exemplo**: Um valor de **14** significa que o ATR será calculado com base nos últimos 14 candles.
   - **Exemplo de Uso**:
     - **Balanceado**: `atr_period = 14`
     - **Agressivo**: `atr_period = 10`

### 7. **`atr_tp_mult` (2.0)**:
   - **Descrição**: Multiplicador utilizado para definir o Take Profit (TP) com base no ATR. Esse valor define o quanto de distância em relação ao preço de entrada o Take Profit estará.
   - **Exemplo**: Um valor de **2.0** significa que o Take Profit estará a 2 vezes o valor do ATR do preço de entrada.
   - **Exemplo de Uso**:
     - **Conservador**: `atr_tp_mult = 1.5`
     - **Agressivo**: `atr_tp_mult = 3.0`

### 8. **`atr_sl_mult` (1.0)**:
   - **Descrição**: Multiplicador utilizado para definir o Stop Loss (SL) com base no ATR. Esse valor define o quanto de distância em relação ao preço de entrada o Stop Loss estará.
   - **Exemplo**: Um valor de **1.0** significa que o Stop Loss estará a 1 vez o valor do ATR do preço de entrada.
   - **Exemplo de Uso**:
     - **Balanceado**: `atr_sl_mult = 1.0`
     - **Agressivo**: `atr_sl_mult = 0.5`

### 9. **`min_volume` (0.0)**:
   - **Descrição**: Volume mínimo necessário para que o sinal de compra ou venda seja gerado. Se o volume do candle for abaixo deste valor, o sinal será **HOLD**.
   - **Exemplo**: Um valor de **0.0** significa que não há restrição de volume.
   - **Exemplo de Uso**:
     - **Conservador**: `min_volume = 2000`
     - **Agressivo**: `min_volume = 100`

### 10. **`require_rsi_filter` (false)**:
   - **Descrição**: Indica se o filtro do RSI deve ser aplicado. Se for **True**, o sinal de compra/venda será filtrado com base nos valores de **rsi_buy** e **rsi_sell**.
   - **Exemplo**: Se for **True**, aplica o filtro RSI; se for **False**, o filtro não é aplicado.
   - **Exemplo de Uso**:
     - **Balanceado**: `require_rsi_filter = true`
     - **Agressivo**: `require_rsi_filter = false`

## Como Usar

### Executando Localmente

1. Instale as dependências:

   ```bash
   pip install -r requirements.txt
