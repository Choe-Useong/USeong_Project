import pandas as pd
import yfinance as yf
import statsmodels.tsa.arima.model
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import root_scalar
from pykrx import stock, bond
from datetime import datetime
from scipy.optimize import minimize
import re



data = pd.read_excel(r"C:\Users\admin\Desktop\갭스.xlsx", header= None)
data = data.drop(index = [0,1,2,3,4,5,6,7,10,11,13])
data = data.set_index(0)
data= data.T

data = data.melt(
    id_vars=data.columns[0:3],          # 그대로 유지할 열
    value_vars=data.columns[3:],  # 세로로 변환할 열
    var_name='날짜',                  # 새로 생길 '변수 이름' 열
    value_name='값'                  # 값이 들어갈 열 이름
)


data['날짜'] = pd.to_datetime(data['날짜'])
data.set_index(data['날짜'], inplace = True)
data.drop(columns = '날짜', inplace = True)
data = data.reset_index()
data = data.pivot(index = ['날짜','Symbol', 'Symbol Name'], columns = 'Item Name', values = '값')
data = data.apply(pd.to_numeric, errors='coerce').reset_index()
data['날짜'] = pd.to_datetime(data['날짜'])
data = data.set_index('날짜')




dataI = pd.read_excel(r"C:\Users\admin\Desktop\갭스지수.xlsx", header= None)
dataI = dataI.drop(index = [0,1,2,3,4,5,6,7,10,11,13])
dataI = dataI.set_index(0)
dataI= dataI.T

dataI = dataI.melt(
    id_vars=dataI.columns[0:3],          # 그대로 유지할 열
    value_vars=dataI.columns[3:],  # 세로로 변환할 열
    var_name='날짜',                  # 새로 생길 '변수 이름' 열
    value_name='값'                  # 값이 들어갈 열 이름
)


dataI['날짜'] = pd.to_datetime(dataI['날짜'])
dataI.set_index(dataI['날짜'], inplace = True)
dataI.drop(columns = '날짜', inplace = True)
dataI = dataI.reset_index()
dataI = dataI.pivot(index = ['날짜','Symbol', 'Symbol Name'], columns = 'Item Name', values = '값')
dataI = dataI.apply(pd.to_numeric, errors='coerce').reset_index()
dataI['날짜'] = pd.to_datetime(dataI['날짜'])
dataI = dataI.set_index('날짜')

#코스피 2,538,235,151 * 백만
#코스닥 414,507,549*백만



def RIM_value_with_terminal(r_annual, B0, roe_forecast):
    """
    r_annual: 연율 기준 할인율 (예: 0.10은 10%)
    B0: 현재 북밸류 (예: 주당 BPS * 주식 수)
    roe_forecast: 연율화된 ROE 예측값 (% 단위, 예: 15.0이면 15%)
    """
    # 연율 → 분기율 변환
    r_q = (1 + r_annual) ** (1 / 4) - 1
    
    Bt = B0
    value = B0
    
    for h in range(1, len(roe_forecast) + 1):
        roe_annual = roe_forecast.iloc[h - 1] / 100
        roe_q = (1 + roe_annual) ** (1/4) - 1  # 연율 ROE → 분기 ROE
        RI = (roe_q - r_q) * Bt
        value += RI / (1 + r_q)**h
        Bt *= (1 + roe_q)
    
    # Terminal value
    terminal_roe_annual = roe_forecast.iloc[-1] / 100
    terminal_roe_q = (1 + terminal_roe_annual) ** (1/4) - 1
    terminal_RI = (terminal_roe_q - r_q) * Bt
    TV = terminal_RI / (r_q * (1 + r_q)**len(roe_forecast))
    
    return value + TV


from pandas.tseries.offsets import MonthEnd
import numpy as np



def compute_daily_beta_1y(sym_code, market_code):
    try:
        from datetime import datetime
        import yfinance as yf
        import numpy as np
        import pandas as pd

        print(f"\n⏳ {sym_code} vs {market_code} 베타 계산 시작")

        end_date = datetime.today()
        start_date = end_date - pd.DateOffset(years=1)
        print(f"📅 기간: {start_date.date()} ~ {end_date.date()}")

        # DataFrame으로 명시적 다운로드
        stock_df = yf.download(sym_code, start=start_date, end=end_date, interval='1d')[['Close']]
        market_df = yf.download(market_code, start=start_date, end=end_date, interval='1d')[['Close']]
        print("📥 다운로드 완료")

        # 열 이름 명시적 변경
        stock_df = stock_df.rename(columns={'Close': 'stock'})
        market_df = market_df.rename(columns={'Close': 'market'})

        # 날짜 기준 병합
        merged = pd.merge(stock_df, market_df, left_index=True, right_index=True).dropna()
        print("📊 병합 후 데이터 수:", len(merged))

        if len(merged) < 30:
            print(f"⚠️ 데이터 너무 짧음 ({len(merged)}일)")
            return None

        # 로그 수익률 계산
        merged['stock_ret'] = np.log(merged['stock'] / merged['stock'].shift(1))
        merged['market_ret'] = np.log(merged['market'] / merged['market'].shift(1))
        merged = merged.dropna()
        print(f"✅ 수익률 계산 완료. 관측치 수: {len(merged)}")

        # numpy로 변환 후 베타 계산
        x = merged['market_ret'].to_numpy()
        y = merged['stock_ret'].to_numpy()
        beta = np.cov(y, x)[0, 1] / np.var(x)
        print(f"✅ 최종 베타: β = {beta:.4f}")
        return beta

    except Exception as e:
        print(f"💥 {sym_code} 베타 계산 실패: {e}")
        return None




# 시장 시가총액 (단위: 원)
market_caps = {
    '코스피': 2538235151 * 1_000_000,
    '코스닥': 414507549 * 1_000_000
}




result_market_list = []

for market in ['코스피', '코스닥']:
    try:
        market_data = dataI[dataI['Symbol Name'].str.contains(market)]
        
        # ROE 시계열
        roe_series = market_data['ROE(지배주주순이익)(%)'].dropna()
        
        # 북밸류 B0
        B0 = market_data['지배주주지분(원)'].dropna().tail(1).values[0]
        
        if len(roe_series) <= 20:
            forecast = pd.Series([roe_series.mean()] * 40)
        else:
            model = ARIMA(roe_series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 4))
            result = model.fit()
            forecast = result.get_forecast(steps=40).predicted_mean
        
        market_cap = market_caps[market]

        # 내재 r 찾기
        def residual_r(r_annual):
            return RIM_value_with_terminal(r_annual, B0, forecast) - market_cap

        r_result = root_scalar(residual_r, bracket=[0.001, 0.9], method='brentq')

        result_market_list.append({
            '시장명': market,
            'r 추정치': r_result.root
        })
    
    except Exception as e:
        print(f"{market} 처리 중 오류 발생: {e}")



market_df = pd.DataFrame(result_market_list)
print(market_df)



# 오늘 날짜를 'YYYYMMDD' 문자열로 변환
today = datetime.today().strftime('%Y%m%d')

# 오늘 날짜로 데이터 요청
rf = bond.get_otc_treasury_yields(today).loc['국고채 10년','수익률'] / 100

# ETF 리스트 지정
etf_list = ['091160', '091180']  # 원하는 ETF 코드를 여기에 추가

# 결과 저장용
merged_dict = {}
etf_returns = {}

for etf_code in etf_list:
    print(f"\n📦 ETF {etf_code} 처리 시작")

    try:
        # 1. ETF 구성 정보 불러오기
        df = stock.get_etf_portfolio_deposit_file(etf_code).reset_index()
        df['심볼A'] = 'A' + df['티커']
        df['비중'] = pd.to_numeric(df['비중']) / 100
        df = df[['심볼A', '비중']]

        # 2. 종목 이름 매칭
        matched_names = data[data['Symbol'].isin(df['심볼A'])]['Symbol Name'].unique().tolist()
        result_list = []

        for name in matched_names:
            try:
                firm_data = data[data['Symbol Name'] == name]
                roe_series = firm_data['ROE(지배주주순이익)(%)'].dropna()
                B0 = firm_data['지배주주지분(원)'].dropna().tail(1).values[0]
                symbol = firm_data['Symbol'].astype(str).iloc[-1]
                base_code = symbol[1:]

                # sym_code 및 시장 구분
                try:
                    sym_code = base_code + '.KS'
                    ticker = yf.Ticker(sym_code)
                    market_cap = ticker.info.get('marketCap')
                    if market_cap is None:
                        raise ValueError()
                    market_name = '코스피'
                    market_code = '^KS11'
                except:
                    sym_code = base_code + '.KQ'
                    ticker = yf.Ticker(sym_code)
                    market_cap = ticker.info.get('marketCap')
                    if market_cap is None:
                        raise ValueError()
                    market_name = '코스닥'
                    market_code = '^KQ11'

                # ROE forecast
                if len(roe_series) <= 20:
                    forecast = pd.Series([roe_series.mean()] * 40)
                else:
                    model = ARIMA(roe_series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 4))
                    result = model.fit()
                    forecast = result.get_forecast(steps=40).predicted_mean

                # RIM 할인율 추정
                def residual_r(r_annual):
                    return RIM_value_with_terminal(r_annual, B0, forecast) - market_cap

                try:
                    r_result = root_scalar(residual_r, bracket=[0.001, 0.9], method='brentq')
                    r_final = r_result.root
                except ValueError:
                    # 대체 방식
                    r_market = market_df[market_df['시장명'] == market_name]['r 추정치'].values[0]
                    beta = compute_daily_beta_1y(sym_code, market_code)
                    if beta is None:
                        print(f"{name} → 베타 계산 실패, 제외")
                        continue
                    r_final = beta * r_market + (1 - beta) * rf
                    print(f"{name}: r 대체됨 → β={beta:.3f}, r_market={r_market:.4f}, rf={rf:.4f} → r={r_final:.4f}")

                result_list.append({
                    '종목명': name,
                    '심볼': sym_code,
                    'r 추정치': r_final
                })

            except Exception as e:
                print(f"{name} 처리 중 오류: {e}")
                continue

        # 병합
        rim_df = pd.DataFrame(result_list)
        rim_df['심볼A'] = 'A' + rim_df['심볼'].str[:6]
        merged = pd.merge(rim_df, df, on='심볼A', how='left')

        # 기대수익률 계산
        etf_r = (merged['r 추정치'] * merged['비중']).sum()
        etf_returns[etf_code] = etf_r

        # 결과 저장
        merged_dict[etf_code] = merged
        print(f"✅ ETF {etf_code} 기대수익률: {etf_r:.4%}")

    except Exception as e:
        print(f"💥 ETF {etf_code} 처리 실패: {e}")







def optimize_weights(mu, cov, objective='sharpe', ridge=1e-3, sum_to_one=True):
    mu_arr = mu.values
    cov_mat = cov.values
    N = len(mu_arr)

    if objective == 'sharpe':
        def obj(w):
            ret = w @ mu_arr
            vol = np.sqrt(w @ cov_mat @ w)
            ratio = -ret / vol if vol > 0 else np.inf
            penalty = ridge * np.sum(w ** 2)
            return ratio + penalty

    elif objective == 'kelly':
        def obj(w):
            utility = -(w @ mu_arr - 0.5 * w @ cov_mat @ w)
            penalty = ridge * np.sum(w ** 2)
            return utility + penalty

    else:
        raise ValueError("objective는 'sharpe' 또는 'kelly'만 가능합니다.")

    bounds = [(0, 1)] * N if sum_to_one else [(0, None)] * N

    # ✅ 비중합 = 1 제약 여부
    if sum_to_one:
        cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1},)
    else:
        cons = ()

    w0 = np.ones(N) / N

    res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons)
    return pd.Series(res.x if res.success else np.full(N, np.nan), index=mu.index)
















def get_annualized_cov_matrix(ticker_list, start="2023-01-01", end=None, span=20):
    """
    EWMA 기반 연율화 공분산 행렬 계산 함수
    - ticker_list: 티커 리스트 (예: ['SPY', 'QQQ', 'TLT'])
    - span: EWMA 가중치
    - start, end: 기간 설정
    """
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')

    price_df = pd.DataFrame()

    for ticker in ticker_list:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)[['Close']]
            price_df[ticker] = df['Close']
        except Exception as e:
            print(f"⚠️ {ticker} 다운로드 실패: {e}")

    returns = np.log(price_df / price_df.shift(1)).dropna()

    # EWMA 공분산 계산
    ewm_cov = returns.ewm(span=span).cov(pairwise=True)

    # 마지막 날짜의 공분산 행렬만 추출
    last_date = ewm_cov.index.get_level_values(0).max()
    cov_matrix = ewm_cov.loc[last_date].copy()

    # 연율화
    annualized_cov = cov_matrix * 252

    return annualized_cov.loc[ticker_list, ticker_list]





ktickers = ['091160.KS', '091180.KS']
us_etfs = ['381170.KS']
cn_etfs = ['371460.KS']


all_ticker = ktickers + us_etfs + cn_etfs


cov_matrix = get_annualized_cov_matrix(all_ticker, start="2023-01-01", span=20)

print("📊 연율화 공분산 행렬 (EWMA 기반):")
print(cov_matrix)
















import pandas_datareader.data as web
import datetime

# 오늘 날짜 지정 (혹시 데이터가 없을 경우 대비해서 하루 전까지 fallback 가능)
endus = datetime.datetime.today()
startus = endus - datetime.timedelta(days=7)  # 최근 일주일

# FRED에서 10년 국채 수익률 'DGS10' 불러오기
frd = web.DataReader('DGS10', 'fred', startus, endus)

# 결측 제거 + 가장 최근 값 추출
frd = frd.dropna()
usrf = frd.iloc[-1, 0] / 100  # 퍼센트 → 소수로 변환




text = """Implied ERP in previous month = 4.41% (Trailing 12 month, with adjusted payout); \
4.58% (Trailing 12 month cash yield); 5.86% (Average CF yield last 10 years); \
4.34% (Net cash yield); 4.03% (Normalized Earnings & Payout)"""

# 수치만 추출 후 소수로 변환
pattern = r"([\d.]+)%\s*\(Trailing 12 month cash yield\)"
match = re.search(pattern, text)

if match:
    userp = float(match.group(1)) / 100
    print(userp)
else:
    print("값을 찾을 수 없습니다.")




chinaerp = 5.27 / 100




from datetime import datetime, timedelta
import pandas_datareader.data as web

# 1. 티커 설정
exticker = us_etfs + cn_etfs

# 벤치마크 매핑
benchmark_map = {etf: '^GSPC' for etf in us_etfs}
benchmark_map.update({etf: '000300.SS' for etf in cn_etfs})

# 2. 기간 설정: 최근 1년
ex_end = datetime.today()
ex_start = ex_end - timedelta(days=365)

# 3. 가격 데이터 다운로드
all_tickers = exticker + list(set(benchmark_map.values()))
data = yf.download(all_tickers, start=ex_start, end=ex_end)['Close']

# 4. 로그수익률 계산 (일간)
log_returns = np.log(data).diff().dropna()

# 5. ERP 및 usrf 설정
userp = 4.58 / 100
chinaerp = 5.27 / 100

# 미국 무위험 수익률: FRED DGS10
try:
    rf_df = web.DataReader('DGS10', 'fred', ex_end - timedelta(days=7), ex_end)
    rf_df = rf_df.dropna()
    usrf = rf_df.iloc[-1, 0] / 100
except:
    usrf = 0.045  # fallback

# 6. ETF별 로그수익률 기반 베타 및 기대수익률 계산
exmu_result = []
for etf in exticker:
    mkt = benchmark_map[etf]
    etf_ret = log_returns[etf]
    mkt_ret = log_returns[mkt]

    aligned = pd.concat([etf_ret, mkt_ret], axis=1).dropna()
    x = aligned.iloc[:, 1]  # 시장
    y = aligned.iloc[:, 0]  # ETF

    cov = np.cov(y, x)[0, 1]
    var = np.var(x)
    beta = cov / var

    erp = userp if mkt == '^GSPC' else chinaerp
    expected_ret = beta * erp + (1 - beta) * usrf

    exmu_result.append({
        'ETF': etf,
        'Market': mkt,
        'Beta (1Y Daily Log)': round(beta, 4),
        'Expected Return (%)': round(expected_ret * 100, 2)
    })

# 7. 결과 정리
exmu_df = pd.DataFrame(exmu_result)
print(exmu_df)




# 1. 한국 ETF 기대수익률 (직접 수익률 - rf)
kr_mu = pd.Series(etf_returns)  # {'091160': val, ...}
kr_mu = kr_mu - usrf            # 동일한 무위험수익률 사용

# 2. 외국 ETF 기대수익률 (exmu_df의 값, 이미 rf 포함됨)
foreign_mu = exmu_df.set_index('ETF')['Expected Return (%)'] / 100  # 소수로

# 3. 티커명 통일
kr_mu.index = [f"{code}.KS" for code in kr_mu.index]

# 4. 통합
mu = pd.concat([kr_mu, foreign_mu])

weight_list = [1.0, 1.0, 1.0, 1.0]

# 리스트 곱하기 (순서 일치해야 함)
adjusted_mu = mu * weight_list


print("통합된 기대수익률 (mu):")
print(mu)


# 켈리 최적 포트폴리오 비중 계산
kelly_weights = optimize_weights(adjusted_mu, cov_matrix, objective='kelly', ridge= 0.1, sum_to_one= False)

print("켈리 기준 최적 투자 비중:")
print(kelly_weights)
