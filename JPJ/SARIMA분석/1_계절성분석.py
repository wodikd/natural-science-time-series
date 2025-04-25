import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
# 한글 폰트 설정
import matplotlib as mpl
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 폴더 생성
output_dir = "계절성분석결과"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 데이터 로드 및 전처리

def load_data(file_path):
    print("데이터 로드 중...")
    # 파일이 있는 경우
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # time 열을 datetime 형식으로 변환
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df[['sic']]  # 시간축과 SIC 값만 사용
    else:
        # 파일이 없는 경우 예시 데이터 생성 (테스트용)
        print(f"파일 '{file_path}'이 없어 예시 데이터를 생성합니다.")
        date_range = pd.date_range(start='1991-01-01', end='2023-12-01', freq='MS')
        # 예시 데이터 - 실제 값과 다를 수 있음
        np.random.seed(42)
        seasonal_pattern = np.tile(np.sin(np.linspace(0, 2*np.pi, 12)) * 0.2 + 0.7, 33)
        trend = np.linspace(0.9, 0.6, len(date_range))
        noise = np.random.normal(0, 0.05, len(date_range))
        sic_values = trend + seasonal_pattern[:len(date_range)] + noise
        # SIC는 0~1 사이 값이므로 범위 제한
        sic_values = np.clip(sic_values, 0, 1)
        
        df = pd.DataFrame({'sic': sic_values}, index=date_range)
        return df

# 정상성 검정
def check_stationarity(ts, output_dir):
    print("\n정상성 검정 중...")
    
    # ADF 테스트
    result = adfuller(ts.dropna())
    print(f'ADF 통계량: {result[0]:.4f}')
    print(f'p-값: {result[1]:.4f}')
    print(f'임계값: {result[4]}')
    
    if result[1] <= 0.05:
        print("시계열이 정상성을 가집니다.")
    else:
        print("시계열이 비정상성을 가집니다.")
    
    # 시계열 플롯
    plt.figure(figsize=(12, 6))
    plt.plot(ts)
    plt.title('보퍼트해 해빙 농도(SIC) 시계열')
    plt.xlabel('시간')
    plt.ylabel('SIC')
    plt.grid(True)
    plt.savefig(f"{output_dir}/1_time_series_plot.png")
    plt.close()
    
    # 롤링 통계량 계산
    rolling_mean = ts.rolling(window=12).mean()
    rolling_std = ts.rolling(window=12).std()
    
    # 롤링 통계량 플롯
    plt.figure(figsize=(12, 6))
    plt.plot(ts, label='원본 데이터')
    plt.plot(rolling_mean, label='12개월 이동 평균', color='red')
    plt.plot(rolling_std, label='12개월 이동 표준편차', color='green')
    plt.title('보퍼트해 해빙 농도(SIC) 이동 평균 및 표준편차')
    plt.xlabel('시간')
    plt.ylabel('SIC')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/2_rolling_statistics.png")
    plt.close()
    
    return result[1] <= 0.05  # 정상성 여부 반환

# 시계열 분해
def decompose_time_series(ts, output_dir):
    print("\n시계열 분해 중...")
    
    # 시계열 분해 (가법 모델)
    decomposition = seasonal_decompose(ts, model='additive', period=12)
    
    # 결과 플롯
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    decomposition.observed.plot(ax=axes[0])
    axes[0].set_title('관측 데이터')
    axes[0].grid(True)
    
    decomposition.trend.plot(ax=axes[1])
    axes[1].set_title('추세 요소')
    axes[1].grid(True)
    
    decomposition.seasonal.plot(ax=axes[2])
    axes[2].set_title('계절성 요소')
    axes[2].grid(True)
    
    decomposition.resid.plot(ax=axes[3])
    axes[3].set_title('잔차')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_time_series_decomposition.png")
    plt.close()
    
    # 월별 계절성 패턴 시각화
    seasonal_pattern = decomposition.seasonal.values[:12]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    plt.figure(figsize=(12, 6))
    plt.bar(months, seasonal_pattern)
    plt.title('월별 계절성 패턴')
    plt.ylabel('계절성 효과')
    plt.grid(True, axis='y')
    plt.savefig(f"{output_dir}/4_monthly_seasonality.png")
    plt.close()
    
    return decomposition

# ACF, PACF 플롯
def plot_acf_pacf(ts, output_dir):
    print("\nACF/PACF 플롯 생성 중...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # ACF 플롯
    plot_acf(ts.dropna(), lags=36, ax=axes[0])
    axes[0].set_title('자기상관함수(ACF)')
    
    # PACF 플롯
    plot_pacf(ts.dropna(), lags=36, ax=axes[1])
    axes[1].set_title('편자기상관함수(PACF)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/5_acf_pacf_plots.png")
    plt.close()

# 차분을 통한 정상화
def make_stationary(ts, output_dir):
    print("\n차분을 통한 정상화 중...")
    
    # 1차 차분
    ts_diff1 = ts.diff().dropna()
    
    # 1차 차분 후 정상성 검정
    result_diff1 = adfuller(ts_diff1.dropna())
    print(f'1차 차분 후 ADF 통계량: {result_diff1[0]:.4f}')
    print(f'1차 차분 후 p-값: {result_diff1[1]:.4f}')
    
    # 1차 차분 시계열 플롯
    plt.figure(figsize=(12, 6))
    plt.plot(ts_diff1)
    plt.title('보퍼트해 해빙 농도(SIC) 1차 차분')
    plt.xlabel('시간')
    plt.ylabel('SIC 1차 차분')
    plt.grid(True)
    plt.savefig(f"{output_dir}/6_first_difference.png")
    plt.close()
    
    # 계절 차분 (12개월)
    ts_diff_seasonal = ts.diff(12).dropna()
    
    # 계절 차분 후 정상성 검정
    result_diff_seasonal = adfuller(ts_diff_seasonal.dropna())
    print(f'계절 차분 후 ADF 통계량: {result_diff_seasonal[0]:.4f}')
    print(f'계절 차분 후 p-값: {result_diff_seasonal[1]:.4f}')
    
    # 계절 차분 시계열 플롯
    plt.figure(figsize=(12, 6))
    plt.plot(ts_diff_seasonal)
    plt.title('보퍼트해 해빙 농도(SIC) 계절 차분 (12개월)')
    plt.xlabel('시간')
    plt.ylabel('SIC 계절 차분')
    plt.grid(True)
    plt.savefig(f"{output_dir}/7_seasonal_difference.png")
    plt.close()
    
    # 1차 차분 + 계절 차분
    ts_diff_both = ts_diff1.diff(12).dropna()
    
    # 두 차분 모두 적용 후 정상성 검정
    result_diff_both = adfuller(ts_diff_both.dropna())
    print(f'1차 + 계절 차분 후 ADF 통계량: {result_diff_both[0]:.4f}')
    print(f'1차 + 계절 차분 후 p-값: {result_diff_both[1]:.4f}')
    
    # 두 차분 모두 적용 시계열 플롯
    plt.figure(figsize=(12, 6))
    plt.plot(ts_diff_both)
    plt.title('보퍼트해 해빙 농도(SIC) 1차 + 계절 차분')
    plt.xlabel('시간')
    plt.ylabel('SIC 1차 + 계절 차분')
    plt.grid(True)
    plt.savefig(f"{output_dir}/8_both_differences.png")
    plt.close()
    
    # 각 차분 방법 후 ACF/PACF 플롯
    # 1차 차분 ACF/PACF
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plot_acf(ts_diff1.dropna(), lags=36, ax=axes[0])
    axes[0].set_title('1차 차분 후 ACF')
    plot_pacf(ts_diff1.dropna(), lags=36, ax=axes[1])
    axes[1].set_title('1차 차분 후 PACF')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/9_acf_pacf_first_diff.png")
    plt.close()
    
    # 계절 차분 ACF/PACF
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plot_acf(ts_diff_seasonal.dropna(), lags=36, ax=axes[0])
    axes[0].set_title('계절 차분 후 ACF')
    plot_pacf(ts_diff_seasonal.dropna(), lags=36, ax=axes[1])
    axes[1].set_title('계절 차분 후 PACF')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/10_acf_pacf_seasonal_diff.png")
    plt.close()
    
    # 1차 + 계절 차분 ACF/PACF
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    plot_acf(ts_diff_both.dropna(), lags=36, ax=axes[0])
    axes[0].set_title('1차 + 계절 차분 후 ACF')
    plot_pacf(ts_diff_both.dropna(), lags=36, ax=axes[1])
    axes[1].set_title('1차 + 계절 차분 후 PACF')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/11_acf_pacf_both_diff.png")
    plt.close()
    
    return ts_diff1, ts_diff_seasonal, ts_diff_both

# 모델 적합 및 진단
def fit_sarima_model(ts, order, seasonal_order, output_dir):
    print(f"\nSARIMA({order[0]},{order[1]},{order[2]})({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]},{seasonal_order[3]}) 모델 적합 중...")
    
    model = SARIMAX(ts, 
                   order=order, 
                   seasonal_order=seasonal_order,
                   enforce_stationarity=False,
                   enforce_invertibility=False)
    
    results = model.fit(disp=False)
    print(results.summary())
    
    # 잔차 진단
    residuals = results.resid
    
    # 잔차 플롯
    plt.figure(figsize=(12, 6))
    plt.plot(residuals)
    plt.title('SARIMA 모델 잔차')
    plt.xlabel('시간')
    plt.ylabel('잔차')
    plt.grid(True)
    plt.savefig(f"{output_dir}/12_residuals.png")
    plt.close()
    
    # 잔차 히스토그램 및 밀도 플롯
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, kde=True)
    plt.title('SARIMA 모델 잔차 분포')
    plt.xlabel('잔차')
    plt.grid(True)
    plt.savefig(f"{output_dir}/13_residual_distribution.png")
    plt.close()
    
    # 잔차 QQ 플롯
    plt.figure(figsize=(12, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('SARIMA 모델 잔차 QQ 플롯')
    plt.grid(True)
    plt.savefig(f"{output_dir}/14_residual_qq_plot.png")
    plt.close()
    
    # 잔차 ACF
    plt.figure(figsize=(12, 6))
    plot_acf(residuals.dropna(), lags=36)
    plt.title('SARIMA 모델 잔차 ACF')
    plt.grid(True)
    plt.savefig(f"{output_dir}/15_residual_acf.png")
    plt.close()
    
    # 융-박스 검정
    lb_test = acorr_ljungbox(residuals, lags=[12, 24, 36], return_df=True)
    print("\n융-박스 검정 결과:")
    print(lb_test)
    
    return results

# Auto ARIMA
def run_auto_arima(ts, output_dir):
    print("\nAuto ARIMA 실행 중...")
    
    # pmdarima의 auto_arima 함수를 사용하여 최적의 모델 파라미터 탐색
    auto_model = pm.auto_arima(ts,
                             seasonal=True,
                             m=12,  # 12개월 계절성
                             start_p=0, max_p=3,
                             start_q=0, max_q=3,
                             start_P=0, max_P=2,
                             start_Q=0, max_Q=2,
                             d=None, D=None,  # 차분 차수 자동 결정
                             trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)
    
    print("\nAuto ARIMA 최적 모델:")
    print(auto_model.summary())
    
    # 결과 저장
    with open(f"{output_dir}/auto_arima_results.txt", "w") as f:
        f.write(str(auto_model.summary()))
        f.write("\n\n최적 모델 파라미터:\n")
        f.write(f"SARIMA({auto_model.order[0]},{auto_model.order[1]},{auto_model.order[2]})")
        f.write(f"({auto_model.seasonal_order[0]},{auto_model.seasonal_order[1]},{auto_model.seasonal_order[2]},{auto_model.seasonal_order[3]})")
    
    return auto_model

# 예측 실행 및 시각화
def forecast_and_plot(model, ts, steps, output_dir):
    print(f"\n향후 {steps}개월 예측 중...")
    
    # 예측 실행
    forecast_result = model.get_forecast(steps=steps)
    forecast_index = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')
    
    # 예측 결과 및 신뢰 구간
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int()
    
    # 예측 결과 데이터프레임 생성
    forecast_df = pd.DataFrame({
        'forecast': forecast_mean,
        'lower_ci': forecast_ci.iloc[:, 0],
        'upper_ci': forecast_ci.iloc[:, 1]
    }, index=forecast_index)
    
    # 예측 결과 시각화
    plt.figure(figsize=(12, 6))
    
    # 원본 데이터
    plt.plot(ts.index, ts, label='관측 데이터', color='blue')
    
    # 예측 결과
    plt.plot(forecast_df.index, forecast_df['forecast'], label='예측', color='red', linestyle='--')
    
    # 신뢰 구간
    plt.fill_between(forecast_df.index, 
                    forecast_df['lower_ci'], 
                    forecast_df['upper_ci'], 
                    color='pink', alpha=0.3, label='95% 신뢰 구간')
    
    plt.title(f'보퍼트해 해빙 농도(SIC) {steps}개월 예측')
    plt.xlabel('시간')
    plt.ylabel('SIC')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/16_forecast_{steps}_months.png")
    plt.close()
    
    # 예측 결과 CSV로 저장
    forecast_df.to_csv(f"{output_dir}/forecast_results_{steps}_months.csv")
    
    print("예측 결과:")
    print(forecast_df)
    
    return forecast_df

# 모델 평가 (학습/테스트 분할)
def evaluate_model(ts, model_order, seasonal_order, output_dir):
    print("\n모델 평가 중...")
    
    # 데이터 분할 (마지막 12개월을 테스트 세트로 사용)
    train_size = len(ts) - 12
    train_data = ts.iloc[:train_size]
    test_data = ts.iloc[train_size:]
    
    print(f"학습 데이터: {train_data.index[0]} ~ {train_data.index[-1]} ({len(train_data)} 개월)")
    print(f"테스트 데이터: {test_data.index[0]} ~ {test_data.index[-1]} ({len(test_data)} 개월)")
    
    # 학습 데이터로 모델 학습
    model = SARIMAX(train_data,
                   order=model_order,
                   seasonal_order=seasonal_order,
                   enforce_stationarity=False,
                   enforce_invertibility=False)
    
    model_fit = model.fit(disp=False)
    
    # 테스트 기간 예측
    forecast = model_fit.get_forecast(steps=len(test_data))
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    # 예측 결과 데이터프레임
    forecast_df = pd.DataFrame({
        'forecast': forecast_mean,
        'lower_ci': forecast_ci.iloc[:, 0],
        'upper_ci': forecast_ci.iloc[:, 1],
        'actual': test_data.values.flatten()
    }, index=test_data.index)
    
    # 평가 지표 계산
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(test_data, forecast_mean))
    mae = mean_absolute_error(test_data, forecast_mean)
    r2 = r2_score(test_data, forecast_mean)
    
    print(f"\n평가 지표:")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    
    # 평가 결과 시각화
    plt.figure(figsize=(12, 6))
    
    # 학습 데이터
    plt.plot(train_data.index, train_data, label='학습 데이터', color='blue')
    
    # 테스트 데이터
    plt.plot(test_data.index, test_data, label='실제 데이터', color='green')
    
    # 예측 결과
    plt.plot(forecast_df.index, forecast_df['forecast'], label='예측', color='red', linestyle='--')
    
    # 신뢰 구간
    plt.fill_between(forecast_df.index, 
                    forecast_df['lower_ci'], 
                    forecast_df['upper_ci'], 
                    color='pink', alpha=0.3, label='95% 신뢰 구간')
    
    plt.title('SARIMA 모델 평가 (학습/테스트 분할)')
    plt.xlabel('시간')
    plt.ylabel('SIC')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/17_model_evaluation.png")
    plt.close()
    
    # 잔차 분석
    residuals = forecast_df['actual'] - forecast_df['forecast']
    
    plt.figure(figsize=(12, 6))
    plt.plot(residuals.index, residuals)
    plt.title('예측 잔차 (실제 - 예측)')
    plt.xlabel('시간')
    plt.ylabel('잔차')
    plt.grid(True)
    plt.savefig(f"{output_dir}/18_forecast_residuals.png")
    plt.close()
    
    # 결과 저장
    evaluation_results = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    with open(f"{output_dir}/model_evaluation_results.txt", "w") as f:
        f.write("모델 평가 결과:\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"R²: {r2:.6f}\n")
    
    return evaluation_results, forecast_df

def main():
    print("보퍼트해 해빙 농도(SIC) SARIMA 분석 시작")
    
    # 데이터 파일 경로
    file_path = r"D:\my_projects\natural-science-time-series\JPJ\데이터\v3_이상치처리.csv"
    
    # 데이터 로드
    df = load_data(file_path)
    
    # 시계열 데이터
    ts = df['sic']
    
    print(f"\n데이터 정보:")
    print(f"시작 날짜: {ts.index[0]}")
    print(f"종료 날짜: {ts.index[-1]}")
    print(f"데이터 길이: {len(ts)} 개월")
    print(f"평균 SIC: {ts.mean():.4f}")
    print(f"최소 SIC: {ts.min():.4f}")
    print(f"최대 SIC: {ts.max():.4f}")
    
    # 정상성 검정
    is_stationary = check_stationarity(ts, output_dir)
    
    # 시계열 분해
    decomposition = decompose_time_series(ts, output_dir)
    
    # ACF, PACF 플롯
    plot_acf_pacf(ts, output_dir)
    
    # 차분을 통한 정상화
    ts_diff1, ts_diff_seasonal, ts_diff_both = make_stationary(ts, output_dir)
    
    # Auto ARIMA 실행
    auto_model = run_auto_arima(ts, output_dir)
    
    # 최적 모델의 파라미터 가져오기
    best_order = auto_model.order
    best_seasonal_order = auto_model.seasonal_order
    
    print(f"\n최적 모델: SARIMA{best_order}{best_seasonal_order}")
    
    # 전체 데이터로 최종 모델 학습
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import scipy.stats as stats
    
    final_model = SARIMAX(ts,
                         order=best_order,
                         seasonal_order=best_seasonal_order,
                         enforce_stationarity=False,
                         enforce_invertibility=False)
    
    final_results = final_model.fit(disp=False)
    
    # 모델 평가 (학습/테스트 분할)
    eval_results, forecast_df = evaluate_model(ts, best_order, best_seasonal_order, output_dir)
    
    # 향후 12개월 예측
    forecast_12m = forecast_and_plot(final_results, ts, 12, output_dir)
    
    # 향후 24개월 예측
    forecast_24m = forecast_and_plot(final_results, ts, 24, output_dir)
    
    print("\n분석 완료! 결과는 '계절성분석결과' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()