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

# 데이터 파일 경로
file_path = r"D:\my_projects\natural-science-time-series\JPJ\데이터\v3_이상치처리.csv"

# 폴더 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"계절성분석결과_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# 데이터 로드 및 전처리
def load_data(file_path):
    print("데이터 로드 중...")
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"데이터 파일 '{file_path}'을 찾을 수 없습니다.")
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # 결측값 처리
        if df['sic'].isna().any():
            print("결측값 발견, 선형 보간법으로 처리합니다.")
            df['sic'] = df['sic'].interpolate(method='linear')
        
        # 값 범위 검증
        if (df['sic'] < 0).any() or (df['sic'] > 1).any():
            print("SIC 값이 0~1 범위를 벗어납니다. 값을 클리핑합니다.")
            df['sic'] = df['sic'].clip(0, 1)
        
        # 시간 간격 검증 (월별 데이터 확인)
        freq = pd.infer_freq(df.index)
        if freq != 'MS':
            print(f"경고: 데이터의 시간 간격이 월별(MS)이 아닙니다. 현재 간격: {freq}")
        
        return df[['sic']]
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {str(e)}")
        raise

# 정상성 검정
def check_stationarity(ts, output_dir):
    print("\n정상성 검정 중...")
    try:
        result = adfuller(ts.dropna())
        print(f'ADF 통계량: {result[0]:.4f}')
        print(f'p-값: {result[1]:.4f}')
        print(f'임계값: {result[4]}')
        
        is_stationary = result[1] <= 0.05
        print("시계열이 정상성을 가집니다." if is_stationary else "시계열이 비정상성을 가집니다.")
        
        # 시계열 플롯
        plt.figure(figsize=(12, 6))
        plt.plot(ts)
        plt.title('보퍼트해 해빙 농도(SIC) 시계열')
        plt.xlabel('시간')
        plt.ylabel('SIC')
        plt.grid(True)
        plt.savefig(f"{output_dir}/1_time_series_plot.png")
        plt.close()
        
        # 롤링 통계량
        rolling_mean = ts.rolling(window=12).mean()
        rolling_std = ts.rolling(window=12).std()
        
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
        
        return is_stationary
    except Exception as e:
        print(f"정상성 검정 중 오류 발생: {str(e)}")
        raise

# 시계열 분해
def decompose_time_series(ts, output_dir):
    print("\n시계열 분해 중...")
    try:
        decomposition = seasonal_decompose(ts, model='additive', period=12)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        decomposition.observed.plot(ax=axes[0], title='관측 데이터')
        decomposition.trend.plot(ax=axes[1], title='추세 요소')
        decomposition.seasonal.plot(ax=axes[2], title='계절성 요소')
        decomposition.resid.plot(ax=axes[3], title='잔차')
        for ax in axes:
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/3_time_series_decomposition.png")
        plt.close()
        
        # 월별 계절성 패턴
        seasonal_pattern = decomposition.seasonal.values[:12]
        months = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']
        
        plt.figure(figsize=(12, 6))
        plt.bar(months, seasonal_pattern)
        plt.title('월별 계절성 패턴')
        plt.ylabel('계절성 효과')
        plt.grid(True, axis='y')
        plt.savefig(f"{output_dir}/4_monthly_seasonality.png")
        plt.close()
        
        return decomposition
    except Exception as e:
        print(f"시계열 분해 중 오류 발생: {str(e)}")
        raise

# ACF, PACF 플롯
def plot_acf_pacf(ts, output_dir):
    print("\nACF/PACF 플롯 생성 중...")
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        plot_acf(ts.dropna(), lags=36, ax=axes[0])
        axes[0].set_title('자기상관함수(ACF)')
        plot_pacf(ts.dropna(), lags=36, ax=axes[1])
        axes[1].set_title('편자기상관함수(PACF)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/5_acf_pacf_plots.png")
        plt.close()
    except Exception as e:
        print(f"ACF/PACF 플롯 생성 중 오류 발생: {str(e)}")
        raise

# 차분을 통한 정상화
def make_stationary(ts, output_dir):
    print("\n차분을 통한 정상화 중...")
    try:
        ts_diff1 = ts.diff().dropna()
        result_diff1 = adfuller(ts_diff1.dropna())
        print(f'1차 차분 후 ADF 통계량: {result_diff1[0]:.4f}')
        print(f'1차 차분 후 p-값: {result_diff1[1]:.4f}')
        
        plt.figure(figsize=(12, 6))
        plt.plot(ts_diff1)
        plt.title('보퍼트해 해빙 농도(SIC) 1차 차분')
        plt.xlabel('시간')
        plt.ylabel('SIC 1차 차분')
        plt.grid(True)
        plt.savefig(f"{output_dir}/6_first_difference.png")
        plt.close()
        
        ts_diff_seasonal = ts.diff(12).dropna()
        result_diff_seasonal = adfuller(ts_diff_seasonal.dropna())
        print(f'계절 차분 후 ADF 통계량: {result_diff_seasonal[0]:.4f}')
        print(f'계절 차분 후 p-값: {result_diff_seasonal[1]:.4f}')
        
        plt.figure(figsize=(12, 6))
        plt.plot(ts_diff_seasonal)
        plt.title('보퍼트해 해빙 농도(SIC) 계절 차분 (12개월)')
        plt.xlabel('시간')
        plt.ylabel('SIC 계절 차분')
        plt.grid(True)
        plt.savefig(f"{output_dir}/7_seasonal_difference.png")
        plt.close()
        
        ts_diff_both = ts_diff1.diff(12).dropna()
        result_diff_both = adfuller(ts_diff_both.dropna())
        print(f'1차 + 계절 차분 후 ADF 통계량: {result_diff_both[0]:.4f}')
        print(f'1차 + 계절 차분 후 p-값: {result_diff_both[1]:.4f}')
        
        plt.figure(figsize=(12, 6))
        plt.plot(ts_diff_both)
        plt.title('보퍼트해 해빙 농도(SIC) 1차 + 계절 차분')
        plt.xlabel('시간')
        plt.ylabel('SIC 1차 + 계절 차분')
        plt.grid(True)
        plt.savefig(f"{output_dir}/8_both_differences.png")
        plt.close()
        
        # ACF/PACF 플롯
        for diff_ts, name, idx in [
            (ts_diff1, '1차 차분', 9),
            (ts_diff_seasonal, '계절 차분', 10),
            (ts_diff_both, '1차 + 계절 차분', 11)
        ]:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            plot_acf(diff_ts.dropna(), lags=36, ax=axes[0])
            axes[0].set_title(f'{name} 후 ACF')
            plot_pacf(diff_ts.dropna(), lags=36, ax=axes[1])
            axes[1].set_title(f'{name} 후 PACF')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{idx}_acf_pacf_{name.replace(' ', '_')}.png")
            plt.close()
        
        return ts_diff1, ts_diff_seasonal, ts_diff_both
    except Exception as e:
        print(f"차분 중 오류 발생: {str(e)}")
        raise

# SARIMA 모델 적합
def fit_sarima_model(ts, order, seasonal_order, output_dir):
    print(f"\nSARIMA({order[0]},{order[1]},{order[2]})({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]},{seasonal_order[3]}) 모델 적합 중...")
    try:
        model = SARIMAX(ts, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        print(results.summary())
        
        residuals = results.resid
        plt.figure(figsize=(12, 6))
        plt.plot(residuals)
        plt.title('SARIMA 모델 잔차')
        plt.xlabel('시간')
        plt.ylabel('잔차')
        plt.grid(True)
        plt.savefig(f"{output_dir}/12_residuals.png")
        plt.close()
        
        plt.figure(figsize=(12, 6))
        sns.histplot(residuals, kde=True)
        plt.title('SARIMA 모델 잔차 분포')
        plt.xlabel('잔차')
        plt.grid(True)
        plt.savefig(f"{output_dir}/13_residual_distribution.png")
        plt.close()
        
        plt.figure(figsize=(12, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('SARIMA 모델 잔차 QQ 플롯')
        plt.grid(True)
        plt.savefig(f"{output_dir}/14_residual_qq_plot.png")
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plot_acf(residuals.dropna(), lags=36)
        plt.title('SARIMA 모델 잔차 ACF')
        plt.grid(True)
        plt.savefig(f"{output_dir}/15_residual_acf.png")
        plt.close()
        
        lb_test = acorr_ljungbox(residuals, lags=[12, 24, 36], return_df=True)
        print("\n융-박스 검정 결과:")
        print(lb_test)
        
        return results
    except Exception as e:
        print(f"SARIMA 모델 적합 중 오류 발생: {str(e)}")
        raise

# Auto ARIMA
def run_auto_arima(ts, output_dir):
    print("\nAuto ARIMA 실행 중...")
    try:
        auto_model = pm.auto_arima(ts, seasonal=True, m=12,
                                  start_p=0, max_p=2, start_q=0, max_q=2,
                                  start_P=0, max_P=1, start_Q=0, max_Q=1,
                                  d=None, D=None, trace=True,
                                  error_action='ignore', suppress_warnings=True,
                                  stepwise=True, n_jobs=-1)
        print("\nAuto ARIMA 최적 모델:")
        print(auto_model.summary())
        
        with open(f"{output_dir}/auto_arima_results.txt", "w") as f:
            f.write(str(auto_model.summary()))
            f.write("\n\n최적 모델 파라미터:\n")
            f.write(f"SARIMA({auto_model.order[0]},{auto_model.order[1]},{auto_model.order[2]})")
            f.write(f"({auto_model.seasonal_order[0]},{auto_model.seasonal_order[1]},{auto_model.seasonal_order[2]},{auto_model.seasonal_order[3]})")
        
        return auto_model
    except Exception as e:
        print(f"Auto ARIMA 실행 중 오류 발생: {str(e)}")
        raise

# 예측 및 시각화
def forecast_and_plot(model, ts, steps, output_dir):
    print(f"\n향후 {steps}개월 예측 중...")
    try:
        forecast_result = model.get_forecast(steps=steps)
        forecast_index = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        
        forecast_df = pd.DataFrame({
            'forecast': forecast_mean,
            'lower_ci': forecast_ci.iloc[:, 0],
            'upper_ci': forecast_ci.iloc[:, 1]
        }, index=forecast_index)
        
        plt.figure(figsize=(12, 6))
        plt.plot(ts.index, ts, label='관측 데이터', color='blue')
        plt.plot(forecast_df.index, forecast_df['forecast'], label='예측', color='red', linestyle='--')
        plt.fill_between(forecast_df.index, forecast_df['lower_ci'], forecast_df['upper_ci'],
                        color='pink', alpha=0.3, label='95% 신뢰 구간')
        plt.title(f'보퍼트해 해빙 농도(SIC) {steps}개월 예측')
        plt.xlabel('시간')
        plt.ylabel('SIC')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/16_forecast_{steps}_months.png")
        plt.close()
        
        forecast_df.to_csv(f"{output_dir}/forecast_results_{steps}_months.csv")
        print("예측 결과:")
        print(forecast_df)
        
        return forecast_df
    except Exception as e:
        print(f"예측 중 오류 발생: {str(e)}")
        raise

# 모델 평가
def evaluate_model(ts, model_order, seasonal_order, output_dir):
    print("\n모델 평가 중...")
    try:
        train_size = len(ts) - 12
        train_data = ts.iloc[:train_size]
        test_data = ts.iloc[train_size:]
        
        print(f"학습 데이터: {train_data.index[0]} ~ {train_data.index[-1]} ({len(train_data)} 개월)")
        print(f"테스트 데이터: {test_data.index[0]} ~ {test_data.index[-1]} ({len(test_data)} 개월)")
        
        model = SARIMAX(train_data, order=model_order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        
        forecast = model_fit.get_forecast(steps=len(test_data))
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        forecast_df = pd.DataFrame({
            'forecast': forecast_mean,
            'lower_ci': forecast_ci.iloc[:, 0],
            'upper_ci': forecast_ci.iloc[:, 1],
            'actual': test_data.values.flatten()
        }, index=test_data.index)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(test_data, forecast_mean))
        mae = mean_absolute_error(test_data, forecast_mean)
        r2 = r2_score(test_data, forecast_mean)
        epsilon = 1e-10
        mape = np.mean(np.abs((test_data.values.flatten() - forecast_mean) / (test_data.values.flatten() + epsilon))) * 100
        
        print(f"\n평가 지표:")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"MAPE: {mape:.2f}%")
        
        plt.figure(figsize=(12, 6))
        plt.plot(train_data.index, train_data, label='학습 데이터', color='blue')
        plt.plot(test_data.index, test_data, label='실제 데이터', color='green')
        plt.plot(forecast_df.index, forecast_df['forecast'], label='예측', color='red', linestyle='--')
        plt.fill_between(forecast_df.index, forecast_df['lower_ci'], forecast_df['upper_ci'],
                        color='pink', alpha=0.3, label='95% 신뢰 구간')
        plt.title('SARIMA 모델 평가 (학습/테스트 분할)')
        plt.xlabel('시간')
        plt.ylabel('SIC')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/17_model_evaluation.png")
        plt.close()
        
        residuals = forecast_df['actual'] - forecast_df['forecast']
        plt.figure(figsize=(12, 6))
        plt.plot(residuals.index, residuals)
        plt.title('예측 잔차 (실제 - 예측)')
        plt.xlabel('시간')
        plt.ylabel('잔차')
        plt.grid(True)
        plt.savefig(f"{output_dir}/18_forecast_residuals.png")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(test_data.values, forecast_mean, alpha=0.7)
        min_val = min(min(test_data), min(forecast_mean))
        max_val = max(max(test_data), max(forecast_mean))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('실제 SIC 값')
        plt.ylabel('예측 SIC 값')
        plt.title('실제 값 vs 예측 값 산점도')
        plt.grid(True)
        plt.savefig(f"{output_dir}/19_actual_vs_predicted_scatter.png")
        plt.close()
        
        evaluation_results = {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape}
        with open(f"{output_dir}/model_evaluation_results.txt", "w") as f:
            f.write("모델 평가 결과:\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"R²: {r2:.6f}\n")
            f.write(f"MAPE: {mape:.2f}%\n")
        
        return evaluation_results, forecast_df
    except Exception as e:
        print(f"모델 평가 중 오류 발생: {str(e)}")
        raise

# 메인 함수
def main():
    print("분석 시작...")
    try:
        ts = load_data(file_path)
        is_stationary = check_stationarity(ts['sic'], output_dir)
        decomposition = decompose_time_series(ts['sic'], output_dir)
        plot_acf_pacf(ts['sic'], output_dir)
        ts_diff1, ts_diff_seasonal, ts_diff_both = make_stationary(ts['sic'], output_dir)
        auto_model = run_auto_arima(ts['sic'], output_dir)
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        sarima_results = fit_sarima_model(ts['sic'], order, seasonal_order, output_dir)
        evaluation_results, forecast_df = evaluate_model(ts['sic'], order, seasonal_order, output_dir)
        forecast_12m = forecast_and_plot(sarima_results, ts['sic'], 12, output_dir)
        forecast_24m = forecast_and_plot(sarima_results, ts['sic'], 24, output_dir)
        
        print("\n분석 완료! 결과는 '{}' 폴더에 저장되었습니다.".format(output_dir))
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()