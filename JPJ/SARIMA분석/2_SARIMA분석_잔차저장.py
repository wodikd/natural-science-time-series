import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import os
from datetime import datetime
import logging
import warnings
warnings.filterwarnings("ignore")

# 한글 폰트 설정 (로그 메시지에 한글 사용 가능성 대비)
import matplotlib as mpl
mpl.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False

# 로그 설정
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"계절성분석_잔차저장_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

logging.basicConfig(
    filename=f"{output_dir}/analysis_log.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger()

# 데이터 파일 경로
file_path = r"D:\my_projects\natural-science-time-series\JPJ\데이터\v3_이상치처리.csv"

# 데이터 로드 및 전처리
def load_data(file_path):
    logger.info("데이터 로드 시작")
    try:
        if not os.path.exists(file_path):
            logger.error(f"데이터 파일 '{file_path}'을 찾을 수 없습니다.")
            raise FileNotFoundError(f"데이터 파일 '{file_path}'을 찾을 수 없습니다.")
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        if df['sic'].isna().any():
            logger.warning("결측값 발견, 선형 보간법으로 처리")
            df['sic'] = df['sic'].interpolate(method='linear')
        
        if (df['sic'] < 0).any() or (df['sic'] > 1).any():
            logger.warning("SIC 값이 0~1 범위를 벗어남, 클리핑 수행")
            df['sic'] = df['sic'].clip(0, 1)
        
        freq = pd.infer_freq(df.index)
        if freq != 'MS':
            logger.warning(f"데이터의 시간 간격이 월별(MS)이 아님, 현재 간격: {freq}")
        
        logger.info("데이터 로드 완료")
        return df[['sic']]
    except Exception as e:
        logger.error(f"데이터 로드 중 오류: {str(e)}")
        raise

# 정상성 검정
def check_stationarity(ts):
    logger.info("정상성 검정 시작")
    try:
        result = adfuller(ts.dropna())
        logger.info(f"ADF 통계량: {result[0]:.4f}, p-값: {result[1]:.4f}, 임계값: {result[4]}")
        is_stationary = result[1] <= 0.05
        logger.info("시계열이 정상성을 가짐" if is_stationary else "시계열이 비정상성을 가짐")
        return is_stationary
    except Exception as e:
        logger.error(f"정상성 검정 중 오류: {str(e)}")
        raise

# Auto ARIMA
def run_auto_arima(ts, output_dir):
    logger.info("Auto ARIMA 실행 시작")
    try:
        auto_model = pm.auto_arima(ts, seasonal=True, m=12,
                                  start_p=0, max_p=2, start_q=0, max_q=2,
                                  start_P=0, max_P=1, start_Q=0, max_Q=1,
                                  d=None, D=None, trace=True,
                                  error_action='ignore', suppress_warnings=True,
                                  stepwise=True, n_jobs=-1)
        logger.info(f"Auto ARIMA 최적 모델: SARIMA{auto_model.order}{auto_model.seasonal_order}")
        
        with open(f"{output_dir}/auto_arima_results.txt", "w", encoding='utf-8') as f:
            f.write(str(auto_model.summary()))
            f.write("\n\n최적 모델 파라미터:\n")
            f.write(f"SARIMA{auto_model.order}{auto_model.seasonal_order}")
        
        return auto_model
    except Exception as e:
        logger.error(f"Auto ARIMA 실행 중 오류: {str(e)}")
        raise

# SARIMA 모델 적합 및 잔차 계산
def fit_sarima_model(ts, order, seasonal_order, output_dir):
    logger.info(f"SARIMA{order}{seasonal_order} 모델 적합 시작")
    try:
        model = SARIMAX(ts, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        logger.info(f"SARIMA 모델 적합 완료, 로그 가능도: {results.llf:.4f}, AIC: {results.aic:.4f}")
        
        # 잔차 추출 및 저장
        residuals = results.resid
        residuals_df = pd.DataFrame({'residual': residuals}, index=ts.index)
        residuals_df.to_csv(f"{output_dir}/sarima_residuals.csv")
        logger.info(f"잔차 데이터가 {output_dir}/sarima_residuals.csv에 저장됨")
        
        return results
    except Exception as e:
        logger.error(f"SARIMA 모델 적합 중 오류: {str(e)}")
        raise

# 메인 함수
def main():
    logger.info("분석 시작")
    try:
        # 데이터 로드
        ts = load_data(file_path)
        
        # 정상성 검정
        check_stationarity(ts['sic'])
        
        # Auto ARIMA로 최적 파라미터 탐색
        auto_model = run_auto_arima(ts['sic'], output_dir)
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        
        # SARIMA 모델 적합 및 잔차 저장
        fit_sarima_model(ts['sic'], order, seasonal_order, output_dir)
        
        logger.info(f"분석 완료, 결과는 '{output_dir}' 폴더에 저장됨")
    except Exception as e:
        logger.error(f"분석 중 오류: {str(e)}")
        raise

if __name__ == "__main__":
    main()