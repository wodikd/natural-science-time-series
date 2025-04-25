"""
SARIMA 모델 유틸리티: SARIMA 모델 학습, 잔차 생성 등
"""
import pandas as pd
import numpy as np
import os
import logging
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

logger = logging.getLogger(__name__)

def check_stationarity(ts):
    """
    시계열 데이터의 정상성 검정
    """
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

def run_auto_arima(ts, output_dir):
    """
    Auto ARIMA를 사용한 최적 SARIMA 모델 파라미터 탐색
    """
    logger.info("Auto ARIMA 실행 시작")
    try:
        auto_model = pm.auto_arima(
            ts, 
            seasonal=True, 
            m=12,
            start_p=0, max_p=2, 
            start_q=0, max_q=2,
            start_P=0, max_P=1, 
            start_Q=0, max_Q=1,
            d=None, D=None, 
            trace=True,
            error_action='ignore', 
            suppress_warnings=True,
            stepwise=True, 
            n_jobs=-1
        )
        
        logger.info(f"Auto ARIMA 최적 모델: SARIMA{auto_model.order}{auto_model.seasonal_order}")
        
        # 결과 저장
        with open(os.path.join(output_dir, "auto_arima_results.txt"), "w", encoding='utf-8') as f:
            f.write(str(auto_model.summary()))
            f.write("\n\n최적 모델 파라미터:\n")
            f.write(f"SARIMA{auto_model.order}{auto_model.seasonal_order}")
        
        return auto_model
    except Exception as e:
        logger.error(f"Auto ARIMA 실행 중 오류: {str(e)}")
        raise

def fit_sarima_model(ts, order, seasonal_order, output_dir):
    """
    SARIMA 모델 적합 및 잔차 계산
    """
    logger.info(f"SARIMA{order}{seasonal_order} 모델 적합 시작")
    try:
        model = SARIMAX(
            ts, 
            order=order, 
            seasonal_order=seasonal_order,
            enforce_stationarity=False, 
            enforce_invertibility=False
        )
        
        results = model.fit(disp=False)
        logger.info(f"SARIMA 모델 적합 완료, 로그 가능도: {results.llf:.4f}, AIC: {results.aic:.4f}")
        
        # 잔차 추출 및 저장
        residuals = results.resid
        residuals_df = pd.DataFrame({'residual': residuals}, index=ts.index)
        residuals_df.to_csv(os.path.join(output_dir, "sarima_residuals.csv"))
        logger.info(f"잔차 데이터가 {os.path.join(output_dir, 'sarima_residuals.csv')}에 저장됨")
        
        # 모델 요약 저장
        with open(os.path.join(output_dir, "sarima_model_summary.txt"), "w", encoding='utf-8') as f:
            f.write(str(results.summary()))
        
        return results, residuals_df
    except Exception as e:
        logger.error(f"SARIMA 모델 적합 중 오류: {str(e)}")
        raise

def sarima_forecast(model_results, steps, output_dir):
    """
    SARIMA 모델을 사용한 예측
    """
    logger.info(f"SARIMA 모델 {steps}개월 예측 시작")
    try:
        forecast = model_results.get_forecast(steps=steps)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # 예측 결과 저장
        forecast_df = pd.DataFrame({
            'forecast': forecast_mean,
            'lower_ci': forecast_ci.iloc[:, 0],
            'upper_ci': forecast_ci.iloc[:, 1]
        }, index=forecast_mean.index)
        
        forecast_df.to_csv(os.path.join(output_dir, "sarima_forecast.csv"))
        logger.info(f"SARIMA 예측 결과가 {os.path.join(output_dir, 'sarima_forecast.csv')}에 저장됨")
        
        return forecast_df
    except Exception as e:
        logger.error(f"SARIMA 예측 중 오류: {str(e)}")
        raise

def run_sarima_pipeline(df, target_col='sic', train_ratio=0.8, output_dir=None):
    """
    SARIMA 분석 전체 파이프라인
    """
    logger.info("SARIMA 분석 파이프라인 시작")
    
    try:
        # 타겟 변수 추출
        ts = df[target_col]
        
        # 학습/테스트 분할
        train_size = int(len(ts) * train_ratio)
        train_ts = ts[:train_size]
        test_ts = ts[train_size:]
        
        logger.info(f"학습 데이터: {train_ts.index[0]} ~ {train_ts.index[-1]} ({len(train_ts)} 개월)")
        logger.info(f"테스트 데이터: {test_ts.index[0]} ~ {test_ts.index[-1]} ({len(test_ts)} 개월)")
        
        # 정상성 검정
        check_stationarity(train_ts)
        
        # Auto ARIMA로 최적 파라미터 탐색
        auto_model = run_auto_arima(train_ts, output_dir)
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        
        # SARIMA 모델 적합 및 잔차 계산
        model_results, residuals_df = fit_sarima_model(train_ts, order, seasonal_order, output_dir)
        
        # 테스트 세트 예측
        test_forecast = sarima_forecast(model_results, len(test_ts), output_dir)
        
        # 평가
        mse = np.mean((test_ts - test_forecast['forecast'])**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_ts - test_forecast['forecast']))
        
        logger.info(f"SARIMA 모델 테스트 세트 평가 - RMSE: {rmse:.6f}, MAE: {mae:.6f}")
        
        # 평가 결과 저장
        with open(os.path.join(output_dir, "sarima_evaluation.txt"), "w", encoding='utf-8') as f:
            f.write(f"테스트 세트 RMSE: {rmse:.6f}\n")
            f.write(f"테스트 세트 MAE: {mae:.6f}\n")
        
        # 최종 모델 (전체 데이터로 학습)
        final_model_results, final_residuals_df = fit_sarima_model(ts, order, seasonal_order, output_dir)
        
        return final_model_results, final_residuals_df, order, seasonal_order
        
    except Exception as e:
        logger.error(f"SARIMA 분석 파이프라인 중 오류: {str(e)}")
        raise


def get_sarima_predictions(sarima_results, original_data, target_col='sic', output_dir=None):
    """
    SARIMA 모델의 학습 기간 내 fitted values와 테스트 기간 예측값을 모두 얻습니다.
    
    Parameters:
    -----------
    sarima_results : 학습된 SARIMA 모델 결과 객체
    original_data : 원본 데이터 (타겟 변수 포함)
    target_col : 타겟 변수명
    output_dir : 결과 저장 디렉토리
    
    Returns:
    --------
    sarima_predictions : 전체 SARIMA 예측값 (인샘플 fitted values + 아웃샘플 예측)
    """
    logger.info("SARIMA 모델 예측값 생성 시작")
    
    try:
        # 1. 인샘플 fitted values 획득
        in_sample_pred = sarima_results.fittedvalues
        
        # 2. 인덱스와 데이터프레임 구성
        sarima_predictions = pd.DataFrame({
            'sarima_pred': in_sample_pred,
            'actual': original_data.loc[in_sample_pred.index, target_col]
        })
        
        # 3. 인샘플 예측의 마지막 날짜 이후 데이터에 대한 예측
        if in_sample_pred.index[-1] < original_data.index[-1]:
            # 예측 기간 계산
            forecast_steps = len(original_data.index) - len(in_sample_pred.index)
            
            # 예측 수행
            out_of_sample_forecast = sarima_results.get_forecast(steps=forecast_steps)
            out_of_sample_pred = out_of_sample_forecast.predicted_mean
            out_of_sample_pred.index = original_data.index[len(in_sample_pred.index):]
            
            # 예측 결과와 실제 값 결합
            out_sample_df = pd.DataFrame({
                'sarima_pred': out_of_sample_pred,
                'actual': original_data.loc[out_of_sample_pred.index, target_col]
            })
            
            # 인샘플과 아웃샘플 결합
            sarima_predictions = pd.concat([sarima_predictions, out_sample_df])
        
        # 4. 예측 결과 저장
        if output_dir:
            sarima_predictions.to_csv(os.path.join(output_dir, 'sarima_predictions_all.csv'))
            logger.info(f"SARIMA 예측 결과 저장 완료: {os.path.join(output_dir, 'sarima_predictions_all.csv')}")
        
        # 5. 잔차 계산 및 추가
        sarima_predictions['residual'] = sarima_predictions['actual'] - sarima_predictions['sarima_pred']
        
        return sarima_predictions
        
    except Exception as e:
        logger.error(f"SARIMA 예측값 생성 중 오류: {str(e)}")
        raise