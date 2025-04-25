"""
데이터 처리 유틸리티: 데이터 로드, 전처리, 특성 엔지니어링 등
"""
import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler
from config import FEATURES, DEFAULT_LAG_FEATURES

logger = logging.getLogger(__name__)

def load_original_data(file_path):
    """
    원본 데이터 로드 및 기본 전처리
    """
    logger.info(f"원본 데이터 파일 '{file_path}' 로드 시작")
    try:
        if not os.path.exists(file_path):
            logger.error(f"데이터 파일 '{file_path}'을 찾을 수 없습니다.")
            raise FileNotFoundError(f"데이터 파일 '{file_path}'을 찾을 수 없습니다.")
            
        df = pd.read_csv(file_path)
        
        # 시간 인덱스 설정
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # 데이터 검증
        missing_features = [feature for feature in FEATURES if feature not in df.columns]
        if missing_features:
            logger.warning(f"설정에 정의된 일부 특성이 데이터에 없습니다: {missing_features}")
        
        available_features = [feature for feature in FEATURES if feature in df.columns]
        logger.info(f"사용 가능한 특성: {available_features}")
        
        # 결측값 확인 및 처리
        if df[available_features].isna().any().any():
            logger.warning("결측값 발견, 선형 보간법으로 처리")
            df[available_features] = df[available_features].interpolate(method='linear')
        
        # 데이터 기간 정보
        logger.info(f"데이터 기간: {df.index.min()} ~ {df.index.max()} ({len(df)} 개월)")
        
        return df
    except Exception as e:
        logger.error(f"데이터 로드 중 오류: {str(e)}")
        raise

def load_residuals(residual_path):
    """
    SARIMA 모델 잔차 파일 로드
    """
    logger.info(f"잔차 파일 '{residual_path}' 로드 시작")
    try:
        if not os.path.exists(residual_path):
            logger.error(f"잔차 파일 '{residual_path}'을 찾을 수 없습니다.")
            raise FileNotFoundError(f"잔차 파일 '{residual_path}'을 찾을 수 없습니다.")
            
        residuals_df = pd.read_csv(residual_path)
        
        # 시간 인덱스 설정
        residuals_df['time'] = pd.to_datetime(residuals_df['time'])
        residuals_df.set_index('time', inplace=True)
        
        logger.info(f"잔차 데이터 기간: {residuals_df.index.min()} ~ {residuals_df.index.max()} ({len(residuals_df)} 개월)")
        
        return residuals_df
    except Exception as e:
        logger.error(f"잔차 데이터 로드 중 오류: {str(e)}")
        raise

def merge_data_with_residuals(data_df, residuals_df):
    """
    원본 데이터와 잔차 데이터 병합
    """
    logger.info("원본 데이터와 잔차 데이터 병합 시작")
    try:
        # sic 컬럼 제거 (타겟 변수이므로 트리 모델 특성에서 제외)
        if 'sic' in data_df.columns:
            data_df = data_df.drop(columns=['sic'])
            logger.info("'sic' 컬럼을 특성에서 제외")
        
        # 데이터 병합
        merged_df = pd.merge(residuals_df, data_df, left_index=True, right_index=True, how='inner')
        
        # 병합 결과 확인
        logger.info(f"병합 후 데이터 크기: {merged_df.shape}")
        
        return merged_df
    except Exception as e:
        logger.error(f"데이터 병합 중 오류: {str(e)}")
        raise

def add_time_features(df):
    """
    시간 관련 특성 추가 (월 사이클릭 인코딩)
    """
    logger.info("시간 관련 특성 추가")
    try:
        # 월 사이클릭 인코딩
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # 연도 정보 추가 (선형 추세 포착 가능)
        df['year'] = df.index.year
        
        return df
    except Exception as e:
        logger.error(f"시간 특성 추가 중 오류: {str(e)}")
        raise

def add_lag_features(df, target_col='residual', lags=DEFAULT_LAG_FEATURES):
    """
    시차 특성 추가 (t-1, t-2, ...)
    """
    logger.info(f"시차 특성 추가 (lags={lags})")
    try:
        for i in range(1, lags + 1):
            df[f'{target_col}_lag_{i}'] = df[target_col].shift(i)
        
        # NaN 값을 가진 행 제거
        orig_len = len(df)
        df = df.dropna()
        logger.info(f"시차 특성 추가 후 NaN 제거: {orig_len} -> {len(df)} 행")
        
        return df
    except Exception as e:
        logger.error(f"시차 특성 추가 중 오류: {str(e)}")
        raise

def split_data(df, target_col='residual', train_ratio=0.7, val_ratio=0.15, forecast_horizon=1):
    """
    데이터를 훈련, 검증, 테스트 세트로 분할
    forecast_horizon: 예측 기간 (개월)
    """
    logger.info(f"데이터 분할 시작 (train_ratio={train_ratio}, val_ratio={val_ratio}, forecast_horizon={forecast_horizon})")
    try:
        # 특성과 타겟 분리
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 시간 기반 분할
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        # 훈련, 검증, 테스트 세트 인덱스
        train_idx = df.index[:train_size]
        val_idx = df.index[train_size:train_size + val_size]
        test_idx = df.index[train_size + val_size:]
        
        # 데이터 분할
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]
        
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]
        
        logger.info(f"훈련 세트: {X_train.index.min()} ~ {X_train.index.max()} ({len(X_train)} 행)")
        logger.info(f"검증 세트: {X_val.index.min()} ~ {X_val.index.max()} ({len(X_val)} 행)")
        logger.info(f"테스트 세트: {X_test.index.min()} ~ {X_test.index.max()} ({len(X_test)} 행)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        logger.error(f"데이터 분할 중 오류: {str(e)}")
        raise

def normalize_features(X_train, X_val, X_test):
    """
    특성 정규화
    """
    logger.info("특성 정규화 시작")
    try:
        # 수치형 특성만 선택
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        
        # StandardScaler로 정규화
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_val_scaled[numeric_features] = scaler.transform(X_val[numeric_features])
        X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
        
        logger.info(f"정규화된 수치형 특성: {list(numeric_features)}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    except Exception as e:
        logger.error(f"특성 정규화 중 오류: {str(e)}")
        raise

def prepare_data_pipeline(data_path, residual_path, train_ratio=0.7, val_ratio=0.15, 
                          add_time_feats=True, add_lag_feats=True, normalize=True,
                          lag_periods=DEFAULT_LAG_FEATURES, forecast_horizon=1):
    """
    데이터 준비 전체 파이프라인
    """
    logger.info("데이터 준비 파이프라인 시작")
    
    # 1. 데이터 로드
    data_df = load_original_data(data_path)
    residuals_df = load_residuals(residual_path)
    
    # 2. 데이터 병합
    merged_df = merge_data_with_residuals(data_df, residuals_df)
    
    # 3. 특성 엔지니어링
    if add_time_feats:
        merged_df = add_time_features(merged_df)
    
    if add_lag_feats:
        merged_df = add_lag_features(merged_df, lags=lag_periods)
    
    # 4. 데이터 분할
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        merged_df, train_ratio=train_ratio, val_ratio=val_ratio,
        forecast_horizon=forecast_horizon
    )
    
    # 5. 특성 정규화 (선택적)
    if normalize:
        X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)
    else:
        scaler = None
    
    logger.info("데이터 준비 파이프라인 완료")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler



def align_train_test_splits(sarima_predictions, original_data, lag_features=DEFAULT_LAG_FEATURES, 
                           train_ratio=0.7, val_ratio=0.15):
    """
    시차 변수 추가와 데이터 손실을 고려하여 SARIMA 예측과 트리 모델을 위한 
    학습/검증/테스트 세트를 일치시킵니다.
    
    Parameters:
    -----------
    sarima_predictions : SARIMA 모델의 예측 및 잔차 데이터프레임
    original_data : 원본 데이터
    lag_features : 추가할 시차 변수의 수
    train_ratio : 훈련 세트 비율
    val_ratio : 검증 세트 비율
    
    Returns:
    --------
    aligned_data : 일치된 데이터 (특성 + 타겟 + SARIMA 예측 + 잔차)
    train_idx, val_idx, test_idx : 각 분할의 인덱스
    """
    logger.info("데이터 분할 일치시키기 시작")
    
    try:
        # 1. 원본 데이터에서 타겟 제거 (특성만 유지)
        if 'sic' in original_data.columns:
            features_only = original_data.drop(columns=['sic'])
        else:
            features_only = original_data.copy()
        
        # 2. 특성과 SARIMA 결과 병합
        merged_data = pd.merge(sarima_predictions, features_only, 
                              left_index=True, right_index=True, how='inner')
        
        # 3. 시간 특성 추가
        merged_data = add_time_features(merged_data)
        
        # 4. 시차 변수 추가 (잔차에 대한 시차 변수)
        for i in range(1, lag_features + 1):
            merged_data[f'residual_lag_{i}'] = merged_data['residual'].shift(i)
        
        # 5. NaN 행 제거
        original_len = len(merged_data)
        merged_data = merged_data.dropna()
        logger.info(f"시차 변수 추가 후 NaN 제거: {original_len} -> {len(merged_data)} 행")
        
        # 6. 데이터 분할 인덱스 계산
        n = len(merged_data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        # 7. 인덱스 정의
        all_indices = merged_data.index
        train_idx = all_indices[:train_size]
        val_idx = all_indices[train_size:train_size + val_size]
        test_idx = all_indices[train_size + val_size:]
        
        logger.info(f"훈련 세트: {train_idx.min()} ~ {train_idx.max()} ({len(train_idx)} 행)")
        logger.info(f"검증 세트: {val_idx.min()} ~ {val_idx.max()} ({len(val_idx)} 행)")
        logger.info(f"테스트 세트: {test_idx.min()} ~ {test_idx.max()} ({len(test_idx)} 행)")
        
        return merged_data, train_idx, val_idx, test_idx
        
    except Exception as e:
        logger.error(f"데이터 분할 일치 중 오류: {str(e)}")
        raise