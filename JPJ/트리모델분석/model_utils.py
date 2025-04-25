"""
모델 유틸리티: 모델 생성, 학습, 최적화, 평가 등
"""
import numpy as np
import pandas as pd
import os
import logging
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import shap
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from config import (
    RF_PARAM_SPACE, LGBM_PARAM_SPACE, XGB_PARAM_SPACE, CAT_PARAM_SPACE,
    DEFAULT_RF_PARAMS, DEFAULT_LGBM_PARAMS, DEFAULT_XGB_PARAMS, DEFAULT_CAT_PARAMS,
    DEFAULT_CV_FOLDS, DEFAULT_BAYES_ITER, DEFAULT_BAYES_INIT_POINTS, RANDOM_SEED
)

logger = logging.getLogger(__name__)

def get_model(model_type, params=None):
    """
    선택한 모델 타입의 인스턴스 생성
    """
    logger.info(f"{model_type} 모델 생성")
    
    if model_type == 'rf':
        model_params = DEFAULT_RF_PARAMS if params is None else params
        return RandomForestRegressor(**model_params)
    
    elif model_type == 'lgbm':
        model_params = DEFAULT_LGBM_PARAMS if params is None else params
        return lgb.LGBMRegressor(**model_params)
    
    elif model_type == 'xgb':
        model_params = DEFAULT_XGB_PARAMS if params is None else params
        return xgb.XGBRegressor(**model_params)
    
    elif model_type == 'cat':
        model_params = DEFAULT_CAT_PARAMS if params is None else params
        return cb.CatBoostRegressor(**model_params, verbose=0)
    
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

def optimize_hyperparams(model_type, X_train, y_train, X_val, y_val, 
                        n_iter=DEFAULT_BAYES_ITER, cv_folds=DEFAULT_CV_FOLDS, 
                        n_initial_points=DEFAULT_BAYES_INIT_POINTS):
    """
    베이지안 최적화를 사용한 하이퍼파라미터 최적화
    """
    logger.info(f"{model_type} 모델 하이퍼파라미터 최적화 시작 (n_iter={n_iter})")
    
    # 파라미터 공간 설정
    if model_type == 'rf':
        param_space = {
            'n_estimators': Integer(RF_PARAM_SPACE['n_estimators'][0], RF_PARAM_SPACE['n_estimators'][1]),
            'max_depth': Integer(RF_PARAM_SPACE['max_depth'][0], RF_PARAM_SPACE['max_depth'][1]),
            'min_samples_split': Integer(RF_PARAM_SPACE['min_samples_split'][0], RF_PARAM_SPACE['min_samples_split'][1]),
            'min_samples_leaf': Integer(RF_PARAM_SPACE['min_samples_leaf'][0], RF_PARAM_SPACE['min_samples_leaf'][1])
        }
        model = RandomForestRegressor(random_state=RANDOM_SEED)
    
    elif model_type == 'lgbm':
        param_space = {
            'n_estimators': Integer(LGBM_PARAM_SPACE['n_estimators'][0], LGBM_PARAM_SPACE['n_estimators'][1]),
            'learning_rate': Real(LGBM_PARAM_SPACE['learning_rate'][0], LGBM_PARAM_SPACE['learning_rate'][1], prior='log-uniform'),
            'max_depth': Integer(LGBM_PARAM_SPACE['max_depth'][0], LGBM_PARAM_SPACE['max_depth'][1]),
            'num_leaves': Integer(LGBM_PARAM_SPACE['num_leaves'][0], LGBM_PARAM_SPACE['num_leaves'][1]),
            'min_child_samples': Integer(LGBM_PARAM_SPACE['min_child_samples'][0], LGBM_PARAM_SPACE['min_child_samples'][1])
        }
        model = lgb.LGBMRegressor(random_state=RANDOM_SEED)
    
    elif model_type == 'xgb':
        param_space = {
            'n_estimators': Integer(XGB_PARAM_SPACE['n_estimators'][0], XGB_PARAM_SPACE['n_estimators'][1]),
            'learning_rate': Real(XGB_PARAM_SPACE['learning_rate'][0], XGB_PARAM_SPACE['learning_rate'][1], prior='log-uniform'),
            'max_depth': Integer(XGB_PARAM_SPACE['max_depth'][0], XGB_PARAM_SPACE['max_depth'][1]),
            'min_child_weight': Integer(XGB_PARAM_SPACE['min_child_weight'][0], XGB_PARAM_SPACE['min_child_weight'][1]),
            'subsample': Real(XGB_PARAM_SPACE['subsample'][0], XGB_PARAM_SPACE['subsample'][1]),
            'colsample_bytree': Real(XGB_PARAM_SPACE['colsample_bytree'][0], XGB_PARAM_SPACE['colsample_bytree'][1])
        }
        model = xgb.XGBRegressor(random_state=RANDOM_SEED)
    
    elif model_type == 'cat':
        param_space = {
            'iterations': Integer(CAT_PARAM_SPACE['iterations'][0], CAT_PARAM_SPACE['iterations'][1]),
            'learning_rate': Real(CAT_PARAM_SPACE['learning_rate'][0], CAT_PARAM_SPACE['learning_rate'][1], prior='log-uniform'),
            'depth': Integer(CAT_PARAM_SPACE['depth'][0], CAT_PARAM_SPACE['depth'][1]),
            'l2_leaf_reg': Real(CAT_PARAM_SPACE['l2_leaf_reg'][0], CAT_PARAM_SPACE['l2_leaf_reg'][1]),
            'bagging_temperature': Real(CAT_PARAM_SPACE['bagging_temperature'][0], CAT_PARAM_SPACE['bagging_temperature'][1])
        }
        model = cb.CatBoostRegressor(random_seed=RANDOM_SEED, verbose=0)
    
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    # 시계열 교차 검증 설정
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    # 베이지안 최적화
    opt = BayesSearchCV(
        model,
        param_space,
        n_iter=n_iter,
        cv=tscv,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        random_state=RANDOM_SEED,
        n_points=n_initial_points,
        verbose=1
    )
    
    # 최적화 실행
    logger.info("베이지안 최적화 실행 중...")
    opt.fit(X_train, y_train)
    
    # 최적 파라미터 및 점수
    best_params = opt.best_params_
    best_score = np.sqrt(-opt.best_score_)
    
    logger.info(f"최적 파라미터: {best_params}")
    logger.info(f"최적 RMSE: {best_score:.6f}")
    
    # 검증 세트에서 평가
    best_model = get_model(model_type, best_params)
    best_model.fit(X_train, y_train)
    y_val_pred = best_model.predict(X_val)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    logger.info(f"검증 세트 성능 - RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.6f}")
    
    return best_params, best_model, val_rmse, val_mae, val_r2

def train_model(model_type, X_train, y_train, params=None):
    """
    모델 학습
    """
    logger.info(f"{model_type} 모델 학습 시작")
    
    model = get_model(model_type, params)
    model.fit(X_train, y_train)
    
    logger.info(f"{model_type} 모델 학습 완료")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    모델 평가
    """
    logger.info(f"모델 평가 시작")
    
    y_pred = model.predict(X_test)
    
    # 평가 지표 계산
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"테스트 세트 성능 - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
    
    # 예측 결과 DataFrame
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'error': y_test - y_pred
    })
    
    return rmse, mae, r2, results_df

def calculate_feature_importance(model, model_type, X_train, output_dir):
    """
    모델 특성 중요도 계산 및 저장
    """
    logger.info(f"{model_type} 모델 특성 중요도 계산 시작")
    
    # 모델 내장 특성 중요도
    if model_type == 'rf':
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    elif model_type == 'lgbm':
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    elif model_type == 'xgb':
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    elif model_type == 'cat':
        importances = model.get_feature_importance()
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    # 특성 중요도 저장
    feature_importance.to_csv(os.path.join(output_dir, f"{model_type}_feature_importance.csv"), index=False)
    logger.info(f"모델 특성 중요도 저장 완료: {os.path.join(output_dir, f'{model_type}_feature_importance.csv')}")
    
    return feature_importance

def calculate_shap_values(model, model_type, X_train, X_test, output_dir):
    """
    SHAP 값 계산 및 저장
    """
    logger.info(f"{model_type} 모델 SHAP 값 계산 시작")
    
    # 모델 타입별 SHAP 값 계산
    if model_type == 'rf':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    
    elif model_type == 'lgbm':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    
    elif model_type == 'xgb':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    
    elif model_type == 'cat':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    
    # SHAP 값 저장
    shap_values_df = pd.DataFrame(
        shap_values, 
        columns=X_test.columns,
        index=X_test.index
    )
    shap_values_df.to_csv(os.path.join(output_dir, f"{model_type}_shap_values.csv"))
    
    # 기준값 저장
    pd.Series(explainer.expected_value).to_csv(os.path.join(output_dir, f"{model_type}_shap_expected_value.csv"))
    
    logger.info(f"SHAP 값 저장 완료: {os.path.join(output_dir, f'{model_type}_shap_values.csv')}")
    
    return explainer, shap_values

def save_model(model, model_type, output_dir):
    """
    학습된 모델 저장
    """
    logger.info(f"{model_type} 모델 저장")
    
    model_path = os.path.join(output_dir, f"{model_type}_model.pkl")
    joblib.dump(model, model_path)
    
    logger.info(f"모델 저장 완료: {model_path}")
    
    return model_path

def load_model(model_path):
    """
    저장된 모델 로드
    """
    logger.info(f"모델 로드: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"모델 파일 '{model_path}'을 찾을 수 없습니다.")
        raise FileNotFoundError(f"모델 파일 '{model_path}'을 찾을 수 없습니다.")
    
    model = joblib.load(model_path)
    
    logger.info(f"모델 로드 완료")
    
    return model


def evaluate_hybrid_model(sarima_predictions, tree_model_predictions, test_idx, output_dir=None):
    """
    하이브리드 모델(SARIMA + 트리 모델)의 성능을 평가합니다.
    
    Parameters:
    -----------
    sarima_predictions : SARIMA 모델 예측 결과 데이터프레임
    tree_model_predictions : 트리 모델 예측 결과 데이터프레임
    test_idx : 테스트 세트 인덱스
    output_dir : 결과 저장 디렉토리
    
    Returns:
    --------
    hybrid_results : 하이브리드 모델 평가 결과
    results_df : 모든 모델의 예측값과 실제값 포함 데이터프레임
    """
    logger.info("하이브리드 모델 평가 시작")
    
    try:
        # 1. 테스트 세트 추출
        sarima_test = sarima_predictions.loc[test_idx]
        tree_test = tree_model_predictions.loc[test_idx]
        
        # 2. 하이브리드 모델 예측값 계산 (SARIMA 예측 + 트리 모델 잔차 예측)
        hybrid_pred = sarima_test['sarima_pred'] + tree_test['predicted']
        
        # 3. 실제값
        actual = sarima_test['actual']
        
        # 4. 결과 데이터프레임 구성
        results_df = pd.DataFrame({
            'actual': actual,
            'sarima_pred': sarima_test['sarima_pred'],
            'tree_pred': tree_test['predicted'],
            'hybrid_pred': hybrid_pred,
            'sarima_error': actual - sarima_test['sarima_pred'],
            'tree_error': actual - tree_test['predicted'],
            'hybrid_error': actual - hybrid_pred
        })
        
        # 5. 평가 지표 계산
        # SARIMA 모델
        sarima_rmse = np.sqrt(mean_squared_error(actual, sarima_test['sarima_pred']))
        sarima_mae = mean_absolute_error(actual, sarima_test['sarima_pred'])
        sarima_r2 = r2_score(actual, sarima_test['sarima_pred'])
        
        # 트리 모델 (잔차만)
        tree_rmse = np.sqrt(mean_squared_error(sarima_test['residual'], tree_test['predicted']))
        tree_mae = mean_absolute_error(sarima_test['residual'], tree_test['predicted'])
        tree_r2 = r2_score(sarima_test['residual'], tree_test['predicted'])
        
        # 하이브리드 모델 (최종 예측)
        hybrid_rmse = np.sqrt(mean_squared_error(actual, hybrid_pred))
        hybrid_mae = mean_absolute_error(actual, hybrid_pred)
        hybrid_r2 = r2_score(actual, hybrid_pred)
        
        # 6. 결과 저장
        hybrid_results = {
            'model_type': 'hybrid',
            'sarima_rmse': sarima_rmse,
            'sarima_mae': sarima_mae,
            'sarima_r2': sarima_r2,
            'tree_rmse': tree_rmse,
            'tree_mae': tree_mae,
            'tree_r2': tree_r2,
            'hybrid_rmse': hybrid_rmse,
            'hybrid_mae': hybrid_mae,
            'hybrid_r2': hybrid_r2,
            'test_period': f"{test_idx.min()} ~ {test_idx.max()}"
        }
        
        logger.info(f"SARIMA 모델 성능 - RMSE: {sarima_rmse:.6f}, MAE: {sarima_mae:.6f}, R²: {sarima_r2:.6f}")
        logger.info(f"트리 모델 성능 - RMSE: {tree_rmse:.6f}, MAE: {tree_mae:.6f}, R²: {tree_r2:.6f}")
        logger.info(f"하이브리드 모델 성능 - RMSE: {hybrid_rmse:.6f}, MAE: {hybrid_mae:.6f}, R²: {hybrid_r2:.6f}")
        
        if output_dir:
            # 평가 지표 저장
            pd.DataFrame([hybrid_results]).to_csv(
                os.path.join(output_dir, 'hybrid_evaluation.csv'), index=False
            )
            
            # 예측 결과 저장
            results_df.to_csv(os.path.join(output_dir, 'hybrid_predictions.csv'))
            
            logger.info(f"하이브리드 모델 평가 결과 저장 완료: {output_dir}")
        
        return hybrid_results, results_df
        
    except Exception as e:
        logger.error(f"하이브리드 모델 평가 중 오류: {str(e)}")
        raise