"""
시각화 유틸리티: 결과 시각화, 특성 중요도 시각화 등
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import shap
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

def plot_residuals(residuals, output_dir):
    """
    SARIMA 모델 잔차 시각화
    """
    logger.info("SARIMA 잔차 시각화 시작")
    try:
        plt.figure(figsize=(12, 8))
        
        # 잔차 시계열 플롯
        plt.subplot(2, 1, 1)
        plt.plot(residuals.index, residuals, label='Residuals')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('SARIMA Model Residuals')
        plt.ylabel('Residual Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 잔차 히스토그램
        plt.subplot(2, 1, 2)
        sns.histplot(residuals, kde=True)
        plt.title('Residuals Distribution')
        plt.xlabel('Residual Value')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sarima_residuals.png'), dpi=300)
        
        # ACF, PACF 플롯
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plot_acf(residuals, ax=plt.gca(), lags=36)
        plt.title('ACF of Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plot_pacf(residuals, ax=plt.gca(), lags=36)
        plt.title('PACF of Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sarima_residuals_acf_pacf.png'), dpi=300)
        
        logger.info(f"SARIMA 잔차 시각화 완료: {output_dir}")
        
    except Exception as e:
        logger.error(f"SARIMA 잔차 시각화 중 오류: {str(e)}")
        raise

def plot_feature_importance(importance_df, model_type, output_dir):
    """
    모델 특성 중요도 시각화
    """
    logger.info(f"{model_type} 모델 특성 중요도 시각화 시작")
    try:
        # 상위 15개 특성만 선택
        top_n = min(15, len(importance_df))
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importance - {model_type.upper()}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_type}_feature_importance.png'), dpi=300)
        
        logger.info(f"특성 중요도 시각화 완료: {os.path.join(output_dir, f'{model_type}_feature_importance.png')}")
        
    except Exception as e:
        logger.error(f"특성 중요도 시각화 중 오류: {str(e)}")
        raise

def plot_shap_summary(explainer, shap_values, X_test, model_type, output_dir):
    """
    SHAP 값 요약 시각화
    """
    logger.info(f"{model_type} 모델 SHAP 요약 시각화 시작")
    try:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f'SHAP Summary Plot - {model_type.upper()}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_type}_shap_summary.png'), dpi=300)
        
        # 상위 10개 특성에 대한 개별 의존성 플롯
        top_features = np.argsort(-np.abs(shap_values).mean(0))[:10]
        for i, feature_idx in enumerate(top_features):
            feature_name = X_test.columns[feature_idx]
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(
                feature_idx, 
                shap_values, 
                X_test, 
                show=False,
                interaction_index=None
            )
            plt.title(f'SHAP Dependence Plot - {feature_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_type}_shap_dependence_{feature_name}.png'), dpi=300)
        
        logger.info(f"SHAP 시각화 완료: {output_dir}")
        
    except Exception as e:
        logger.error(f"SHAP 시각화 중 오류: {str(e)}")
        raise

def plot_actual_vs_predicted(actual, predicted, model_type, output_dir):
    """
    실제값 vs 예측값 시각화
    """
    logger.info(f"{model_type} 모델 실제값 vs 예측값 시각화 시작")
    try:
        plt.figure(figsize=(12, 8))
        
        # 시계열 플롯
        plt.subplot(2, 1, 1)
        plt.plot(actual.index, actual, label='Actual', marker='o')
        plt.plot(predicted.index, predicted, label='Predicted', marker='x')
        plt.title(f'Actual vs Predicted Residuals - {model_type.upper()}')
        plt.ylabel('Residual Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 산점도
        plt.subplot(2, 1, 2)
        plt.scatter(actual, predicted)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
        plt.title('Actual vs Predicted Scatter Plot')
        plt.xlabel('Actual Residual')
        plt.ylabel('Predicted Residual')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_type}_actual_vs_predicted.png'), dpi=300)
        
        # 오차 히스토그램
        plt.figure(figsize=(10, 6))
        error = actual - predicted
        sns.histplot(error, kde=True)
        plt.title(f'Prediction Error Distribution - {model_type.upper()}')
        plt.xlabel('Error')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_type}_error_distribution.png'), dpi=300)
        
        logger.info(f"실제값 vs 예측값 시각화 완료: {output_dir}")
        
    except Exception as e:
        logger.error(f"실제값 vs 예측값 시각화 중 오류: {str(e)}")
        raise

def plot_model_comparison(results_dict, output_dir):
    """
    여러 모델 성능 비교 시각화
    """
    logger.info(f"모델 성능 비교 시각화 시작")
    try:
        metrics = ['rmse', 'mae', 'r2']
        models = list(results_dict.keys())
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            values = [results_dict[model][metric] for model in models]
            
            # R2는 높을수록 좋음, 나머지는 낮을수록 좋음
            if metric == 'r2':
                colors = ['g' if v == max(values) else 'b' for v in values]
                plt.axhline(y=1, color='r', linestyle='--', alpha=0.3, label='Perfect Score')
            else:
                colors = ['g' if v == min(values) else 'b' for v in values]
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Perfect Score')
            
            bars = plt.bar(models, values, color=colors)
            
            # 바 위에 값 표시
            for bar, value in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(values) * 0.01),
                    f'{value:.6f}',
                    ha='center', va='bottom',
                    rotation=0
                )
            
            plt.title(f'Model Comparison - {metric.upper()}')
            plt.ylabel(metric.upper())
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'model_comparison_{metric}.png'), dpi=300)
        
        logger.info(f"모델 성능 비교 시각화 완료: {output_dir}")
        
    except Exception as e:
        logger.error(f"모델 성능 비교 시각화 중 오류: {str(e)}")
        raise

def plot_forecast_horizon_comparison(horizons, metric_values, metric_name, output_dir):
    """
    다양한 예측 기간별 성능 비교 시각화
    """
    logger.info(f"예측 기간별 {metric_name} 비교 시각화 시작")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(horizons, metric_values, marker='o', linestyle='-')
        
        # 각 점에 값 표시
        for i, (h, v) in enumerate(zip(horizons, metric_values)):
            plt.text(h, v + (max(metric_values) * 0.02), f'{v:.6f}', ha='center')
        
        plt.title(f'{metric_name} by Forecast Horizon')
        plt.xlabel('Forecast Horizon (Months)')
        plt.ylabel(metric_name)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'horizon_comparison_{metric_name.lower()}.png'), dpi=300)
        
        logger.info(f"예측 기간별 비교 시각화 완료: {output_dir}")
        
    except Exception as e:
        logger.error(f"예측 기간별 비교 시각화 중 오류: {str(e)}")
        raise

def plot_hybrid_comparison(results_df, model_type, output_dir):
    """
    SARIMA, 트리 모델, 하이브리드 모델의 예측 성능 비교 시각화
    
    Parameters:
    -----------
    results_df : 모든 모델의 예측값과 실제값 포함 데이터프레임
    model_type : 트리 모델 타입
    output_dir : 결과 저장 디렉토리
    """
    logger.info("하이브리드 모델 비교 시각화 시작")
    
    try:
        plt.figure(figsize=(12, 8))
        
        # 1. 시계열 비교 플롯
        plt.subplot(2, 1, 1)
        plt.plot(results_df.index, results_df['actual'], label='Actual', marker='o', alpha=0.7)
        plt.plot(results_df.index, results_df['sarima_pred'], label='SARIMA', linestyle='--', alpha=0.7)
        plt.plot(results_df.index, results_df['hybrid_pred'], label='Hybrid', linestyle='-', alpha=0.7)
        
        plt.title(f'Actual vs Predicted SIC - SARIMA and Hybrid ({model_type.upper()})')
        plt.ylabel('SIC Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. 오차 비교 플롯
        plt.subplot(2, 1, 2)
        plt.plot(results_df.index, results_df['sarima_error'], label='SARIMA Error', alpha=0.7)
        plt.plot(results_df.index, results_df['hybrid_error'], label='Hybrid Error', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        plt.title('Prediction Errors Comparison')
        plt.ylabel('Error')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'hybrid_{model_type}_comparison.png'), dpi=300)
        
        # 3. 모델별 RMSE, MAE, R² 비교 바차트
        metrics_data = []
        
        # SARIMA 모델 성능
        sarima_rmse = np.sqrt(mean_squared_error(results_df['actual'], results_df['sarima_pred']))
        sarima_mae = mean_absolute_error(results_df['actual'], results_df['sarima_pred'])
        sarima_r2 = r2_score(results_df['actual'], results_df['sarima_pred'])
        
        # 하이브리드 모델 성능
        hybrid_rmse = np.sqrt(mean_squared_error(results_df['actual'], results_df['hybrid_pred']))
        hybrid_mae = mean_absolute_error(results_df['actual'], results_df['hybrid_pred'])
        hybrid_r2 = r2_score(results_df['actual'], results_df['hybrid_pred'])
        
        # 데이터 준비
        metrics_data.append({
            'Model': 'SARIMA', 
            'RMSE': sarima_rmse, 
            'MAE': sarima_mae, 
            'R²': sarima_r2
        })
        
        metrics_data.append({
            'Model': f'Hybrid ({model_type.upper()})', 
            'RMSE': hybrid_rmse, 
            'MAE': hybrid_mae, 
            'R²': hybrid_r2
        })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # 각 지표별 막대 그래프
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RMSE (낮을수록 좋음)
        sns.barplot(x='Model', y='RMSE', data=metrics_df, ax=axes[0])
        for i, v in enumerate(metrics_df['RMSE']):
            axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center')
        axes[0].set_title('RMSE Comparison (Lower is Better)')
        
        # MAE (낮을수록 좋음)
        sns.barplot(x='Model', y='MAE', data=metrics_df, ax=axes[1])
        for i, v in enumerate(metrics_df['MAE']):
            axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center')
        axes[1].set_title('MAE Comparison (Lower is Better)')
        
        # R² (높을수록 좋음)
        sns.barplot(x='Model', y='R²', data=metrics_df, ax=axes[2])
        for i, v in enumerate(metrics_df['R²']):
            axes[2].text(i, v + 0.01, f'{v:.4f}', ha='center')
        axes[2].set_title('R² Comparison (Higher is Better)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'hybrid_{model_type}_metrics_comparison.png'), dpi=300)
        
        logger.info(f"하이브리드 모델 비교 시각화 완료: {output_dir}")
        
    except Exception as e:
        logger.error(f"하이브리드 모델 비교 시각화 중 오류: {str(e)}")
        raise