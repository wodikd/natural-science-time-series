"""
해빙 농도(SIC) 분석 도구 - 메인 실행 스크립트
SARIMA 모델과 트리 모델을 결합한 하이브리드 접근을 통해 해빙 농도 예측 및 특성 중요도 분석
"""
import os
import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# 한글 폰트 설정 (로그 메시지에 한글 사용 가능성 대비)
import matplotlib as mpl
mpl.rc('font', family='Malgun Gothic')
mpl.rcParams['axes.unicode_minus'] = False

# 모듈 임포트
from config import (
    DEFAULT_DATA_PATH, DEFAULT_RESIDUAL_PATH, DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, DEFAULT_LAG_FEATURES,
    DEFAULT_FORECAST_HORIZON, DEFAULT_CV_FOLDS, DEFAULT_BAYES_ITER
)

from data_utils import (
    load_original_data, load_residuals, merge_data_with_residuals,
    add_time_features, add_lag_features, split_data, normalize_features,
    prepare_data_pipeline, align_train_test_splits
)

from sarima_utils import (
    check_stationarity, run_auto_arima, fit_sarima_model,
    sarima_forecast, run_sarima_pipeline, get_sarima_predictions
)

from model_utils import (
    get_model, optimize_hyperparams, train_model, evaluate_model,
    calculate_feature_importance, calculate_shap_values, save_model,
    evaluate_hybrid_model
)

from visualization import (
    plot_residuals, plot_feature_importance, plot_shap_summary,
    plot_actual_vs_predicted, plot_model_comparison,
    plot_forecast_horizon_comparison, plot_hybrid_comparison
)

# 로그 설정
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'analysis_log.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger()
    return logger

def interactive_mode():
    """
    대화형 모드 실행
    """
    print("\n===== 해빙 농도 예측 분석 도구 =====\n")
    
    # 실행 모드 선택
    print("실행 모드를 선택하세요:")
    print("1. 전체 파이프라인 (SARIMA + 트리 모델)")
    print("2. 기존 잔차로 트리 모델만 실행")
    print("3. 하이브리드 모델 (SARIMA + 트리 모델 결합)")
    
    while True:
        mode_choice = input("\n선택 (1/2/3): ")
        if mode_choice in ['1', '2', '3']:
            break
        print("잘못된 선택입니다. 1, 2 또는 3을 입력하세요.")
    
    mode_map = {
        '1': 'full',
        '2': 'tree_only',
        '3': 'hybrid'
    }
    
    mode = mode_map[mode_choice]
    
    # 데이터 경로 설정
    if mode == 'full' or mode == 'hybrid':
        data_path = input("\n원본 데이터 파일 경로 (Enter로 기본값 사용): ")
        data_path = data_path.strip() if data_path.strip() else DEFAULT_DATA_PATH
        residual_path = None
    else:
        data_path = input("\n원본 데이터 파일 경로 (Enter로 기본값 사용): ")
        data_path = data_path.strip() if data_path.strip() else DEFAULT_DATA_PATH
        
        residual_path = input("잔차 데이터 파일 경로: ")
        if not residual_path.strip():
            print("잔차 파일 경로는 필수입니다.")
            return
    
    # 트리 모델 선택
    print("\n사용할 트리 모델을 선택하세요:")
    print("1. Random Forest")
    print("2. LightGBM")
    print("3. XGBoost")
    print("4. CatBoost")
    
    model_map = {
        '1': 'rf',
        '2': 'lgbm',
        '3': 'xgb',
        '4': 'cat'
    }
    
    while True:
        model_choice = input("\n선택 (1/2/3/4): ")
        if model_choice in model_map:
            break
        print("잘못된 선택입니다. 1, 2, 3, 4 중 하나를 입력하세요.")
    
    model_type = model_map[model_choice]
    
    # 예측 기간 설정
    while True:
        forecast_horizon = input(f"\n예측 기간(개월) (Enter로 기본값 {DEFAULT_FORECAST_HORIZON} 사용): ")
        if not forecast_horizon:
            forecast_horizon = DEFAULT_FORECAST_HORIZON
            break
        try:
            forecast_horizon = int(forecast_horizon)
            if forecast_horizon > 0:
                break
            print("예측 기간은 1 이상이어야 합니다.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    # 하이퍼파라미터 최적화 여부
    while True:
        optimize = input("\n하이퍼파라미터 최적화를 수행하시겠습니까? (y/n): ")
        if optimize.lower() in ['y', 'n']:
            break
        print("y 또는 n을 입력하세요.")
    
    optimize = optimize.lower() == 'y'
    
    # 출력 디렉토리 설정
    output_dir = input(f"\n결과 저장 디렉토리 (Enter로 타임스탬프 자동 생성): ")
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_type.upper()
        output_dir = f"analysis_{model_name}_{timestamp}"
    
    # 실행 명령어 구성
    cmd_args = {
        'mode': mode,
        'model': model_type,
        'data_path': data_path,
        'residual_path': residual_path,
        'output_dir': output_dir,
        'forecast_horizon': forecast_horizon,
        'optimize': optimize,
        'train_ratio': DEFAULT_TRAIN_RATIO,
        'val_ratio': DEFAULT_VAL_RATIO,
        'lag_features': DEFAULT_LAG_FEATURES,
        'bayes_iter': DEFAULT_BAYES_ITER,
        'cv_folds': DEFAULT_CV_FOLDS
    }
    
    print("\n===== 실행 설정 =====")
    for key, value in cmd_args.items():
        print(f"{key}: {value}")
    
    confirm = input("\n이 설정으로 분석을 시작하시겠습니까? (y/n): ")
    if confirm.lower() != 'y':
        print("분석이 취소되었습니다.")
        return
    
    # 분석 실행
    run_with_args(cmd_args)
    
    print(f"\n분석이 완료되었습니다. 결과는 '{output_dir}' 디렉토리에 저장되었습니다.")

def run_with_args(args):
    """
    명령줄 인자로 분석 실행
    """
    # 출력 디렉토리 및 로깅 설정
    logger = setup_logging(args['output_dir'])
    logger.info("해빙 농도 예측 분석 시작")
    logger.info(f"실행 설정: {args}")
    
    try:
        # 실행 모드에 따른 분기
        if args['mode'] == 'full':
            # 전체 파이프라인 (SARIMA + 트리 모델)
            run_full_pipeline(args, logger)
        elif args['mode'] == 'tree_only':
            # 트리 모델만 실행
            run_tree_only_pipeline(args, logger)
        elif args['mode'] == 'hybrid':
            # 하이브리드 모델 실행
            hybrid_results, hybrid_df, model = run_hybrid_pipeline(args, logger)
            logger.info(f"하이브리드 모델 성능 - RMSE: {hybrid_results['hybrid_rmse']:.6f}, MAE: {hybrid_results['hybrid_mae']:.6f}, R²: {hybrid_results['hybrid_r2']:.6f}")
            
        logger.info("분석 완료")
        
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}", exc_info=True)
        raise

def run_full_pipeline(args, logger):
    """
    SARIMA와 트리 모델을 포함한 전체 파이프라인 실행
    """
    logger.info("전체 파이프라인 (SARIMA + 트리 모델) 실행")
    
    # 1. 원본 데이터 로드
    data_df = load_original_data(args['data_path'])
    
    # 2. SARIMA 모델 학습 및 잔차 생성
    sarima_results, residuals_df, order, seasonal_order = run_sarima_pipeline(
        data_df, target_col='sic', train_ratio=args['train_ratio'], 
        output_dir=args['output_dir']
    )
    
    # 3. 잔차 시각화
    plot_residuals(residuals_df['residual'], args['output_dir'])
    
    # 4. 데이터 준비 (트리 모델용)
    merged_df = merge_data_with_residuals(data_df, residuals_df)
    merged_df = add_time_features(merged_df)
    merged_df = add_lag_features(merged_df, lags=args['lag_features'])
    
    # 5. 데이터 분할
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        merged_df, train_ratio=args['train_ratio'], val_ratio=args['val_ratio'],
        forecast_horizon=args['forecast_horizon']
    )
    
    # 6. 특성 정규화
    X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)
    
    # 7. 모델 학습 및 최적화
    if args['optimize']:
        logger.info(f"{args['model']} 모델 하이퍼파라미터 최적화")
        best_params, model, val_rmse, val_mae, val_r2 = optimize_hyperparams(
            args['model'], X_train, y_train, X_val, y_val,
            n_iter=args['bayes_iter'], cv_folds=args['cv_folds']
        )
    else:
        logger.info(f"{args['model']} 모델 기본 파라미터로 학습")
        model = train_model(args['model'], X_train, y_train)
    
    # 8. 모델 평가
    rmse, mae, r2, results_df = evaluate_model(model, X_test, y_test)
    
    # 9. 결과 저장
    model_path = save_model(model, args['model'], args['output_dir'])
    
    # 10. 특성 중요도 계산 및 시각화
    feature_importance = calculate_feature_importance(model, args['model'], X_train, args['output_dir'])
    plot_feature_importance(feature_importance, args['model'], args['output_dir'])
    
    # 11. SHAP 값 계산 및 시각화
    explainer, shap_values = calculate_shap_values(model, args['model'], X_train, X_test, args['output_dir'])
    plot_shap_summary(explainer, shap_values, X_test, args['model'], args['output_dir'])
    
    # 12. 실제값 vs 예측값 시각화
    plot_actual_vs_predicted(y_test, results_df['predicted'], args['model'], args['output_dir'])
    
    # 13. 최종 평가 결과 저장
    evaluation_results = {
        'sarima_order': order,
        'sarima_seasonal_order': seasonal_order,
        'model_type': args['model'],
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'test_period': f"{X_test.index.min()} ~ {X_test.index.max()}"
    }
    
    pd.DataFrame([evaluation_results]).to_csv(
        os.path.join(args['output_dir'], 'final_evaluation.csv'), index=False
    )
    
    logger.info(f"전체 파이프라인 완료 - 평가 지표: RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")

def run_tree_only_pipeline(args, logger):
    """
    기존 잔차를 사용하여 트리 모델만 실행
    """
    logger.info("트리 모델만 실행 (기존 잔차 사용)")
    
    # 1. 데이터 준비
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data_pipeline(
        args['data_path'], args['residual_path'],
        train_ratio=args['train_ratio'], val_ratio=args['val_ratio'],
        add_time_feats=True, add_lag_feats=True, normalize=True,
        lag_periods=args['lag_features'], forecast_horizon=args['forecast_horizon']
    )
    
    # 2. 모델 학습 및 최적화
    if args['optimize']:
        logger.info(f"{args['model']} 모델 하이퍼파라미터 최적화")
        best_params, model, val_rmse, val_mae, val_r2 = optimize_hyperparams(
            args['model'], X_train, y_train, X_val, y_val,
            n_iter=args['bayes_iter'], cv_folds=args['cv_folds']
        )
    else:
        logger.info(f"{args['model']} 모델 기본 파라미터로 학습")
        model = train_model(args['model'], X_train, y_train)
    
    # 3. 모델 평가
    rmse, mae, r2, results_df = evaluate_model(model, X_test, y_test)
    
    # 4. 결과 저장
    model_path = save_model(model, args['model'], args['output_dir'])
    
    # 5. 특성 중요도 계산 및 시각화
    feature_importance = calculate_feature_importance(model, args['model'], X_train, args['output_dir'])
    plot_feature_importance(feature_importance, args['model'], args['output_dir'])
    
    # 6. SHAP 값 계산 및 시각화
    explainer, shap_values = calculate_shap_values(model, args['model'], X_train, X_test, args['output_dir'])
    plot_shap_summary(explainer, shap_values, X_test, args['model'], args['output_dir'])
    
    # 7. 실제값 vs 예측값 시각화
    plot_actual_vs_predicted(y_test, results_df['predicted'], args['model'], args['output_dir'])
    
    # 8. 최종 평가 결과 저장
    evaluation_results = {
        'model_type': args['model'],
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'test_period': f"{X_test.index.min()} ~ {X_test.index.max()}"
    }
    
    pd.DataFrame([evaluation_results]).to_csv(
        os.path.join(args['output_dir'], 'final_evaluation.csv'), index=False
    )
    
    logger.info(f"트리 모델 파이프라인 완료 - 평가 지표: RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")

def run_hybrid_pipeline(args, logger):
    """
    SARIMA 모델과 트리 모델을 결합한 하이브리드 모델 파이프라인 실행
    """
    logger.info("하이브리드 모델 파이프라인 시작")
    
    try:
        # 1. 원본 데이터 로드
        data_df = load_original_data(args['data_path'])
        
        # 2. SARIMA 모델 학습 및 예측
        sarima_results, residuals_df, order, seasonal_order = run_sarima_pipeline(
            data_df, target_col='sic', train_ratio=args['train_ratio'], 
            output_dir=args['output_dir']
        )
        
        # 3. SARIMA 모델 전체 예측값 생성 (인샘플 + 아웃샘플)
        sarima_predictions = get_sarima_predictions(
            sarima_results, data_df, target_col='sic', output_dir=args['output_dir']
        )
        
        # 4. 잔차 시각화
        plot_residuals(sarima_predictions['residual'], args['output_dir'])
        
        # 5. 데이터 분할 일치시키기 (SARIMA + 트리 모델)
        aligned_data, train_idx, val_idx, test_idx = align_train_test_splits(
            sarima_predictions, data_df, lag_features=args['lag_features'],
            train_ratio=args['train_ratio'], val_ratio=args['val_ratio']
        )
        
        # 6. 특성과 타겟 분리
        X = aligned_data.drop(columns=['actual', 'sarima_pred', 'residual'])
        y = aligned_data['residual']  # 트리 모델은 잔차를 예측
        
        # 7. 각 세트 분할
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]
        
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]
        
        # 8. 특성 정규화
        X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)
        
        # 9. 모델 학습 및 최적화
        if args['optimize']:
            logger.info(f"{args['model']} 모델 하이퍼파라미터 최적화")
            best_params, model, val_rmse, val_mae, val_r2 = optimize_hyperparams(
                args['model'], X_train, y_train, X_val, y_val,
                n_iter=args['bayes_iter'], cv_folds=args['cv_folds']
            )
        else:
            logger.info(f"{args['model']} 모델 기본 파라미터로 학습")
            model = train_model(args['model'], X_train, y_train)
        
        # 10. 트리 모델 평가
        tree_rmse, tree_mae, tree_r2, tree_results_df = evaluate_model(model, X_test, y_test)
        
        # 11. 모델 저장
        model_path = save_model(model, args['model'], args['output_dir'])
        
        # 12. 특성 중요도 계산 및 시각화
        feature_importance = calculate_feature_importance(model, args['model'], X_train, args['output_dir'])
        plot_feature_importance(feature_importance, args['model'], args['output_dir'])
        
        # 13. SHAP 값 계산 및 시각화
        explainer, shap_values = calculate_shap_values(model, args['model'], X_train, X_test, args['output_dir'])
        plot_shap_summary(explainer, shap_values, X_test, args['model'], args['output_dir'])
        
        # 14. 하이브리드 모델 평가
        hybrid_results, hybrid_df = evaluate_hybrid_model(
            sarima_predictions, tree_results_df, test_idx, args['output_dir']
        )
        
        # 15. 하이브리드 모델 시각화
        plot_hybrid_comparison(hybrid_df, args['model'], args['output_dir'])
        
        logger.info(f"하이브리드 모델 파이프라인 완료")
        
        return hybrid_results, hybrid_df, model
        
    except Exception as e:
        logger.error(f"하이브리드 모델 파이프라인 중 오류: {str(e)}")
        raise

def parse_arguments():
    """
    명령줄 인자 파싱
    """
    parser = argparse.ArgumentParser(description='해빙 농도(SIC) 예측 분석 도구')
    
    # 실행 모드
    parser.add_argument('--mode', type=str, choices=['full', 'tree_only', 'hybrid'], default='full',
                        help='실행 모드 (full: SARIMA+트리 모델, tree_only: 트리 모델만, hybrid: 하이브리드 모델)')
    
    # 모델 선택
    parser.add_argument('--model', type=str, choices=['rf', 'lgbm', 'xgb', 'cat'], default='xgb',
                        help='사용할 트리 모델 (rf: Random Forest, lgbm: LightGBM, xgb: XGBoost, cat: CatBoost)')
    
    # 데이터 경로
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH,
                        help='원본 데이터 파일 경로')
    
    parser.add_argument('--residual_path', type=str, default=DEFAULT_RESIDUAL_PATH,
                        help='잔차 데이터 파일 경로 (tree_only 모드에서 필수)')
    
    # 출력 디렉토리
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='결과 저장 디렉토리')
    
    # 데이터 분할 설정
    parser.add_argument('--train_ratio', type=float, default=DEFAULT_TRAIN_RATIO,
                        help='훈련 데이터 비율')
    
    parser.add_argument('--val_ratio', type=float, default=DEFAULT_VAL_RATIO,
                        help='검증 데이터 비율')
    
    # 특성 엔지니어링 설정
    parser.add_argument('--lag_features', type=int, default=DEFAULT_LAG_FEATURES,
                        help='시차 특성 개수')
    
    parser.add_argument('--no_time_features', action='store_true',
                        help='시간 관련 특성 추가하지 않음')
    
    parser.add_argument('--no_normalize', action='store_true',
                        help='특성 정규화하지 않음')
    
    # 예측 설정
    parser.add_argument('--forecast_horizon', type=int, default=DEFAULT_FORECAST_HORIZON,
                        help='예측 기간(개월)')
    
    # 최적화 설정
    parser.add_argument('--optimize', action='store_true',
                        help='하이퍼파라미터 최적화 수행')
    
    parser.add_argument('--cv_folds', type=int, default=DEFAULT_CV_FOLDS,
                        help='교차 검증 폴드 수')
    
    parser.add_argument('--bayes_iter', type=int, default=DEFAULT_BAYES_ITER,
                        help='베이지안 최적화 반복 횟수')
    
    # 대화형 모드
    parser.add_argument('--interactive', action='store_true',
                        help='대화형 모드로 실행')
    
    args = parser.parse_args()
    
    # tree_only 모드에서 residual_path 필수 확인
    if args.mode == 'tree_only' and args.residual_path is None:
        parser.error("tree_only 모드에서는 --residual_path가 필요합니다.")
    
    return args

def main():
    """
    메인 함수
    """
    # 명령줄 인자 파싱
    args = parse_arguments()
    
    # 대화형 모드
    if args.interactive:
        interactive_mode()
        return
    
    # 출력 디렉토리 및 로깅 설정
    logger = setup_logging(args.output_dir)
    logger.info("해빙 농도 예측 분석 시작")
    logger.info(f"실행 설정: {vars(args)}")
    
    try:
        # 실행 모드에 따른 분기
        if args.mode == 'full':
            # 전체 파이프라인 (SARIMA + 트리 모델)
            run_full_pipeline(vars(args), logger)
        elif args.mode == 'tree_only':
            # 트리 모델만 실행
            run_tree_only_pipeline(vars(args), logger)
        elif args.mode == 'hybrid':
            # 하이브리드 모델 실행
            hybrid_results, hybrid_df, model = run_hybrid_pipeline(vars(args), logger)
            logger.info(f"하이브리드 모델 성능 - RMSE: {hybrid_results['hybrid_rmse']:.6f}, MAE: {hybrid_results['hybrid_mae']:.6f}, R²: {hybrid_results['hybrid_r2']:.6f}")
            
        logger.info("분석 완료")
        
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

    