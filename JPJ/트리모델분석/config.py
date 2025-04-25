"""
설정 파일: 경로, 변수 목록, 기본 파라미터 등 정의
"""
import os
from datetime import datetime

# 기본 경로 설정
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = r"D:\my_projects\natural-science-time-series\JPJ\데이터\v4_전역변수포함.csv"  # 사용할 데이터 파일명 (필요시 수정)
DEFAULT_RESIDUAL_PATH = None  # 기본값은 None, CLI에서 지정 가능

# 결과 저장 디렉토리
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, f"tree_model_analysis_{timestamp}")

# 사용할 특성 목록 (하드코딩)
FEATURES = [
    'sisnthick', 'sithick', 'vxo', 'vyo', '10u', '10v', '2t',
    'sdlwrf', 'sdswrf', 'snswrf', 'tprate', 'skt', 'sp', 'tp','sssn','ao',
]

# 모델 설정
RANDOM_SEED = 42

# 데이터 분할 설정
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
# 테스트 비율은 1 - (TRAIN_RATIO + VAL_RATIO)로 계산

# 시계열 특성 설정
DEFAULT_LAG_FEATURES = 2  # t-1, t-2 시점의 값을 특성으로 추가

# 예측 설정
DEFAULT_FORECAST_HORIZON = 1  # 기본 예측 기간 (월)

# 모델 최적화 설정
DEFAULT_CV_FOLDS = 5  # 교차 검증 폴드 수
DEFAULT_BAYES_ITER = 50  # 베이지안 최적화 반복 횟수
DEFAULT_BAYES_INIT_POINTS = 10  # 베이지안 최적화 초기 탐색 포인트 수

# 하이퍼파라미터 탐색 범위 (베이지안 최적화용)
# RandomForest 하이퍼파라미터 탐색 범위
RF_PARAM_SPACE = {
    'n_estimators': (50, 500),
    'max_depth': (3, 15),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10)
}

# LightGBM 하이퍼파라미터 탐색 범위
LGBM_PARAM_SPACE = {
    'n_estimators': (50, 500),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 15),
    'num_leaves': (20, 150),
    'min_child_samples': (5, 100)
}

# XGBoost 하이퍼파라미터 탐색 범위
XGB_PARAM_SPACE = {
    'n_estimators': (50, 500),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 15),
    'min_child_weight': (1, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0)
}

# CatBoost 하이퍼파라미터 탐색 범위
CAT_PARAM_SPACE = {
    'iterations': (50, 500),
    'learning_rate': (0.01, 0.3),
    'depth': (3, 15),
    'l2_leaf_reg': (1, 10),
    'bagging_temperature': (0, 10)
}

# 모델 별 기본 파라미터
DEFAULT_RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': RANDOM_SEED
}

DEFAULT_LGBM_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 10,
    'random_state': RANDOM_SEED
}

DEFAULT_XGB_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 10,
    'random_state': RANDOM_SEED
}

DEFAULT_CAT_PARAMS = {
    'iterations': 100,
    'learning_rate': 0.1,
    'depth': 10,
    'random_seed': RANDOM_SEED
}