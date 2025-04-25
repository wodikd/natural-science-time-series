# 해빙 농도(SIC) 예측 분석 도구

이 프로젝트는 보퍼트해 해빙 면적(SIC) 데이터를 분석하고 예측하기 위한 파이프라인을 제공합니다. SARIMA 모델과 트리 기반 모델을 결합한 하이브리드 접근법을 통해 예측 정확도를 높이고 다양한 환경 변수의 영향력을 파악할 수 있습니다.

## 목차

- [배경](#배경)
- [설치 및 요구사항](#설치-및-요구사항)
- [사용법](#사용법)
  - [대화형 모드](#대화형-모드)
  - [명령줄 모드](#명령줄-모드)
  - [주요 인자](#주요-인자)
- [실행 모드](#실행-모드)
  - [전체 파이프라인 (SARIMA + 트리 모델)](#전체-파이프라인-sarima--트리-모델)
  - [트리 모델만 실행](#트리-모델만-실행)
  - [하이브리드 모델](#하이브리드-모델)
- [프로젝트 구조](#프로젝트-구조)
- [데이터 전처리](#데이터-전처리)
  - [시간 특성 추가](#시간-특성-추가)
  - [시차 변수 추가](#시차-변수-추가)
  - [데이터 분할](#데이터-분할)
  - [특성 정규화](#특성-정규화)
- [하이퍼파라미터 최적화](#하이퍼파라미터-최적화)
  - [베이지안 최적화](#베이지안-최적화)
  - [탐색 공간 설정](#탐색-공간-설정)
- [모델 평가 지표](#모델-평가-지표)
- [결과 해석](#결과-해석)
  - [SARIMA 분석 결과](#sarima-분석-결과)
  - [트리 모델 결과](#트리-모델-결과)
  - [하이브리드 모델 결과](#하이브리드-모델-결과)
  - [특성 중요도 분석](#특성-중요도-분석)
  - [SHAP 값 분석](#shap-값-분석)
- [고급 옵션](#고급-옵션)
- [문제 해결](#문제-해결)

## 배경

해빙 농도(Sea Ice Concentration, SIC)는 기후 변화의 중요한 지표로, 그 예측은 기후 연구와 해양 활동 계획에 매우 중요합니다. 이 프로젝트는 시계열 분석 기법과 머신러닝 접근법을 결합하여 보다 정확한 해빙 농도 예측을 제공합니다.

전통적인 시계열 모델인 SARIMA(Seasonal AutoRegressive Integrated Moving Average)는 계절성과 추세를 잘 포착하지만, 외부 변수의 영향을 고려하는 데 제한적입니다. 반면, 트리 기반 머신러닝 모델은 다양한 특성 간의 복잡한 관계를 학습할 수 있지만, 시계열 데이터의 계절성이나 추세를 직접적으로 모델링하는 데 어려움이 있습니다.

이 도구는 두 접근법의 장점을 결합한 하이브리드 모델을 제공합니다:
1. SARIMA 모델로 시계열의 계절성 및 추세를 학습
2. 트리 기반 모델(RandomForest, LightGBM, XGBoost, CatBoost)을 사용하여 SARIMA의 잔차(residuals) 예측
3. 두 예측을 결합하여 최종 예측값 생성

이러한 접근법은 예측 정확도를 높이고, 환경 변수(온도, 바람, 압력 등)가 해빙 농도에 미치는 영향을 정량적으로 분석할 수 있게 합니다.

## 설치 및 요구사항

### 필수 패키지

```bash
pip install -r requirements.txt
```

requirements.txt 내용:
```
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
pmdarima
xgboost
lightgbm
catboost
shap
scikit-optimize
joblib
```

## 사용법

### 대화형 모드

가장 간단한 방법은 대화형 모드를 사용하는 것입니다:

```bash
python run_analysis.py --interactive
```

이 모드에서는 단계별로 안내에 따라 설정을 선택할 수 있습니다.

### 명령줄 모드

모든 옵션을 명령줄 인자로 지정할 수도 있습니다:

```bash
# 전체 파이프라인 실행 (SARIMA + 트리 모델)
python run_analysis.py --mode full --model xgb --data_path beaufort_sea_ice_data.csv --output_dir results_xgb --forecast_horizon 1 --optimize

# 기존 잔차로 트리 모델만 실행
python run_analysis.py --mode tree_only --model lgbm --data_path beaufort_sea_ice_data.csv --residual_path sarima_residuals.csv --output_dir results_lgbm --forecast_horizon 6

# 하이브리드 모델 실행
python run_analysis.py --mode hybrid --model xgb --data_path beaufort_sea_ice_data.csv --output_dir hybrid_results --optimize
```

### 주요 인자

- `--mode`: 실행 모드 (`full`: SARIMA+트리 모델, `tree_only`: 트리 모델만, `hybrid`: 하이브리드 모델)
- `--model`: 사용할 트리 모델 (`rf`: Random Forest, `lgbm`: LightGBM, `xgb`: XGBoost, `cat`: CatBoost)
- `--data_path`: 원본 데이터 파일 경로
- `--residual_path`: 잔차 데이터 파일 경로 (`tree_only` 모드에서 필수)
- `--output_dir`: 결과 저장 디렉토리
- `--forecast_horizon`: 예측 기간(개월)
- `--optimize`: 하이퍼파라미터 최적화 수행 여부
- `--train_ratio`: 훈련 데이터 비율 (기본값: 0.7)
- `--val_ratio`: 검증 데이터 비율 (기본값: 0.15)
- `--lag_features`: 시차 특성 개수 (기본값: 2)
- `--cv_folds`: 교차 검증 폴드 수 (기본값: 5)
- `--bayes_iter`: 베이지안 최적화 반복 횟수 (기본값: 50)

## 실행 모드

### 전체 파이프라인 (SARIMA + 트리 모델)

`full` 모드에서는 다음 단계를 수행합니다:

1. 원본 데이터 로드
2. SARIMA 모델 학습 및 잔차 생성
3. 잔차 시각화
4. 특성 엔지니어링 (시간 특성, 시차 변수 추가)
5. 데이터 분할 (훈련/검증/테스트)
6. 트리 모델 학습 및 최적화
7. 모델 평가 및 결과 저장
8. 특성 중요도 및 SHAP 값 분석

### 트리 모델만 실행

`tree_only` 모드에서는 기존에 생성된 SARIMA 잔차를 사용하여 트리 모델만 실행합니다:

1. 원본 데이터 및 잔차 데이터 로드
2. 데이터 병합 및 특성 엔지니어링
3. 데이터 분할 및 정규화
4. 트리 모델 학습 및 최적화
5. 모델 평가 및 결과 저장
6. 특성 중요도 및 SHAP 값 분석

### 하이브리드 모델

`hybrid` 모드에서는 SARIMA와 트리 모델을 결합한 하이브리드 모델을 실행합니다:

1. 원본 데이터 로드
2. SARIMA 모델 학습 및 전체 예측값 생성 (인샘플 + 아웃샘플)
3. 데이터 분할 일치시키기 (SARIMA와 트리 모델 간)
4. 트리 모델 학습 및 잔차 예측
5. 두 예측을 결합하여 최종 예측 생성 (SARIMA 예측 + 트리 모델 잔차 예측)
6. 단일 SARIMA 모델과 하이브리드 모델의 성능 비교

## 프로젝트 구조

```
.
├── config.py                 # 설정 파일 (경로, 변수 목록, 파라미터 등)
├── data_utils.py             # 데이터 관련 유틸리티
├── sarima_utils.py           # SARIMA 모델 관련 유틸리티
├── model_utils.py            # 트리 모델 관련 유틸리티
├── visualization.py          # 시각화 유틸리티
├── run_analysis.py           # 메인 실행 스크립트
└── results_[timestamp]/      # 결과 디렉토리
    ├── analysis_log.log      # 로그 파일
    ├── sarima_residuals.csv  # SARIMA 잔차
    ├── sarima_*.png          # SARIMA 결과 시각화
    ├── *_model.pkl           # 학습된 모델
    ├── *_feature_importance.*# 특성 중요도
    ├── *_shap_*.csv          # SHAP 값
    ├── *_shap_*.png          # SHAP 시각화
    ├── hybrid_*.png          # 하이브리드 모델 시각화
    └── *_evaluation.csv      # 평가 결과
```

## 데이터 전처리

### 시간 특성 추가

시간 관련 특성은 시계열 데이터의 계절성을 포착하는 데 중요합니다. 이 도구는 다음과 같은 시간 특성을 추가합니다:

- 월 주기성 인코딩: `month_sin = sin(2π × month / 12)`, `month_cos = cos(2π × month / 12)`
- 연도: 선형 추세 포착을 위한 연도 정보

```python
def add_time_features(df):
    # 월 사이클릭 인코딩
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # 연도 정보 추가
    df['year'] = df.index.year
    
    return df
```

### 시차 변수 추가

시차 변수(lag features)는 이전 시점의 값을 현재 예측에 사용하기 위한 특성입니다. 기본적으로 t-1, t-2 시점의 값을 사용합니다:

```python
def add_lag_features(df, target_col='residual', lags=2):
    for i in range(1, lags + 1):
        df[f'{target_col}_lag_{i}'] = df[target_col].shift(i)
    
    # NaN 값을 가진 행 제거
    df = df.dropna()
    
    return df
```

시차 변수의 개수는 `--lag_features` 인자로 조절할 수 있습니다.

### 데이터 분할

데이터는 시간 순서대로 훈련, 검증, 테스트 세트로 분할됩니다:

- 훈련 세트: 처음 70% (기본값, `--train_ratio`로 조절 가능)
- 검증 세트: 다음 15% (기본값, `--val_ratio`로 조절 가능)
- 테스트 세트: 나머지 15%

하이브리드 모드에서는 SARIMA 모델과 트리 모델 간의 데이터 분할이 일치되도록 특별히 처리됩니다.

### 특성 정규화

수치형 특성은 StandardScaler를 사용하여 정규화됩니다:

```python
def normalize_features(X_train, X_val, X_test):
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
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
```

## 하이퍼파라미터 최적화

### 베이지안 최적화

하이퍼파라미터 최적화는 `--optimize` 옵션을 사용하여 활성화할 수 있습니다. 이 도구는 베이지안 최적화(Bayesian Optimization)를 사용하여 모델의 하이퍼파라미터를 튜닝합니다:

```python
def optimize_hyperparams(model_type, X_train, y_train, X_val, y_val, 
                        n_iter=50, cv_folds=5, n_initial_points=10):
    # 파라미터 공간 설정
    # ...
    
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
    opt.fit(X_train, y_train)
    
    # 최적 파라미터 및 모델 반환
    # ...
```

베이지안 최적화는 일반적인 그리드 서치나 랜덤 서치보다 효율적으로 최적의 하이퍼파라미터를 찾을 수 있습니다.

### 탐색 공간 설정

각 모델 유형별로 다음과 같은 하이퍼파라미터 탐색 공간이 설정되어 있습니다:

**RandomForest:**
```python
RF_PARAM_SPACE = {
    'n_estimators': (50, 500),
    'max_depth': (3, 15),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10)
}
```

**LightGBM:**
```python
LGBM_PARAM_SPACE = {
    'n_estimators': (50, 500),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 15),
    'num_leaves': (20, 150),
    'min_child_samples': (5, 100)
}
```

**XGBoost:**
```python
XGB_PARAM_SPACE = {
    'n_estimators': (50, 500),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 15),
    'min_child_weight': (1, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0)
}
```

**CatBoost:**
```python
CAT_PARAM_SPACE = {
    'iterations': (50, 500),
    'learning_rate': (0.01, 0.3),
    'depth': (3, 15),
    'l2_leaf_reg': (1, 10),
    'bagging_temperature': (0, 10)
}
```

## 모델 평가 지표

모델 평가에는 다음과 같은 지표가 사용됩니다:

- **RMSE(Root Mean Squared Error)**: 예측 오차의 제곱평균에 루트를 씌운 값으로, 작을수록 좋습니다. 큰 오차에 더 민감합니다.
  ```
  RMSE = sqrt(mean((y_true - y_pred)²))
  ```

- **MAE(Mean Absolute Error)**: 평균 절대 오차로, 작을수록 좋습니다. 모든 오차를 동일하게 취급합니다.
  ```
  MAE = mean(|y_true - y_pred|)
  ```

- **R²(결정 계수)**: 모델이 데이터 변동성을 얼마나 설명하는지 나타내며, 1에 가까울수록 좋습니다.
  ```
  R² = 1 - sum((y_true - y_pred)²) / sum((y_true - mean(y_true))²)
  ```

하이브리드 모델 평가 시에는 다음 세 가지 성능을 모두 측정합니다:
1. SARIMA 모델의 성능 (실제 SIC vs SARIMA 예측)
2. 트리 모델의 성능 (SARIMA 잔차 vs 트리 모델 예측)
3. 하이브리드 모델의 성능 (실제 SIC vs (SARIMA 예측 + 트리 모델 예측))

## 결과 해석

분석 결과는 지정된 `output_dir`에 저장됩니다. 각 결과 파일과 그래프를 해석하는 방법은 다음과 같습니다.

### SARIMA 분석 결과

#### sarima_model_summary.txt
- SARIMA 모델의 상세 정보와 파라미터를 제공합니다.
- AIC, BIC, 로그 가능도 등의 지표를 통해 모델 적합도를 평가할 수 있습니다.
- p-값이 0.05 미만인 계수는 통계적으로 유의미합니다.

#### sarima_residuals.png
- 상단 그래프: 시간에 따른 잔차 분포를 보여줍니다. 잔차가 시간에 따라 패턴 없이 무작위로 분포되어야 합니다.
- 하단 그래프: 잔차의 분포를 보여줍니다. 정규 분포에 가까울수록 좋습니다.

#### sarima_residuals_acf_pacf.png
- ACF(자기상관함수) 및 PACF(편자기상관함수) 그래프입니다.
- 파란색 띠를 벗어나는 스파이크가 거의 없어야 합니다. 여전히 패턴이 남아있다면 SARIMA 모델이 모든 시계열 패턴을 포착하지 못했다는 의미입니다.

### 트리 모델 결과

#### final_evaluation.csv
- 모델의 최종 평가 지표를 제공합니다.
- RMSE, MAE, R² 값을 통해 모델 성능을 평가할 수 있습니다.

#### [model]_actual_vs_predicted.png
- 상단 그래프: 실제 잔차값과 예측값을 시간에 따라 보여줍니다. 두 선이 가까울수록 예측이 정확합니다.
- 하단 그래프: 실제값과 예측값의 산점도입니다. 45도 대각선(빨간 점선)에 가까울수록 예측이 정확합니다.

#### [model]_error_distribution.png
- 예측 오차의 분포를 보여줍니다. 평균이 0에 가깝고 분포가 좁을수록 좋습니다.

### 하이브리드 모델 결과

#### hybrid_evaluation.csv
- SARIMA 모델, 트리 모델, 하이브리드 모델의 성능 지표를 비교합니다.
- 하이브리드 모델이 SARIMA 단독 모델보다 낮은 RMSE/MAE와 높은 R² 값을 가질 경우, 하이브리드 접근법이 성공적이라고 볼 수 있습니다.

#### hybrid_[model]_comparison.png
- 상단 그래프: 실제 SIC 값, SARIMA 예측값, 하이브리드 모델 예측값을 시간에 따라 비교합니다.
- 하단 그래프: SARIMA 모델과 하이브리드 모델의 예측 오차를 비교합니다.

#### hybrid_[model]_metrics_comparison.png
- SARIMA 모델과 하이브리드 모델의 RMSE, MAE, R² 값을 막대 그래프로 비교합니다.
- 이를 통해 하이브리드 모델이 SARIMA 모델에 비해 얼마나 성능이 향상되었는지 직관적으로 확인할 수 있습니다.

### 특성 중요도 분석

#### [model]_feature_importance.csv
- 각 특성의 중요도를 수치로 제공합니다.
- 값이 클수록 해당 특성이 예측에 더 큰 영향을 미쳤다는 의미입니다.

#### [model]_feature_importance.png
- 특성 중요도를 시각적으로 보여줍니다.
- 상위 특성들이 모델의 예측에 가장 큰 영향을 미치며, 이 특성들을 중점적으로 분석하는 것이 좋습니다.

### SHAP 값 분석

SHAP(SHapley Additive exPlanations) 값은 각 특성이 개별 예측에 미치는 영향을 보여줍니다.

#### [model]_shap_summary.png
- SHAP 요약 그래프입니다.
- 각 점은 하나의 샘플에 대한 하나의 특성의 SHAP 값을 나타냅니다.
- 색상: 특성 값의 크기를 나타냅니다 (빨간색: 높음, 파란색: 낮음).
- x축: SHAP 값으로, 양수는 예측값을 증가시키는 방향, 음수는 감소시키는 방향입니다.
- 특성이 위에서부터 중요도 순으로 정렬됩니다.

#### [model]_shap_dependence_[feature].png
- 개별 특성에 대한 의존성 그래프입니다.
- x축: 특성 값
- y축: SHAP 값
- 이 그래프를 통해 특성 값이 변할 때 예측에 미치는 영향의 패턴을 확인할 수 있습니다.
- 예: 온도(2t) 특성에 대한 그래프에서 상승 추세가 보인다면, 온도가 높을수록 해빙 면적 잔차에 양의 영향을 미친다는 의미입니다.

## 고급 옵션

### 예측 기간(forecast_horizon) 선택
- 단기 예측(1-3개월): 일반적으로 가장 정확한 예측을 제공합니다.
- 중기 예측(6-12개월): 계절 주기를 포함한 예측이 가능합니다.
- 장기 예측(24개월 이상): 장기 추세를 파악하기 위한 목적으로 사용할 수 있지만, 정확도는 낮아질 수 있습니다.

### 하이퍼파라미터 최적화
`--optimize` 옵션을 사용하면 베이지안 최적화를 통해 모델의 하이퍼파라미터를 튜닝합니다. 이는 더 나은 성능을 제공할 수 있지만, 계산 시간이 길어집니다.

```bash
python run_analysis.py --mode hybrid --model xgb --optimize --bayes_iter 50
```

최적화 반복 횟수는 `--bayes_iter` 인자로 조절할 수 있습니다. 기본값은 50회입니다.

## 문제 해결

### 패키지 임포트 오류
다음과 같은 오류가 발생할 경우:
```
ImportError: cannot import name 'BayesSearchCV' from 'skopt'
```

scikit-optimize 패키지를 설치하세요:
```
pip install scikit-optimize
```

### 메모리 오류
대규모 데이터셋에서 메모리 오류가 발생할 경우:
- `--lag_features` 값을 줄여 시차 변수 개수를 줄입니다.
- 모델의 복잡도를 줄입니다 (예: max_depth 감소).

### 데이터 분할 문제
SARIMA 모델과 트리 모델의 테스트 세트가 일치하지 않는 경우:
- `hybrid` 모드를 사용하면 두 모델의 데이터 분할이 자동으로 일치됩니다.
- 시차 변수 추가로 인한 데이터 손실을 고려하여 데이터 분할을 조정합니다.

### 데이터 파일 경로 오류
데이터 파일 경로가 올바른지 확인하세요. 상대 경로보다는 절대 경로를 사용하는 것이 더 안정적입니다.