# 이상치 처리 및 저장
import pandas as pd
import numpy as np
from scipy import stats
import os
from datetime import datetime

# 데이터 로드
df = pd.read_csv(r"D:\my_projects\natural-science-time-series\JPJ\데이터\v2_염분_수온제거.csv")

# 시간 열을 datetime 형식으로 변환
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# 결과 폴더 생성
result_folder = r"D:\my_projects\natural-science-time-series\JPJ\전처리\결과"
os.makedirs(result_folder, exist_ok=True)
print(f"결과 폴더 생성: {result_folder}")

# 원본 데이터 백업
df_original = df.copy()

# 처리할 컬럼 확인
available_columns = list(df.columns)
print(f"처리할 컬럼: {', '.join(available_columns)}")

# 이상치 처리 로그 초기화
log_file = os.path.join(result_folder, f"outlier_treatment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"이상치 처리 로그 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"원본 데이터 행 수: {len(df)}\n\n")

# =========== 1. 각 변수별 이상치 처리 함수 ===========

def treat_outliers_zscore(df, column, threshold=3.0, log_file=None):
    """Z-score 방식으로 이상치 처리"""
    if column not in df.columns:
        return df, 0
    
    # 결측값이 아닌 데이터에 대해서만 Z-score 계산
    mask = ~df[column].isna()
    data = df.loc[mask, column]
    z = np.abs(stats.zscore(data))
    
    # 이상치 플래그
    outliers = z > threshold
    outlier_count = np.sum(outliers)
    
    if outlier_count > 0:
        # 이상치 인덱스
        outlier_indices = data.index[outliers]
        
        # 이상치 처리 - 월별 중앙값으로 대체
        for idx in outlier_indices:
            month = idx.month
            month_median = df[df.index.month == month][column].median()
            df.loc[idx, column] = month_median
        
        # 로그 기록
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{column} Z-score 이상치 처리: {outlier_count}개 ({outlier_count/len(df)*100:.2f}%)\n")
                f.write(f"  - 임계값: Z-score > {threshold}\n")
                f.write(f"  - 처리 방법: 월별 중앙값으로 대체\n")
    
    return df, outlier_count

def treat_outliers_iqr(df, column, factor=1.5, log_file=None):
    """IQR 방식으로 이상치 처리"""
    if column not in df.columns:
        return df, 0
    
    # 결측값 제외
    data = df[column].dropna()
    
    # IQR 계산
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    # 이상치 경계 설정
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # 이상치 플래그
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_count = np.sum(outliers)
    
    if outlier_count > 0:
        # 이상치 처리 - 월별 중앙값으로 대체
        for idx in df[outliers].index:
            month = idx.month
            month_median = df[df.index.month == month][column].median()
            df.loc[idx, column] = month_median
        
        # 로그 기록
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{column} IQR 이상치 처리: {outlier_count}개 ({outlier_count/len(df)*100:.2f}%)\n")
                f.write(f"  - 임계값: < {lower_bound:.4f} 또는 > {upper_bound:.4f} (factor={factor})\n")
                f.write(f"  - 처리 방법: 월별 중앙값으로 대체\n")
    
    return df, outlier_count

def treat_seasonal_outliers(df, column, log_file=None):
    """계절성을 고려한 이상치 처리"""
    if column not in df.columns:
        return df, 0
    
    total_outliers = 0
    
    # 월별로 이상치 처리
    for month in range(1, 13):
        month_data = df[df.index.month == month][column].dropna()
        
        if len(month_data) < 5:  # 데이터가 너무 적으면 처리하지 않음
            continue
        
        # 월별 통계 계산
        month_mean = month_data.mean()
        month_std = month_data.std()
        
        # 이상치 임계값 (월별 평균 ± 3 * 표준편차)
        lower_bound = month_mean - 3 * month_std
        upper_bound = month_mean + 3 * month_std
        
        # 이상치 플래그
        outliers = (df[column][df.index.month == month] < lower_bound) | \
                   (df[column][df.index.month == month] > upper_bound)
        outlier_count = np.sum(outliers)
        total_outliers += outlier_count
        
        if outlier_count > 0:
            # 이상치 처리 - 월별 중앙값으로 대체
            outlier_indices = df[df.index.month == month].loc[outliers].index
            month_median = month_data.median()
            
            for idx in outlier_indices:
                df.loc[idx, column] = month_median
    
    # 로그 기록
    if total_outliers > 0 and log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{column} 계절성 고려 이상치 처리: {total_outliers}개 ({total_outliers/len(df)*100:.2f}%)\n")
            f.write(f"  - 임계값: 월별 평균 ± 3 * 월별 표준편차\n")
            f.write(f"  - 처리 방법: 월별 중앙값으로 대체\n")
    
    return df, total_outliers

# =========== 2. 각 변수별 맞춤 이상치 처리 ===========

# 처리 결과 요약
summary = {}

# vxo, vyo (해류) 처리
if 'vxo' in df.columns:
    df, outlier_count = treat_outliers_zscore(df, 'vxo', threshold=3.0, log_file=log_file)
    summary['vxo'] = outlier_count

if 'vyo' in df.columns:
    df, outlier_count = treat_outliers_zscore(df, 'vyo', threshold=3.0, log_file=log_file)
    summary['vyo'] = outlier_count

# 10u, 10v (바람) 처리
if '10u' in df.columns:
    df, outlier_count = treat_outliers_zscore(df, '10u', threshold=3.0, log_file=log_file)
    summary['10u'] = outlier_count

if '10v' in df.columns:
    df, outlier_count = treat_outliers_zscore(df, '10v', threshold=3.0, log_file=log_file)
    summary['10v'] = outlier_count

# 2t (기온) 처리 - 계절성 고려
if '2t' in df.columns:
    df, outlier_count = treat_seasonal_outliers(df, '2t', log_file=log_file)
    summary['2t'] = outlier_count

# skt (표면 온도) 처리 - 계절성 고려
if 'skt' in df.columns:
    df, outlier_count = treat_seasonal_outliers(df, 'skt', log_file=log_file)
    summary['skt'] = outlier_count

# sp (기압) 처리
if 'sp' in df.columns:
    df, outlier_count = treat_outliers_iqr(df, 'sp', factor=1.8, log_file=log_file)  # 약간 더 관대한 임계값
    summary['sp'] = outlier_count

# sithick (해빙 두께) 처리 - 계절성 고려
if 'sithick' in df.columns:
    df, outlier_count = treat_seasonal_outliers(df, 'sithick', log_file=log_file)
    summary['sithick'] = outlier_count

# sisnthick (눈 두께) 처리 - 계절성 고려
if 'sisnthick' in df.columns:
    df, outlier_count = treat_seasonal_outliers(df, 'sisnthick', log_file=log_file)
    summary['sisnthick'] = outlier_count

# sic (해빙 농도) 처리 - 계절성 고려, 물리적 임계값 적용
if 'sic' in df.columns:
    # 우선 계절성 고려 처리
    df, outlier_count_seasonal = treat_seasonal_outliers(df, 'sic', log_file=log_file)
    
    # 물리적 임계값 적용 (최대값 1.0, 최소값 0.0)
    invalid_mask = (df['sic'] < 0.0) | (df['sic'] > 1.0)
    invalid_count = np.sum(invalid_mask)
    
    if invalid_count > 0:
        df.loc[df['sic'] < 0.0, 'sic'] = 0.0
        df.loc[df['sic'] > 1.0, 'sic'] = 1.0
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"sic 물리적 제약 적용: {invalid_count}개 ({invalid_count/len(df)*100:.2f}%)\n")
            f.write(f"  - 임계값: < 0.0 또는 > 1.0\n")
            f.write(f"  - 처리 방법: 0.0/1.0으로 클리핑\n")
    
    summary['sic'] = outlier_count_seasonal + invalid_count

# 복사 관련 변수 처리 (sdlwrf, sdswrf, snswrf)
for var in ['sdlwrf', 'sdswrf', 'snswrf']:
    if var in df.columns:
        df, outlier_count = treat_seasonal_outliers(df, var, log_file=log_file)
        summary[var] = outlier_count

# 강수 관련 변수 처리 (tprate, tp)
for var in ['tprate', 'tp']:
    if var in df.columns:
        # 음수 값은 0으로 대체
        neg_mask = df[var] < 0
        neg_count = np.sum(neg_mask)
        
        if neg_count > 0:
            df.loc[neg_mask, var] = 0
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{var} 음수 값 처리: {neg_count}개 ({neg_count/len(df)*100:.2f}%)\n")
                f.write(f"  - 처리 방법: 0으로 대체\n")
        
        # 극단적 이상치 처리
        df, outlier_count = treat_outliers_zscore(df, var, threshold=3.5, log_file=log_file)
        summary[var] = neg_count + outlier_count

# =========== 3. 처리 결과 요약 및 저장 ===========

# 변경된 데이터 수 계산
changed_count = sum(summary.values())
with open(log_file, 'a', encoding='utf-8') as f:
    f.write(f"\n총 이상치 처리 개수: {changed_count}개 ({changed_count/(len(df)*len(available_columns))*100:.2f}%)\n")
    f.write("\n변수별 이상치 처리 요약:\n")
    for var, count in summary.items():
        f.write(f"  - {var}: {count}개 ({count/len(df)*100:.2f}%)\n")

# 처리된 데이터 저장
output_file = os.path.join(result_folder, f"preprocessed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
df.to_csv(output_file)
print(f"전처리된 데이터 저장 완료: {output_file}")

# 처리 전후 비교 정보 저장
comparison_file = os.path.join(result_folder, f"before_after_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
with open(comparison_file, 'w', encoding='utf-8') as f:
    f.write(f"처리 전후 통계 비교 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    for column in available_columns:
        if column in df.columns:
            f.write(f"=== {column} 변수 ===\n")
            f.write("처리 전 통계:\n")
            f.write(f"{df_original[column].describe().to_string()}\n\n")
            f.write("처리 후 통계:\n")
            f.write(f"{df[column].describe().to_string()}\n\n")
            
            # 변경된 최대/최소값 확인
            orig_min, orig_max = df_original[column].min(), df_original[column].max()
            new_min, new_max = df[column].min(), df[column].max()
            
            if orig_min != new_min or orig_max != new_max:
                f.write(f"범위 변경: [{orig_min}, {orig_max}] -> [{new_min}, {new_max}]\n\n")
            else:
                f.write("범위 변경 없음\n\n")

print(f"처리 전후 통계 정보 저장 완료: {comparison_file}")
print(f"이상치 처리 로그 저장 완료: {log_file}")
print("전처리 완료!")