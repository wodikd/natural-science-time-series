# 1단계: 데이터 로드 및 기본 탐색
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# 한글 폰트 설정
import matplotlib as mpl
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False


# 데이터 로드 (파일 경로를 실제 CSV 파일 위치로 변경하세요)
df = pd.read_csv(r"D:\my_projects\natural-science-time-series\JPJ\데이터\라스트마스크cmems.csv")

# 시간 열을 datetime 형식으로 변환
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# 데이터셋 기본 정보 확인
print("데이터셋 기본 정보:")
print(f"행 수: {df.shape[0]}, 열 수: {df.shape[1]}")
print("\n첫 5개 행:")
print(df.head())

# 기술 통계량 확인
print("\n기술 통계량:")
print(df.describe())

# 결측치 확인
print("\n결측치 개수:")
print(df.isna().sum())

# 2단계: 각 변수별 이상치 탐지 및 시각화

# 주요 변수 리스트
key_variables = ['thetao', 'so', 'sithick']

# 1. 시계열 플롯 생성
plt.figure(figsize=(15, 20))
for i, var in enumerate(key_variables, 1):
    plt.subplot(len(key_variables), 1, i)
    plt.plot(df.index, df[var])
    plt.title(f'{var} 시계열 (1991-2023)')
    plt.grid(True)
plt.tight_layout()
plt.savefig('time_series_plots.png')
plt.close()

# 2. 월별 박스플롯 생성
plt.figure(figsize=(15, 20))
for i, var in enumerate(key_variables, 1):
    plt.subplot(len(key_variables), 1, i)
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly.index.month
    sns.boxplot(x='month', y=var, data=df_monthly)
    plt.title(f'{var}의 월별 분포')
    plt.xlabel('월')
    plt.ylabel(var)
plt.tight_layout()
plt.savefig('monthly_boxplots.png')
plt.close()

# 3. Z-score를 사용한 이상치 탐지
z_scores = {}
z_threshold = 3.0  # 일반적으로 사용되는 임계값

plt.figure(figsize=(15, 20))
for i, var in enumerate(key_variables, 1):
    plt.subplot(len(key_variables), 1, i)
    
    # Z-score 계산
    z = np.abs(stats.zscore(df[var], nan_policy='omit'))
    z_scores[var] = z
    
    # 이상치 플래그
    outliers = z > z_threshold
    
    # 이상치 시각화
    plt.scatter(df.index, df[var], c=['red' if x else 'blue' for x in outliers], alpha=0.5)
    plt.title(f'{var}의 Z-score 이상치 (빨간색)')
    plt.grid(True)
    
    # 이상치 개수 출력
    outlier_count = np.sum(outliers)
    print(f"{var}의 Z-score 이상치 개수: {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
plt.tight_layout()
plt.savefig('z_score_outliers.png')
plt.close()

# 4. IQR 방법을 사용한 이상치 탐지
plt.figure(figsize=(15, 20))
for i, var in enumerate(key_variables, 1):
    plt.subplot(len(key_variables), 1, i)
    
    # IQR 계산
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    
    # 이상치 경계 설정
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 이상치 플래그
    outliers = (df[var] < lower_bound) | (df[var] > upper_bound)
    
    # 이상치 시각화
    plt.scatter(df.index, df[var], c=['red' if x else 'blue' for x in outliers], alpha=0.5)
    plt.axhline(y=lower_bound, color='green', linestyle='--', label='Lower Bound')
    plt.axhline(y=upper_bound, color='green', linestyle='--', label='Upper Bound')
    plt.title(f'{var}의 IQR 이상치 (빨간색)')
    plt.grid(True)
    plt.legend()
    
    # 이상치 개수 출력
    outlier_count = np.sum(outliers)
    print(f"{var}의 IQR 이상치 개수: {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
plt.tight_layout()
plt.savefig('iqr_outliers.png')
plt.close()

# 5. 특정 변수 집중 분석
# thetao 분석 (여름철 급등 스파이크 확인)
df['year'] = df.index.year
df['month'] = df.index.month

# 여름철 데이터 추출 (6-8월)
summer_df = df[(df['month'] >= 6) & (df['month'] <= 8)]

plt.figure(figsize=(15, 8))
# 년도별로 색상을 다르게 하여 시각화
for year in sorted(summer_df['year'].unique()):
    year_data = summer_df[summer_df['year'] == year]
    plt.scatter(year_data.index, year_data['thetao'], label=str(year) if year % 5 == 0 else None)

plt.title('여름철 thetao 분포 (1991-2023)')
plt.axhline(y=1.5, color='red', linestyle='--', label='임계값 1.5℃')
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('summer_thetao.png')
plt.close()

# so 분석 (2000-2003 flat 구간과 2021-2022 급락 구간)
plt.figure(figsize=(15, 8))
plt.plot(df.index, df['so'])
plt.axvspan(pd.Timestamp('2000-01-01'), pd.Timestamp('2003-12-31'), alpha=0.3, color='yellow', label='2000-2003 의심 구간')
plt.axvspan(pd.Timestamp('2021-01-01'), pd.Timestamp('2022-12-31'), alpha=0.3, color='red', label='2021-2022 의심 구간')
plt.title('so (염분) 시계열 및 의심 구간')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('so_suspicious_periods.png')
plt.close()

# skt 분석 (230K 이하 극단값)
plt.figure(figsize=(15, 8))
plt.scatter(df.index, df['skt'], c=['red' if x < 230 else 'blue' for x in df['skt']], alpha=0.5)
plt.axhline(y=230, color='red', linestyle='--', label='임계값 230K')
plt.title('skt (표면 온도) 및 230K 이하 극단값')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('skt_extreme_values.png')
plt.close()

print("\n분석 완료! 이미지 파일을 확인하세요.")