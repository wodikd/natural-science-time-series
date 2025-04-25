# 이상치 탐지 시각화 (추가 변수)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime

# 한글 폰트 설정
import matplotlib as mpl
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv(r"D:\my_projects\natural-science-time-series\JPJ\데이터\라스트마스크cmems.csv")

# 시간 열을 datetime 형식으로 변환
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# 결과 폴더 생성
result_folder = r"D:\my_projects\natural-science-time-series\JPJ\전처리\테스트"
os.makedirs(result_folder, exist_ok=True)
print(f"결과 폴더 생성: {result_folder}")

# 분석하려는 모든 변수 리스트
all_variables = ['sisnthick', 'sithick', 'so', 'thetao', 'vxo', 'vyo', '10u', '10v', '2t', 
               'sdlwrf', 'sdswrf', 'snswrf', 'tprate', 'sic', 'skt', 'sp', 'tp']

# 실제 데이터에 존재하는 변수만 필터링
available_variables = [var for var in all_variables if var in df.columns]
print(f"분석 가능한 변수: {', '.join(available_variables)}")

# 변수가 없을 경우 처리
if not available_variables:
    print("분석할 수 있는 변수가 없습니다. 데이터를 확인해주세요.")
    exit()

# 1. 시계열 플롯 생성
try:
    plt.figure(figsize=(15, len(available_variables) * 2.5))
    for i, var in enumerate(available_variables, 1):
        plt.subplot(len(available_variables), 1, i)
        plt.plot(df.index, df[var])
        plt.title(f'{var} 시계열 (1991-2023)')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, 'all_time_series_plots.png'))
    plt.close()
    print("1. 시계열 플롯 생성 완료")
except Exception as e:
    print(f"시계열 플롯 생성 중 오류 발생: {e}")

# 2. 월별 박스플롯 생성
for var in available_variables:
    try:
        plt.figure(figsize=(15, 8))
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly.index.month
        sns.boxplot(x='month', y=var, data=df_monthly)
        plt.title(f'{var}의 월별 분포')
        plt.xlabel('월')
        plt.ylabel(var)
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, f'{var}_monthly_boxplot.png'))
        plt.close()
    except Exception as e:
        print(f"{var} 월별 박스플롯 생성 중 오류 발생: {e}")
print("2. 월별 박스플롯 생성 완료")

# 3. Z-score를 사용한 이상치 탐지
z_threshold = 3.0  # 일반적으로 사용되는 임계값
for var in available_variables:
    try:
        # 결측값 확인
        if df[var].isna().all():
            print(f"{var}는 모든 값이 결측값입니다. Z-score 이상치 탐지를 건너뜁니다.")
            continue
            
        plt.figure(figsize=(15, 8))
        
        # Z-score 계산
        z = np.abs(stats.zscore(df[var], nan_policy='omit'))
        
        # 이상치 플래그
        outliers = z > z_threshold
        
        # 이상치 시각화
        plt.scatter(df.index, df[var], c=['red' if x else 'blue' for x in outliers], alpha=0.5)
        plt.title(f'{var}의 Z-score 이상치 (빨간색)')
        plt.grid(True)
        
        # 이상치 개수 출력
        outlier_count = np.sum(outliers)
        outlier_percent = outlier_count/len(df)*100
        plt.figtext(0.5, 0.01, f"이상치 개수: {outlier_count} ({outlier_percent:.2f}%)", 
                    ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, f'{var}_z_score_outliers.png'))
        plt.close()
    except Exception as e:
        print(f"{var} Z-score 이상치 탐지 중 오류 발생: {e}")
print("3. Z-score 이상치 탐지 완료")

# 4. IQR 방법을 사용한 이상치 탐지
for var in available_variables:
    try:
        # 결측값 확인
        if df[var].isna().all():
            print(f"{var}는 모든 값이 결측값입니다. IQR 이상치 탐지를 건너뜁니다.")
            continue
            
        plt.figure(figsize=(15, 8))
        
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
        plt.axhline(y=lower_bound, color='green', linestyle='--', label='하한 경계')
        plt.axhline(y=upper_bound, color='green', linestyle='--', label='상한 경계')
        plt.title(f'{var}의 IQR 이상치 (빨간색)')
        plt.grid(True)
        plt.legend()
        
        # 이상치 개수 출력
        outlier_count = np.sum(outliers)
        outlier_percent = outlier_count/len(df)*100
        plt.figtext(0.5, 0.01, f"이상치 개수: {outlier_count} ({outlier_percent:.2f}%)", 
                    ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_folder, f'{var}_iqr_outliers.png'))
        plt.close()
    except Exception as e:
        print(f"{var} IQR 이상치 탐지 중 오류 발생: {e}")
print("4. IQR 이상치 탐지 완료")

# 5. 계절별 분석
# 계절을 정의 (기상학적 계절)
seasons = {
    'winter': [12, 1, 2],  # 겨울
    'spring': [3, 4, 5],   # 봄
    'summer': [6, 7, 8],   # 여름
    'autumn': [9, 10, 11]  # 가을
}

# 계절 컬럼 추가
df['season'] = df.index.month.map(lambda m: 
                                 'winter' if m in seasons['winter'] else
                                 'spring' if m in seasons['spring'] else
                                 'summer' if m in seasons['summer'] else 'autumn')

# 기상/해양 관련 변수 선택 (가능한 변수 중에서만)
weather_vars = [var for var in ['10u', '10v', 'tprate', 'sic', 'thetao', 'so'] if var in available_variables]
if weather_vars:
    for var in weather_vars[:3]:  # 최대 3개만 처리
        try:
            plt.figure(figsize=(15, 8))
            sns.boxplot(x='season', y=var, data=df, order=['winter', 'spring', 'summer', 'autumn'])
            plt.title(f'{var}의 계절별 분포')
            plt.xlabel('계절')
            plt.ylabel(var)
            plt.tight_layout()
            plt.savefig(os.path.join(result_folder, f'{var}_seasonal_boxplot.png'))
            plt.close()
        except Exception as e:
            print(f"{var} 계절별 분석 중 오류 발생: {e}")
    print("5. 계절별 분석 완료")
else:
    print("5. 계절별 분석에 적합한 변수가 없습니다.")

# 요약 정보 저장
try:
    with open(os.path.join(result_folder, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"분석 날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"분석 데이터: 라스트마스크cmems.csv\n")
        f.write(f"분석 변수: {', '.join(available_variables)}\n\n")
        
        f.write("각 변수별 기술 통계량:\n")
        for var in available_variables:
            try:
                # 결측값 확인
                if df[var].isna().all():
                    f.write(f"\n{var} 통계: 모든 값이 결측값입니다.\n")
                    continue
                    
                f.write(f"\n{var} 통계:\n")
                f.write(f"{df[var].describe().to_string()}\n")
                
                # 이상치 개수 계산
                z = np.abs(stats.zscore(df[var], nan_policy='omit'))
                z_outliers = np.sum(z > z_threshold)
                
                Q1 = df[var].quantile(0.25)
                Q3 = df[var].quantile(0.75)
                IQR = Q3 - Q1
                iqr_outliers = np.sum((df[var] < Q1 - 1.5 * IQR) | (df[var] > Q3 + 1.5 * IQR))
                
                f.write(f"Z-score 이상치 개수: {z_outliers} ({z_outliers/len(df)*100:.2f}%)\n")
                f.write(f"IQR 이상치 개수: {iqr_outliers} ({iqr_outliers/len(df)*100:.2f}%)\n")
            except Exception as e:
                f.write(f"\n{var} 통계 계산 중 오류 발생: {e}\n")
    print("요약 정보 저장 완료")
except Exception as e:
    print(f"요약 정보 저장 중 오류 발생: {e}")

print("분석 완료! 결과를 확인하세요.")