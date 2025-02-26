# 清華大學 動力機械系 機器學習與應用課程   開課教授: 黃仲誼 Chung-I Huang

import sys
import subprocess
import pkg_resources

required = {'pandas', 'scikit-learn', 'matplotlib'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


#Pandas是用於資料操縱和分析的Python軟體庫。它建造在NumPy基礎上，並為操縱數值表格和時間序列，提供了資料結構和運算操作。
#Scikit-learn（曾叫做scikits.learn與sklearn）是用於Python程式語言的自由軟體機器學習庫。
#它包含了各種分類、回歸和聚類算法，包括多層感知器、支持向量機、隨機森林、梯度提升、k-平均聚類和DBSCAN

#matplotlib 畫圖與秀圖


# 讀取數據
data = pd.read_csv('dataset.csv')

# 分割特徵和目標變量
X = data[['Runtime']]  # 特徵矩陣
y = data['faults']     # 目標變量

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建線性回歸模型實例
model = LinearRegression()

# 擬合模型
model.fit(X_train, y_train)

# 預測測試集的結果
y_pred = model.predict(X_test)

# 計算性能指標
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 打印性能指標
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 繪製觀測值
plt.scatter(X_test, y_test, color='black', label='Actual faults')

# 繪製預測值
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted faults')

# 設定圖標題和軸標籤
plt.title('Linear Regression for Fault Prediction')
plt.xlabel('Runtime')
plt.ylabel('Faults')

# 顯示圖例
plt.legend()

# 顯示圖表
plt.savefig('LR.png')
