# 清華大學 動力機械系 機器學習與應用課程   開課教授: 黃仲誼 Chung-I Huang

import sys
import subprocess
import pkg_resources

# 確保所有需要的套件都已安裝
required = {'pandas', 'scikit-learn', 'matplotlib', 'numpy'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 讀取數據
data = pd.read_csv('dataset.csv')

# 分割特徵和目標變量
X = data[['Runtime']].values  # 特徵矩陣
y = data['faults'].values     # 目標變量

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建一個多項式特徵生成器
degree = 5
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# 擬合模型
polyreg.fit(X_train, y_train)

# 預測測試集的結果
y_pred = polyreg.predict(X_test)

# 計算性能指標
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 打印性能指標
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 繪製觀測值
plt.scatter(X_test, y_test, color='black', label='Actual faults')

# 繪製預測值
# 為了畫出平滑的曲線，我們對一系列點進行預測
X_fit = np.linspace(X.min(), X.max(), 100)[:, np.newaxis]
y_fit = polyreg.predict(X_fit)

plt.plot(X_fit, y_fit, color='blue', linewidth=3, label='Predicted faults')

# 設定圖標題和軸標籤
plt.title('Nonlinear Regression for Fault Prediction')
plt.xlabel('Runtime')
plt.ylabel('Faults')

# 顯示圖例
plt.legend()

# 顯示圖表
plt.show()
