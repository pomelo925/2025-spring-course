# 清華大學 動力機械系 機器學習與應用課程   開課教授: 黃仲誼 Chung-I Huang
import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
install("numpy")
install("pandas")
install("matplotlib")
install("scikit-learn")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Read the dataset
yearly_sales_data = pd.read_csv('salesdata.csv')

# Polynomial regression analysis
X = yearly_sales_data['DayOfWeek'].values.reshape(-1, 1)
y = yearly_sales_data['Sales'].values
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# Predict sales for each day of the week
X_week = np.arange(1, 8).reshape(-1, 1)
X_week_poly = poly.transform(X_week)
y_week_pred = model.predict(X_week_poly)

# Plotting the average actual vs predicted sales by day of the week
weekly_sales_avg = yearly_sales_data.groupby('DayOfWeek')['Sales'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales_avg['DayOfWeek'], weekly_sales_avg['Sales'], 'bo-', label='Average Actual Sales')
plt.plot(X_week.flatten(), y_week_pred, 'ro-', label='Predicted Sales')
plt.title('Average Actual vs Predicted Sales by Day of the Week')
plt.xlabel('Day of the Week')
plt.xticks(np.arange(1, 8), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.ylabel('Sales')
plt.legend()
plt.savefig('sales_prediction.png')