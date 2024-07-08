import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score, classification_report

#importing the data
ffunds_path = [r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\FEDFUNDS.csv']
ffunds_path_str = ffunds_path[0]

#reading the files
aapl = yf.download('AAPL', start='2010-01-01', end='2024-01-01', progress=False)
sp500 = yf.download('^GSPC', start='2010-01-01', end='2024-01-01', progress=False)
ffunds = pd.read_csv(ffunds_path_str)

#ffunds Data Cleanup + Manipulation
ffunds['DATE'] = pd.to_datetime(ffunds['DATE'])
ffunds.set_index('DATE', inplace=True)

#aapl + sp500 Data Manipulation
aapl['Return'] = (aapl['Close'].shift(-21) - aapl['Close']) / aapl['Close']
sp500['Return'] = (sp500['Close'].shift(-21) - sp500['Close']) / sp500['Close']


aapl['monthly_average'] = aapl['Return'].resample('ME').mean()
sp500['monthly_average'] = sp500['Return'].resample('ME').mean()

#print(aapl[['Close', 'Return']].head(25))
#print(sp500[['Close', 'Return']].head(25))

data = pd.DataFrame(index=aapl.index)
data['AAPL_Return'] = aapl['monthly_average']
data['SP500_Return'] = sp500['monthly_average']
data['Fed Rate'] = ffunds['FEDFUNDS']

print(data)

data['Target'] = (data['AAPL_Return'] > data['SP500_Return']).astype(int)

data.dropna(inplace=True)

X = data[['AAPL_Return', 'SP500_Return', 'Fed Rate']]
y = data['Target']

#Test Section - testing for stationarity
#adf_test = adfuller(X['AAPL_Return'])
#adf_test = adfuller(X['SP500_Return'])
#print(f'p-value: {adf_test[1]}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

new_data = X_test.iloc[-1:].copy()  
prediction = model.predict(new_data)
print("Prediction: ", "AAPL will outperform S&P 500" if prediction[0] == 1 else "AAPL will underperform S&P 500")


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Scores: {scores}")
print(f"Mean CV Score: {scores.mean()}")


import matplotlib.pyplot as plt
importances = model.feature_importances_
feature_names = X.columns
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest Model")
plt.show()

