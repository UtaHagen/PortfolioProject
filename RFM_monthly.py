#I think this was created to test the monthly period?
#is functional but I don't know why this exists lol

import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

aapl = yf.download('AAPL', start='2010-01-01', end='2024-01-01', progress=False)
sp500 = yf.download('^GSPC', start='2010-01-01', end='2024-01-01', progress=False)
#importing and using the csv
ffunds_path = [r'C:\Users\bsung\OneDrive\Documents\GitHub\PortfolioProject\Data\FEDFUNDS.csv']
ffunds_path_str = ffunds_path[0]
ffunds = pd.read_csv(ffunds_path_str)

#manipulating the Feds Rate
ffunds = ffunds.drop(ffunds.index[0])
ffunds['DATE'] = pd.to_datetime(ffunds['DATE'])
ffunds.set_index('DATE', inplace=True)
ffunds.index = ffunds.index.to_period('M').to_timestamp('M')

#resampling stock data to be on the monthly frequency
monthly_aapl = aapl.resample('ME').mean()
monthly_sp500 = sp500.resample('ME').mean()

#editted the shift to reflect the data's employment of a monthly frequency
monthly_aapl['Return'] = (monthly_aapl['Close'].shift(-1) - monthly_aapl['Close']) / monthly_aapl['Close']
monthly_sp500['Return'] = (monthly_sp500['Close'].shift(-1) - monthly_sp500['Close']) / monthly_sp500['Close']

data = pd.DataFrame(index=aapl.index)
data['AAPL_Return'] = monthly_aapl['Return']
data['SP500_Return'] = monthly_sp500['Return']
data['ffunds'] = ffunds['FEDFUNDS']
data['Target'] = (data['AAPL_Return'] > data['SP500_Return']).astype(int)

data.dropna(inplace=True)

X = data[['AAPL_Return', 'SP500_Return','ffunds']]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

new_data = X_test.iloc[-1:].copy()  
prediction = model.predict(new_data)
print("Prediction: ", "AAPL will outperform S&P 500" if prediction[0] == 1 else "AAPL will underperform S&P 500")

#RFM
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Scores: {scores}")
print(f"Mean CV Score: {scores.mean()}")

#Model Validation
import matplotlib.pyplot as plt
importances = model.feature_importances_
feature_names = X.columns
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest Model")
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()