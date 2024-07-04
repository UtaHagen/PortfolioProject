#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

aapl = yf.download('AAPL', start='2010-01-01', end='2024-01-01', progress=False)
sp500 = yf.download('^GSPC', start='2010-01-01', end='2024-01-01', progress=False)

aapl['Return'] = (aapl['Close'].shift(-21) - aapl['Close']) / aapl['Close']
sp500['Return'] = (sp500['Close'].shift(-21) - sp500['Close']) / sp500['Close']

print(aapl[['Close', 'Return']].head(25))
print(sp500[['Close', 'Return']].head(25))

data = pd.DataFrame(index=aapl.index)
data['AAPL_Return'] = aapl['Return']
data['SP500_Return'] = sp500['Return']
data['Target'] = (data['AAPL_Return'] > data['SP500_Return']).astype(int)

data.dropna(inplace=True)

X = data[['AAPL_Return', 'SP500_Return']]
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


# In[2]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Scores: {scores}")
print(f"Mean CV Score: {scores.mean()}")


# In[3]:


import matplotlib.pyplot as plt
importances = model.feature_importances_
feature_names = X.columns
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest Model")
plt.show()


# In[4]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


# In[ ]:




