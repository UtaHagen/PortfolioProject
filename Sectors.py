import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

csv_path = '/Users/allison/Desktop/sectorsxd - Sheet1.csv'  
sectors_df = pd.read_csv(csv_path)

print("Column names:", sectors_df.columns)

sectors_df = sectors_df.dropna(subset=['Stock'])

results = []

for index, row in sectors_df.iterrows():
    company = row['Stock']
    sector = row['Sector']

    company_data = yf.download(company, start='2010-01-01', end='2024-01-01', progress=False)
    sector_data = yf.download(sector, start='2010-01-01', end='2024-01-01', progress=False)

    print(f"Company: {company}")
    print(f"Company data head: {company_data.head()}")
    print(f"Sector: {sector}")
    print(f"Sector data head: {sector_data.head()}")

    #put in some checks to attempt to identify why some data is not pulling
    if company_data.empty or sector_data.empty:
        print(f"Data for {company} or {sector} is empty")
        continue

    monthly_company = company_data.resample('M').mean()
    monthly_sector = sector_data.resample('M').mean()

    if 'Close' not in monthly_company.columns or 'Close' not in monthly_sector.columns:
        print(f"Close column missing in data for {company} or {sector}")
        continue

    monthly_company['Return'] = (monthly_company['Close'].shift(-1) - monthly_company['Close']) / monthly_company['Close']
    monthly_sector['Return'] = (monthly_sector['Close'].shift(-1) - monthly_sector['Close']) / monthly_sector['Close']

    data = pd.DataFrame(index=company_data.index)
    data['Company_Return'] = monthly_company['Return']
    data['Sector_Return'] = monthly_sector['Return']
    data['Target'] = (data['Company_Return'] > data['Sector_Return']).astype(int)

    data.dropna(inplace=True)

    X = data[['Company_Return', 'Sector_Return']]
    y = data['Target']

    random_state = np.random.randint(0, 1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Company: {company}, Sector: {sector}")
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    new_data = X_test.iloc[-1:].copy()
    prediction = model.predict(new_data)
    prediction_text = f"{company} will outperform {sector}" if prediction[0] == 1 else f"{company} will underperform {sector}"
    print("Prediction: ", prediction_text)
    
    results.append([company, sector, accuracy, prediction_text])
    
    print("\n" + "="*50 + "\n")

# save as excel file for analysis of results
results_df = pd.DataFrame(results, columns=['Company', 'Sector', 'Accuracy', 'Prediction'])
results_df.to_excel('/Users/allison/Desktop/prediction_results.xlsx', index=False)

print("Results have been saved to prediction_results.xlsx")