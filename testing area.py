import yfinance as yf
import pandas as pd
import requests
import tkinter as tk
from io import StringIO
from tkinter import ttk

url = 'https://stockanalysis.com/list/sp-500-stocks/'
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)

html = StringIO(response.text)

tables = pd.read_html(html)
df = tables[0]

print(df.columns)