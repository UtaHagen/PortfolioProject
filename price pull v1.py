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

user_input1 = {'value': ''} 
user_input2 = {'value': ''}

def get_answer():
    for i in tree.get_children():
        tree.delete(i)
    user_input1['value'] = float(entry1.get())
    user_input2['value'] = float(entry2.get())
    filtered_df = df[(df['Stock Price'] >= user_input1['value']) & (df['Stock Price'] <= user_input2['value'])]
    for index, row in filtered_df.iterrows():
        tree.insert("", "end", values=list(row))

root = tk.Tk()

root.geometry("500x500")

root.title("Portfolio Project - Model 1")

label = tk.Label(root, text="Please input your stock price range:")
label.pack()

label1 = tk.Label(root, text="Minimum stock price")
label1.pack()

entry1 = tk.Entry(root)
entry1.pack()

label2 = tk.Label(root, text="Maximum stock price")
label2.pack()

entry2 = tk.Entry(root)
entry2.pack()

button = tk.Button(root, text="Search", command=get_answer)
button.pack()

tree = ttk.Treeview(root, columns=df.columns, show='headings')
tree.pack()

for column in df.columns:
    tree.heading(column, text=column)

root.mainloop()