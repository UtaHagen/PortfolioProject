#Use R package to backtest portfolio over one period and get summary statistics

library(readr)     
library(quantmod)     
library(portfolioBacktest) 

csv_path <- "/Users/allison/Desktop/pick.csv"
stock_list <- read_csv(csv_path)

tickers <- stock_list[[1]]  

tickers <- tickers[!is.na(tickers) & tickers != ""]  
getSymbols(tickers, src = "yahoo", from = "2020-01-01", to = "2024-01-01", auto.assign = TRUE)

stock_prices <- do.call(merge, lapply(tickers, function(ticker) Ad(get(ticker))))

dataset_list <- list(list(adjusted = stock_prices))  

buyHoldPortfolio <- function(dataset, ...) {
  N <- ncol(dataset$adjusted)
  return(rep(1/N, N))  
}

portfolio_funs <- list("BuyHold" = buyHoldPortfolio)

lookback_period <- min(252, nrow(stock_prices))  

bt_results <- portfolioBacktest(
  portfolio_funs = portfolio_funs,   
  dataset_list = dataset_list,       
  lookback = lookback_period         
)

print(bt_results)

backtestSelector(bt_results, portfolio_name = "BuyHold", measures = c("Sharpe ratio", "max drawdown"))
backtestTable(bt_results, measures = c("Sharpe ratio", "max drawdown"))
bt_summary <- backtestSummary(bt_results)
summaryTable(bt_summary)
