#Script to backtest portfolio over 10 defined periods. To do over more periods, just add more dates. 
# Load necessary libraries
library(readr)     
library(quantmod)     
library(portfolioBacktest) 

# Step 1: Read the CSV file for stock tickers
csv_path <- "/Users/allison/Desktop/portfolio10.csv"
stock_list <- read_csv(csv_path)

# Extract stock tickers from the correct column
tickers <- stock_list[[1]]  

# Step 2: Define the periods
periods <- list(
  c("2015-04-24", "2017-04-24"),
  c("2013-10-09", "2015-10-08"),
  c("2012-07-06", "2014-07-09"),
  c("2016-07-05", "2018-07-03"),
  c("2016-11-22", "2018-11-21"),
  c("2015-11-23", "2017-11-21"),
  c("2011-05-12", "2013-05-14"),
  c("2010-11-17", "2012-11-17"),
  c("2016-08-19", "2018-08-20"),
  c("2015-05-15", "2017-05-15")
)

# Step 3: Prepare an empty list to hold results
all_results <- list()
all_sharpe_ratios <- numeric()
all_max_drawdowns <- numeric()

# Step 4: Loop through each period and perform the backtest
for (period in periods) {
  start_date <- period[1]
  end_date <- period[2]
  
  # Fetch historical data for the stocks in the given period
  getSymbols(tickers, src = "yahoo", from = start_date, to = end_date, auto.assign = TRUE)
  
  # Combine all stock data into a single xts object for the given period
  stock_prices <- do.call(merge, lapply(tickers, function(ticker) Ad(get(ticker))))
  
  # Prepare dataset list for buy-and-hold strategy
  dataset_list <- list(list(adjusted = stock_prices))  
  
  # Define a buy-and-hold strategy function (no rebalancing)
  buyHoldPortfolio <- function(dataset, ...) {
    N <- ncol(dataset$adjusted)
    return(rep(1/N, N))  # Equal weights at the start, no rebalancing
  }
  
  # Create a list of portfolio functions
  portfolio_funs <- list("BuyHold" = buyHoldPortfolio)
  
  # Perform the backtest for the current period
  bt_results <- portfolioBacktest(
    portfolio_funs = portfolio_funs,   # List of portfolio functions
    dataset_list = dataset_list        # List with one dataset
  )
  
  # Store the result
  all_results[[paste0(start_date, "_to_", end_date)]] <- bt_results
  
  # Extract performance metrics for Sharpe ratio and max drawdown
  sharpe <- backtestSummary(bt_results)$results$measures["Sharpe ratio", "BuyHold"]
  drawdown <- backtestSummary(bt_results)$results$measures["max drawdown", "BuyHold"]
  
  # Append results to lists for later median calculation
  all_sharpe_ratios <- c(all_sharpe_ratios, sharpe)
  all_max_drawdowns <- c(all_max_drawdowns, drawdown)
}

# Step 5: View backtest results for all periods
for (name in names(all_results)) {
  cat("\nBacktest results for period:", name, "\n")
  print(all_results[[name]])
  
  # Visualize the backtest results for each period
  backtestSelector(all_results[[name]], portfolio_name = "BuyHold", measures = c("Sharpe ratio", "max drawdown"))
  backtestTable(all_results[[name]], measures = c("Sharpe ratio", "max drawdown"))
}

# Step 6: Calculate and display the median Sharpe ratio and max drawdown
median_sharpe <- median(all_sharpe_ratios, na.rm = TRUE)
median_drawdown <- median(all_max_drawdowns, na.rm = TRUE)

cat("\nMedian Sharpe Ratio across all periods: ", median_sharpe, "\n")
cat("Median Max Drawdown across all periods: ", median_drawdown, "\n")

# Step 7: Optionally, summarize results for all periods
bt_summary <- lapply(all_results, backtestSummary)
for (name in names(bt_summary)) {
  cat("\nSummary for period:", name, "\n")
  summaryTable(bt_summary[[name]])
}
# Convert the backtest result into a dataframe if needed
backtest_df <- as.data.frame(bt_summary)

 # Save to CSV file
 write.csv(backtest_df, file = "backtest_results_2015-11-23_to_2017-11-21.csv", row.names = FALSE)
 
 # Repeat for other periods or adjust file names accordingly
 write.csv(backtest_df, file = "/Users/allison/Downloads/backtest_results.csv", row.names = FALSE)