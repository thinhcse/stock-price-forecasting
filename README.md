Stock price S&P500 index is examined in this project. Before running or training the model, you need to collect the S&P500 data. In this repo, I provide a small tool for crawling the data from [Yahoo](https://finance.yahoo.com/quote/%5EGSPC/history?period1=1136073600&period2=1624665600&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true) using Selenium and Beautiful Soup. Once success, the data will be located in folder "data/".

1. Setup the environment with required packages and libraries using environment.yml (for running and training on CPU):
   ```
   conda update conda
   conda env create -f environment.yml
   sudo apt-get update
   sudo apt install chromium-chromedriver
   ```
   If you aim to run or train the model on GPU then you need to install torch library with CUDA support. You need to find an appropriate version of torch library on [https://pytorch.org/get-started/previous-versions/] to your local CUDA version.
3. Prepare S&P500 data:
   ```
   cd utils
   python data_crawler.py
   cd ..
   ```
   It needs time to download on the data. When finish, a successful message will be displayed.
4. Run the pretrained forecaster:
   ```
   python main.py
   ```
   In the case where you need to train it again, you can modify the configuration file in the folder "configs/" and run the command above with flag ```--train```.
6. Result:
   
  ![image](https://github.com/thinhcse/stock-price-forecasting/assets/111031775/5edfb6b6-d93e-4a0d-a43c-6ffe20af10c5)

   Instead of forecasting the open/close/volume/volatility of the stock index directly, I forecasted the scaled return price by applying Exponential Weighted Moving Average technique due to the fact that older observations should be given lower weights in addition. The model consists of a stack of 5 LSTM layers, each hidden state $h_i^t$ of layer $i$ has dimension 10. Before feeding the prices to the LSTM layers, I used a convolutinal layers to capture more important features of the data. The window size here is 14 (days) i.e. I use data of 14 days to predict the scaled return of the next day. Eventhough the forecasting is not complely exact, it is not an easy task!  
