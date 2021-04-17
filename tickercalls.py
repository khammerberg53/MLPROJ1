import yfinance as yf
import matplotlib.pyplot as plt

ticker = "MSFT"
data = yf.download("MSFT", start="2017-01-01", end="2017-12-30")

plt.figure()
plt.plot(data["Open"])
plt.plot(data["High"])
plt.plot(data["Low"])
plt.plot(data["Close"])
plt.title(f'{ticker} Price Chart')
plt.ylabel('Price (USD)')
plt.xlabel('Days')
plt.legend(['Open','High','Low','Close'], loc='upper left')
plt.show()