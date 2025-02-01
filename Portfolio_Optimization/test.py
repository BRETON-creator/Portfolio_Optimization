import yfinance as yf
import matplotlib.pyplot as plt

# Télécharger les données
stock = yf.Ticker("AAPL")
hist = stock.history(period="1y")

# Créer le graphique
plt.figure(figsize=(12, 6))
plt.plot(hist['Close'])
plt.title('Prix de clôture sur un an')
plt.xlabel('Date')
plt.ylabel('Prix')
plt.grid(True)
plt.show()