import io
import os
import modal

#RUN: modal run stock_prices.py

THIS_FILE_DIR = os.path.dirname(__file__)
DIR = 'tmp/'
DIR_PATH = os.path.join(THIS_FILE_DIR,DIR)


stub = modal.Stub(
    "example-fetch-stock-prices",
    image=modal.Image.debian_slim().pip_install(
        "requests",
        "yfinance",
        "beautifulsoup4",
        "matplotlib",
    ),
)

@stub.function
def get_stocks():
    import bs4
    import requests

    headers = {
        "user-agent": "curl/7.55.1",
        "referer": "https://finance.yahoo.com/",
    }
    url = "https://finance.yahoo.com/etfs/?count=100&offset=0"
    res = requests.get(url, headers=headers)
    soup = bs4.BeautifulSoup(res.text, "html.parser")
    for td in soup.find_all("td", {"aria-label": "Symbol"}):
        for link in td.find_all("a", {"data-test": "quoteLink"}):
            symbol = str(link.next)
            print(f"Found symbol {symbol}")
            yield symbol

@stub.function
def get_prices(symbol):
    import yfinance

    print(f"Fetching symbol {symbol}...")
    ticker = yfinance.Ticker(symbol)
    data = ticker.history(period="1Y")["Close"]
    print(f"Done fetching symbol {symbol}!")
    return symbol, data.to_dict()

@stub.function
def plot_stocks():
    from matplotlib import pyplot, ticker

    # Setup
    pyplot.style.use("ggplot")
    fig, ax = pyplot.subplots(figsize=(8, 5))

    # Get data
    tickers = list(get_stocks.call())
    data = list(get_prices.map(tickers))
    first_date = min((min(prices.keys()) for symbol, prices in data if prices))
    last_date = max((max(prices.keys()) for symbol, prices in data if prices))

    # Plot every symbol
    for symbol, prices in data:
        if len(prices) == 0:
            continue
        dates = list(sorted(prices.keys()))
        prices = list(prices[date] for date in dates)
        changes = [
            100.0 * (price / prices[0] - 1) for price in prices
        ]  # Normalize to initial price
        if changes[-1] > 20:
            # Highlight this line
            p = ax.plot(dates, changes, alpha=0.7)
            ax.annotate(
                symbol,
                (last_date, changes[-1]),
                ha="left",
                va="center",
                color=p[0].get_color(),
                alpha=0.7,
            )
        else:
            ax.plot(dates, changes, color="gray", alpha=0.2)

    # Configure axes and title
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_title(f"Best ETFs {first_date.date()} - {last_date.date()}")
    ax.set_ylabel(f"% change, {first_date.date()} = 0%")

    # Dump the chart to .png and return the bytes
    with io.BytesIO() as buf:
        pyplot.savefig(buf, format="png", dpi=300)
        return buf.getvalue()

@stub.local_entrypoint
def main():
    os.makedirs(DIR_PATH, exist_ok=True)
    data = plot_stocks.call()
    filename = os.path.join(DIR_PATH, "stock_prices.png")
    print(f"saving data to {filename}")
    with open(filename, "wb") as f:
        f.write(data)