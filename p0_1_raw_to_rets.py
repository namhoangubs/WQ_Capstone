from pandas import read_csv, merge
from functools import reduce
from numpy import log
from pathlib import Path


from pathlib import Path

BASE_DIR = Path(r"C:\Users\namho\Desktop\Previous PC\World Quant\Thesis\Replication")
INPUT_DIR = BASE_DIR / "Inputs"

def p0_1_raw_to_rets():
    EXOG_TICKERS = ['SPXT', 'DBLCIX', 'IBOXIG', 'JPEICORE', 'LT11TRUU', 'LBUTTRUU']
    FOLDER = BASE_DIR / 'processed'

    exog_files = [INPUT_DIR / f'{ticker}.csv' for ticker in EXOG_TICKERS]
    exog_list = [read_csv(file, index_col=[0], parse_dates=[0]) for file in exog_files]
    exog_px = reduce(lambda left, right: merge(left, right, on='DATE'), exog_list)

    stock_px = read_csv(INPUT_DIR / 'STOCKS.csv', index_col=[0], parse_dates=[0])
    stock_status_1999 = read_csv(INPUT_DIR / 'STOCK_STATUS_1999.csv')
    stock_status_2023 = read_csv(INPUT_DIR / 'STOCK_STATUS_2023.csv')

    stock_1999 = stock_status_1999[stock_status_1999.STATUS == 'Active'].TICKER
    stock_2023 = stock_status_2023[stock_status_2023.STATUS == 'Active'].TICKER
    stock_list = sorted(set(stock_1999).intersection(stock_2023))
    stock_px = stock_px[stock_list].dropna(axis=1)
    stock_tickers = stock_px.columns

    asset_px = merge(exog_px, stock_px, on='DATE', sort=True)
    asset_rets = log(asset_px).diff().dropna()

    Path(FOLDER).mkdir(parents=True, exist_ok=True)
    round(asset_px[EXOG_TICKERS], 5).to_csv(FOLDER / 'exog_px.csv')
    round(asset_px[stock_tickers], 5).to_csv(FOLDER / 'stock_px.csv')
    round(asset_rets[EXOG_TICKERS], 5).to_csv(FOLDER / 'exog_rets.csv')
    round(asset_rets[stock_tickers], 5).to_csv(FOLDER / 'stock_rets.csv')


if __name__ == '__main__':
    p0_1_raw_to_rets()