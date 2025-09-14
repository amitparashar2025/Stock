# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 16:23:25 2025
@author: amiparas
"""

#!/usr/bin/env python3
"""
COMPLETE STOCK SCREENER WITH TECHNICAL, FUNDAMENTAL, LIQUIDITY, NEWS & TRIPLE TIMEFRAME ANALYSIS (4H, 1D, 1W)
"""

from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
import gspread
import yfinance as yf
import os
import time, logging, warnings
import pandas as pd
import numpy as np
import json

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('stock_screener.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class CompleteStockScreener:
    def __init__(self, creds_file="service_account.json"):
        self.creds_file = creds_file
        self.sheet = "Stock Selection by Python"
        self.ws_name = "Screener"
        self.start_row = 4
        self.rate_limit = 60
        self.requests = 0
        self.last_reset = time.time()
        self.delay = 1.0
        self.period = "6mo"
        self.interval_4h = "4h"
        self.interval_1d = "1d"
        self.interval_1w = "1wk"
        self.setup()

    def setup(self):
        service_account_info = json.loads(os.environ["GOOGLE_CREDENTIALS"])
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
        self.client = gspread.authorize(creds)
        self.ws = self.client.open(self.sheet).worksheet(self.ws_name)

    def throttle(self):
        now = time.time()
        if now - self.last_reset >= 60:
            self.requests = 0
            self.last_reset = now
        if self.requests >= self.rate_limit:
            time.sleep(60 - (now - self.last_reset) + 1)
            self.requests = 0
            self.last_reset = time.time()

    def safe_batch_update(self, updates):
        delay = 1
        for _ in range(5):
            try:
                self.ws.batch_update(updates, value_input_option='USER_ENTERED')
                return
            except APIError as e:
                if '429' in str(e):
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
        logger.error("Batch update failed after retries")

    def get_symbols(self):
        vals = self.ws.col_values(1)
        return [{'symbol': v.strip().upper(), 'row': i}
                for i, v in enumerate(vals[self.start_row - 1:], start=self.start_row) if v.strip()]

    def fetch_data(self, sym, interval):
        for _ in range(3):
            try:
                self.throttle()
                df = yf.Ticker(sym).history(period=self.period, interval=interval)
                self.requests += 1
                min_len = 200 if interval == self.interval_4h else 50
                if df.empty or len(df) < min_len:
                    return None
                time.sleep(self.delay)
                return df
            except:
                time.sleep(2)
        return None

    def fetch_fundamentals(self, sym):
        try:
            info = yf.Ticker(sym).info
            return (
                info.get('returnOnEquity'),
                info.get('trailingPE'),
                info.get('debtToEquity'),
                info.get('pegRatio'),
                info.get('currentRatio')
            )
        except:
            return (None, None, None, None, None)

    def fetch_latest_news(self, sym):
        try:
            self.throttle()
            news = yf.Ticker(sym).news
            self.requests += 1
            if news:
                return news[0].get('title', 'No recent news available')
        except:
            pass
        return "No recent news available"

    def calculate(self, df):
        # (keep the calculation body exactly as you had, indented)
        # ...
        return {...}

    def rec_sr_trend(self, df):
        # ...
        return sr, trend

    def rec_rsi_div(self, df):
        # ...
        return "🟡 No divergence"

    def rec_macd(self, df):
        # ...
        return "🟡 No MACD"

    def rec_rsi_macd(self, rsi, macd_rec):
        # ...
        return "🟡 Neutral"

    def rec_obv(self, obv):
        # ...
        return "🟡 Neutral OBV"

    def rec_vwap(self, close, vwap):
        # ...
        return "🟡 At VWAP"

    def rec_boll(self, close, ub, lb):
        # ...
        return "🟡 Within BB"

    def rec_comb_obv_vwap_macd(self, obv_rec, vwap_rec, macd_rec):
        # ...
        return "🟡 MIXED"

    def rec_stoch(self, pctK, pctD):
        # ...
        return "🟡 Stoch neutral"

    def rec_fib(self, fib_pct):
        # ...
        return f"{fib_pct:.1f}% near 50%"

    def rec_alligator(self, lips, teeth, jaws):
        # ...
        return "🟡 Alligator inactive"

    def rec_roe(self, roe):
        # ...
        return "No ROE data"

    def rec_pe(self, pe):
        # ...
        return "No P/E data"

    def rec_de(self, de):
        # ...
        return "No D/E data"

    def rec_peg(self, peg):
        # ...
        return "No PEG data"

    def rec_comb_fundamental(self, roe_rec, pe_rec, de_rec, peg_rec):
        # ...
        return "🟡 MIXED FUNDAMENTALS"

    def rec_current_ratio(self, cr):
        # ...
        return "No Current Ratio data"

    def rec_adtv(self, adtv):
        # ...
        return "No Volume data"

    def rec_liquidity(self, cr_rec, adtv_rec):
        # ...
        return "🟡 Moderate Liquidity"

    def score_sentiment(self, text):
        if not isinstance(text, str):
            return 0
        return text.count("🟢") - text.count("🔴")

    def rec_final(self, *recs):
        # ...
        return "🟡 HOLD"

    def calc_timeframe_rec(self, df, fund_comb, liquidity):
        # ...
        return "🟡 INSUFFICIENT DATA"

    def process(self):
        symbols = self.get_symbols()
        updates, success, fail = [], 0, 0
        for idx, info in enumerate(symbols, 1):
            sym, row = info['symbol'], info['row']
            print(f"\r[{idx}/{len(symbols)}] {sym}", end='')
            # (full process loop body here, indented properly)
        print()
        if updates:
            self.safe_batch_update(updates)
        print(f"\n✅ Done: {success} succeeded, {fail} failed")


if __name__ == '__main__':
    logger.info("Running stock screener without user prompts...")
    CompleteStockScreener().process()
