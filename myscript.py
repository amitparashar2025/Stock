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
    def __init__(self):
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
        c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
        ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
        sma50 = c.rolling(50).mean().iloc[-1] if len(c) >= 50 else c.mean()
        sma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else c.mean()
        d = c.diff()
        g = d.clip(lower=0).rolling(14).mean()
        lo = -d.clip(upper=0).rolling(14).mean()
        rsi = (100 - 100 / (1 + g / lo)).iloc[-1]
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        obv = (np.sign(c.diff()) * v).fillna(0).cumsum().iloc[-1]
        tp = (h + l + c) / 3
        vwap = (tp * v).rolling(len(df)).sum().iloc[-1] / v.rolling(len(df)).sum().iloc[-1]
        mb = c.rolling(20).mean()
        sd = c.rolling(20).std()
        ub, lb = mb + 2 * sd, mb - 2 * sd
        lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
        pctK = 100 * (c - lo14) / (hi14 - lo14)
        pctD = pctK.rolling(3).mean()
        seg = c[-200:]
        fh, fl = seg.max(), seg.min()
        fib_pct = (c.iloc[-1] - fl) / (fh - fl) * 100 if fh != fl else 0
        lips = c.rolling(5).mean().iloc[-1]
        teeth = c.rolling(8).mean().iloc[-1]
        jaws = c.rolling(13).mean().iloc[-1]
        adtv = df['Volume'].groupby(df.index.date).sum().mean()
        return {
            'ema20': round(ema20, 2),
            'sma50': round(sma50, 2),
            'sma200': round(sma200, 2),
            'rsi': round(rsi, 2),
            'macd': macd,
            'sig': sig,
            'obv': round(obv, 0),
            'vwap': round(vwap, 2),
            'ub': ub.iloc[-1],
            'mb': mb.iloc[-1],
            'lb': lb.iloc[-1],
            'pctK': pctK.iloc[-1],
            'pctD': pctD.iloc[-1],
            'fib_pct': round(fib_pct, 2),
            'lips': lips,
            'teeth': teeth,
            'jaws': jaws,
            'close': c.iloc[-1],
            'adtv': adtv
        }

    # --- recommendation methods (rec_xxx) ---
    def rec_sr_trend(self, df):
        c = df['Close'].iloc[-1]
        e20 = df['Close'].ewm(span=20).mean().iloc[-1]
        pct = abs(c - e20) / e20 * 100 if e20 else 0
        sr = f"{'Support' if c > e20 else 'Resistance'}:{pct:.1f}%"
        s50 = df['Close'].rolling(min(50, len(df))).mean().iloc[-1]
        s200 = df['Close'].rolling(min(200, len(df))).mean().iloc[-1]
        trend = "🟡 NO CLEAR TREND"
        if len(df) >= 2:
            p50 = df['Close'].rolling(min(50, len(df))).mean().iloc[-2]
            p200 = df['Close'].rolling(min(200, len(df))).mean().iloc[-2]
            if p50 <= p200 < s50 > s200:
                trend = "🟢 GOLDEN CROSS"
            elif p50 >= p200 >= s50 < s200:
                trend = "🔴 DEATH CROSS"
            elif c > e20 > s50 > s200:
                trend = "🟢 UPTREND"
            elif c < e20 < s50 < s200:
                trend = "🔴 DOWNTREND"
        return sr, trend

    def rec_rsi_div(self, df):
        if len(df) < 2:
            return "🟡 No divergence"
        c = df['Close']
        rsi = 100 - 100 / (1 + (c.diff().clip(0).rolling(14).mean() /
                           (-c.diff().clip(upper=0).rolling(14).mean())))
        pdiff, rdiff = c.iloc[-1] - c.iloc[-2], rsi.iloc[-1] - rsi.iloc[-2]
        if pdiff > 0 and rdiff < 0:
            return "🔴 Bearish divergence"
        if pdiff < 0 and rdiff > 0:
            return "🟢 Bullish divergence"
        return "🟡 No divergence"

    def rec_macd(self, df):
        if len(df) < 2:
            return "🟡 No MACD"
        macd = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        d0, d1 = macd.iloc[-1] - sig.iloc[-1], macd.iloc[-2] - sig.iloc[-2]
        if d1 <= 0 < d0:
            return "🟢 MACD Bullish"
        if d1 >= 0 > d0:
            return "🔴 MACD Bearish"
        return "🟡 No MACD"

    def rec_rsi_macd(self, rsi, macd_rec):
        if "Bullish" in macd_rec and rsi > 50:
            return "🟢 Strong buy"
        if "Bearish" in macd_rec and rsi < 50:
            return "🔴 Strong sell"
        return "🟡 Neutral"

    def rec_obv(self, obv):
        return "🟢 Bullish OBV" if obv > 0 else "🔴 Bearish OBV" if obv < 0 else "🟡 Neutral OBV"

    def rec_vwap(self, close, vwap):
        if close > vwap:
            return "🟢 Above VWAP"
        if close < vwap:
            return "🔴 Below VWAP"
        return "🟡 At VWAP"

    def rec_boll(self, close, ub, lb):
        if close >= ub:
            return "🔴 Overbought BB"
        if close <= lb:
            return "🟢 Oversold BB"
        return "🟡 Within BB"

    def rec_comb_obv_vwap_macd(self, obv_rec, vwap_rec, macd_rec):
        b = sum("🟢" in x for x in [obv_rec, vwap_rec, macd_rec])
        r = sum("🔴" in x for x in [obv_rec, vwap_rec, macd_rec])
        if b == 3:
            return "🟢 VERY STRONG BUY"
        if b >= 2 and r == 0:
            return "🟢 STRONG BUY"
        if r == 3:
            return "🔴 VERY STRONG SELL"
        if r >= 2 and b == 0:
            return "🔴 STRONG SELL"
        if b > r:
            return "🟡 MODERATE BUY"
        if r > b:
            return "🟡 MODERATE SELL"
        return "🟡 MIXED"

    def rec_stoch(self, pctK, pctD):
        if pctK > 80 and pctD > 80:
            return "🔴 Overbought Stoch"
        if pctK < 20 and pctD < 20:
            return "🟢 Oversold Stoch"
        if pctK > pctD:
            return "🟢 Stoch rising"
        if pctK < pctD:
            return "🔴 Stoch falling"
        return "🟡 Stoch neutral"

    def rec_fib(self, fib_pct):
        lvl = ("23.6%" if fib_pct <= 23.6 else "38.2%" if fib_pct <= 38.2 else
               "50%" if fib_pct <= 50 else "61.8%" if fib_pct <= 61.8 else "78.6%")
        return f"{fib_pct:.1f}% near {lvl}"

    def rec_alligator(self, lips, teeth, jaws):
        if lips > teeth > jaws:
            return "🟢 Alligator awake (uptrend)"
        if lips < teeth < jaws:
            return "🔴 Alligator asleep (downtrend)"
        return "🟡 Alligator inactive"

    def rec_roe(self, roe):
        if roe is None:
            return "No ROE data"
        pct = roe * 100
        if pct > 15:
            return "🟢 ROE >15%"
        if pct > 5:
            return "🟡 ROE 5–15%"
        return "🔴 ROE <5%"

    def rec_pe(self, pe):
        if pe is None:
            return "No P/E data"
        if pe < 15:
            return "🟢 P/E <15"
        if pe < 25:
            return "🟡 P/E 15–25"
        return "🔴 P/E >25"

    def rec_de(self, de):
        if de is None:
            return "No D/E data"
        if de < 1:
            return "🟢 D/E <1"
        if de < 2:
            return "🟡 D/E 1–2"
        return "🔴 D/E >2"

    def rec_peg(self, peg):
        if peg is None:
            return "No PEG data"
        if peg < 1:
            return "🟢 PEG <1"
        if peg <= 1.5:
            return "🟡 PEG ~1"
        return "🔴 PEG >1.5"

    def rec_comb_fundamental(self, roe_rec, pe_rec, de_rec, peg_rec):
        bullish = sum("🟢" in x for x in [roe_rec, pe_rec, de_rec, peg_rec])
        bearish = sum("🔴" in x for x in [roe_rec, pe_rec, de_rec, peg_rec])
        if bullish == 4:
            return "🟢 EXCELLENT FUNDAMENTALS"
        if bullish >= 3 and bearish == 0:
            return "🟢 STRONG FUNDAMENTALS"
        if bearish == 4:
            return "🔴 POOR FUNDAMENTALS"
        if bearish >= 3 and bullish == 0:
            return "🔴 WEAK FUNDAMENTALS"
        if bullish > bearish:
            return "🟡 MODERATELY STRONG"
        if bearish > bullish:
            return "🟡 MODERATELY WEAK"
        return "🟡 MIXED FUNDAMENTALS"

    def rec_current_ratio(self, cr):
        if cr is None:
            return "No Current Ratio data"
        if cr > 1.5:
            return "🟢 Healthy Liquidity"
        if cr >= 1:
            return "🟡 Moderate Liquidity"
        return "🔴 Poor Liquidity"

    def rec_adtv(self, adtv):
        if not adtv:
            return "No Volume data"
        if adtv > 1e6:
            return "🟢 High Volume"
        if adtv > 3e5:
            return "🟡 Moderate Volume"
        return "🔴 Low Volume"

    def rec_liquidity(self, cr_rec, adtv_rec):
        if "🟢" in cr_rec and "🟢" in adtv_rec:
            return "🟢 Strong Liquidity"
        if "🔴" in cr_rec and "🔴" in adtv_rec:
            return "🔴 Weak Liquidity"
        if "🔴" in cr_rec or "🔴" in adtv_rec:
            return "🟡 Moderate Liquidity"
        return "🟡 Moderate Liquidity"

    def score_sentiment(self, text):
        if not isinstance(text, str):
            return 0
        return text.count("🟢") - text.count("🔴")

    def rec_final(self, *recs):
        total = sum(self.score_sentiment(r) for r in recs)
        if total > 6:
            return "🟢 STRONG BUY"
        if total > 2:
            return "🟢 BUY"
        if total >= -2:
            return "🟡 HOLD"
        if total >= -6:
            return "🔴 SELL"
        return "🔴 STRONG SELL"

    def calc_timeframe_rec(self, df, fund_comb, liquidity):
        if df is None or len(df) < 20:
            return "🟡 INSUFFICIENT DATA"
        v = self.calculate(df)
        sr, trend = self.rec_sr_trend(df)
        rdiv = self.rec_rsi_div(df)
        macd_rec = self.rec_macd(df)
        rmacd = self.rec_rsi_macd(v['rsi'], macd_rec)
        obv_rec = self.rec_obv(v['obv'])
        vwap_rec = self.rec_vwap(v['close'], v['vwap'])
        bb_rec = self.rec_boll(v['close'], v['ub'], v['lb'])
        comb_ovm = self.rec_comb_obv_vwap_macd(obv_rec, vwap_rec, macd_rec)
        stoch = self.rec_stoch(v['pctK'], v['pctD'])
        fib = self.rec_fib(v['fib_pct'])
        all_rec = self.rec_alligator(v['lips'], v['teeth'], v['jaws'])
        return self.rec_final(trend, rmacd, comb_ovm, stoch, fib, all_rec, fund_comb, liquidity)

    def process(self):
        symbols = self.get_symbols()
        updates, success, fail = [], 0, 0
        for idx, info in enumerate(symbols, 1):
            sym, row = info['symbol'], info['row']
            print(f"\r[{idx}/{len(symbols)}] {sym}", end='')
            df4 = self.fetch_data(sym, self.interval_4h)
            df1 = self.fetch_data(sym, self.interval_1d)
            dfw = self.fetch_data(sym, self.interval_1w)
            roe, pe, de, peg, cr = self.fetch_fundamentals(sym)
            if df4 is not None:
                v4 = self.calculate(df4)
                sr4, trend4 = self.rec_sr_trend(df4)
                rdiv4 = self.rec_rsi_div(df4)
                macd4 = self.rec_macd(df4)
                rmacd4 = self.rec_rsi_macd(v4['rsi'], macd4)
                obv4 = self.rec_obv(v4['obv'])
                vwap4 = self.rec_vwap(v4['close'], v4['vwap'])
                bb4 = self.rec_boll(v4['close'], v4['ub'], v4['lb'])
                comb4 = self.rec_comb_obv_vwap_macd(obv4, vwap4, macd4)
                stoch4 = self.rec_stoch(v4['pctK'], v4['pctD'])
                fib4 = self.rec_fib(v4['fib_pct'])
                all4 = self.rec_alligator(v4['lips'], v4['teeth'], v4['jaws'])
                roe_rec = self.rec_roe(roe)
                pe_rec = self.rec_pe(pe)
                de_rec = self.rec_de(de)
                peg_rec = self.rec_peg(peg)
                fund_comb = self.rec_comb_fundamental(roe_rec, pe_rec, de_rec, peg_rec)
                cr_rec = self.rec_current_ratio(cr)
                adtv_rec = self.rec_adtv(v4['adtv'])
                liq_rec = self.rec_liquidity(cr_rec, adtv_rec)
                final4 = self.rec_final(trend4, rmacd4, comb4, stoch4, fib4, all4, fund_comb, liq_rec)
                news = self.fetch_latest_news(sym)
                rec1 = self.calc_timeframe_rec(df1, fund_comb, liq_rec)
                recw = self.calc_timeframe_rec(dfw, fund_comb, liq_rec)
                updates.extend([
                    {'range': f'B{row}', 'values': [[v4['ema20']]]},
                    {'range': f'C{row}', 'values': [[v4['sma50']]]},
                    {'range': f'D{row}', 'values': [[v4['sma200']]]},
                    {'range': f'E{row}', 'values': [[sr4]]},
                    {'range': f'F{row}', 'values': [[trend4]]},
                    {'range': f'G{row}', 'values': [[rdiv4]]},
                    {'range': f'H{row}', 'values': [[macd4]]},
                    {'range': f'I{row}', 'values': [[rmacd4]]},
                    {'range': f'J{row}', 'values': [[obv4]]},
                    {'range': f'K{row}', 'values': [[vwap4]]},
                    {'range': f'L{row}', 'values': [[bb4]]},
                    {'range': f'M{row}', 'values': [[comb4]]},
                    {'range': f'N{row}', 'values': [[stoch4]]},
                    {'range': f'O{row}', 'values': [[fib4]]},
                    {'range': f'P{row}', 'values': [[all4]]},
                    {'range': f'Q{row}', 'values': [[roe_rec]]},
                    {'range': f'R{row}', 'values': [[pe_rec]]},
                    {'range': f'S{row}', 'values': [[de_rec]]},
                    {'range': f'T{row}', 'values': [[peg_rec]]},
                    {'range': f'U{row}', 'values': [[fund_comb]]},
                    {'range': f'V{row}', 'values': [[round(cr, 2) if cr else "N/A"]]},
                    {'range': f'W{row}', 'values': [[int(v4["adtv"])]]},
                    {'range': f'X{row}', 'values': [[liq_rec]]},
                    {'range': f'Y{row}', 'values': [[final4]]},
                    {'range': f'Z{row}', 'values': [[news]]},
                    {'range': f'AA{row}', 'values': [[rec1]]},
                    {'range': f'AB{row}', 'values': [[recw]]},
                ])
                success += 1
            else:
                for col in list('BCDEFGHIJKLMNO') + ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'AB']:
                    updates.append({'range': f'{col}{row}', 'values': [['NO DATA']]})
                fail += 1
        print()
        if updates:
            self.safe_batch_update(updates)
        print(f"\n✅ Done: {success} succeeded, {fail} failed")


if __name__ == '__main__':
    logger.info("Running stock screener without user prompts...")
    CompleteStockScreener().process()
