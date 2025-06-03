# Cryptocurrency Analysis and Trading Library - REFACTORED VERSION
# Comprehensive toolkit for cryptocurrency data analysis, sentiment analysis, and trading signals
# All functions consolidated and duplicates removed

# ===================== UNIFIED IMPORTS SECTION =====================

# Core Python imports
import os
import gzip
import pickle
import requests
import pandas as pd
import numpy as np
import logging
import time
import gc
import shutil
import json
import contextlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import pytz
from scipy.stats import ks_2samp

# Machine Learning imports
try:
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import StackingClassifier
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False

# Try importing optional dependencies
try:
    import feedparser
    from bs4 import BeautifulSoup
    from deep_translator import GoogleTranslator
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from transformers import pipeline
    from googletrans import Translator
    WEB_SCRAPING_AVAILABLE = True
    NLP_AVAILABLE = True
except ImportError:
    feedparser = None
    BeautifulSoup = None
    GoogleTranslator = None
    SentimentIntensityAnalyzer = None
    pipeline = None
    Translator = None
    WEB_SCRAPING_AVAILABLE = False
    NLP_AVAILABLE = False

# GPU dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUtil = None
    GPUTIL_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    PYNVML_AVAILABLE = False

# Streamlit for UI
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Constants for Binance API
_KLINES_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
]

# ===================== CONFIGURATION AND CONSTANTS =====================
BINANCE_BASE_URL = "https://api.binance.com"
DEFAULT_symbol = "BTC"
DEFAULT_QUOTE_BINANCE = "USDT"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
DEFAULT_TIMEOUT = 10

# Configuration settings
@dataclass
class Settings:
    DEFAULT_QUOTE_BINANCE: str = "USDT"
    TIMEZONE: str = "UTC"
    DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")
    MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models")
    CACHE_DIR: str = os.path.join(os.path.dirname(__file__), "cache")
    HISTORY_REQUIREMENTS: Dict[str, int] = None
    
    def __post_init__(self):
        if self.HISTORY_REQUIREMENTS is None:
            self.HISTORY_REQUIREMENTS = {
                "5m": 7, "15m": 30, "30m": 60, "1h": 90, "4h": 180, "1d": 365
            }
        # Create directories
        for directory in [self.DATA_DIR, self.MODEL_DIR, self.CACHE_DIR]:
            os.makedirs(directory, exist_ok=True)

settings = Settings()
DATA_DIR = settings.DATA_DIR
MODEL_DIR = settings.MODEL_DIR
CACHE_DIR = settings.CACHE_DIR
HISTORY_REQUIREMENTS = settings.HISTORY_REQUIREMENTS

# Thiết lập logging thống nhất
def setup_unified_logging(log_level=logging.INFO):
    """Setup unified logging for the entire application"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('crypto_analysis.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def get_logger(name):
    """Get logger instance"""
    return logging.getLogger(name)

# Logger chính
logger = setup_unified_logging()

# ===================== GPU INFO CLASSES =====================
@dataclass
class GPUInfo:
    """GPU information container"""
    id: int
    name: str
    memory_total: float  # GB
    memory_free: float   # GB
    memory_used: float   # GB
    memory_percent: float
    temperature: Optional[float] = None
    power_usage: Optional[float] = None
    utilization: float = 0.0
    compute_capability: Optional[Tuple[int, int]] = None
    is_available: bool = True

@dataclass
class GPUAllocation:
    """GPU allocation tracking"""
    gpu_id: int
    allocated_memory: float  # GB
    allocated_tasks: List[str]
    max_memory: float  # GB
    reservation_time: float

# Logger chính
logger = setup_unified_logging()

# ===================== CLASS 1: CONSOLIDATED DATA COLLECTION =====================
class DataCollection:
    """
    CONSOLIDATED Data Collection Class - All data fetching and management
    
    Features consolidated from multiple duplicate implementations:
    - Binance API data fetching with pagination and error handling
    - Local data caching with gzip compression
    - Data validation and cleaning
    - Incremental updates with smart timing
    - News data collection (moved from NewsAnalysis)
    
    Removed duplications:
    - Multiple fetch_binance_data implementations
    - Redundant update_data methods
    - Duplicate fetch_klines functions
    """
    
    def __init__(self):
        """Initialize unified data collection system"""
        self.settings = settings
        self.data_dir = DATA_DIR
        self.logger = get_logger('data_collection')
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Exchange configuration
        self.binance_base_url = BINANCE_BASE_URL
        self.default_symbol = DEFAULT_symbol
        self.default_quote = DEFAULT_QUOTE_BINANCE
        self.timeout = DEFAULT_TIMEOUT
        
        # Cache paths
        self.news_cache_path = os.path.join(self.data_dir, "news_cache.csv")
    
    # =============== CORE DATA FETCHING METHODS ===============
    
    def fetch_klines(self, symbol: str, tf: str, start_time: Optional[int] = None) -> pd.DataFrame:
        """
        CONSOLIDATED klines fetching with smart caching
        Combines all duplicate implementations into one optimized method
        """
        # Normalize symbol format
        symbol = symbol.upper()
        quote = getattr(self.settings, "DEFAULT_QUOTE_BINANCE", "USDT")
        if not symbol.endswith(quote):
            symbol = f"{symbol}{quote}"
        
        path = os.path.join(self.data_dir, f"{symbol}@{tf}.pkl.gz")

        # If no existing data or full reload requested
        if not os.path.exists(path) or start_time is None:
            df_full = self.fetch_new_data_from_binance(symbol, tf, start_time)
            if df_full.empty:
                return pd.DataFrame()
                
            # Normalize timestamp column
            if "close_time" not in df_full.columns:
                raise KeyError(f"Binance data missing 'close_time' for {symbol}@{tf}")
            df_full["timestamp"] = pd.to_datetime(df_full["close_time"], unit='ms', utc=True)
            
            # Save to cache
            with gzip.open(path, "wb") as f:
                pickle.dump(df_full, f)
            self.logger.info(f"Saved {len(df_full)} rows to cache: {symbol}@{tf}")
            return df_full.reset_index(drop=True)

        # Fetch only new data since last cached timestamp
        df_new = self.fetch_new_data_from_binance(symbol, tf, start_time)
        if df_new is None or df_new.empty:
            self.logger.info(f"No new data available for {symbol}@{tf}")
            return pd.DataFrame()

        # Normalize timestamp for new data
        if "close_time" not in df_new.columns:
            raise KeyError(f"Binance data missing 'close_time' for {symbol}@{tf}")
        df_new["timestamp"] = pd.to_datetime(df_new["close_time"], unit='ms', utc=True)        self.logger.info(f"Fetched {len(df_new)} new rows for {symbol}@{tf}")
        return df_new.reset_index(drop=True)
    def fetch_new_data_from_binance(self, symbol: str, timeframe: str, start_time: Optional[int] = None) -> pd.DataFrame:
        """
        CONSOLIDATED Binance API fetching with advanced pagination and error handling
        Removes all duplicate implementations and combines best features
        """
        # Normalize symbol format
        symbol = symbol.upper()
        quote = getattr(self.settings, "DEFAULT_QUOTE_BINANCE", "USDT")
        if not symbol.endswith(quote):
            symbol = f"{symbol}{quote}"

        url = "https://api.binance.com/api/v3/klines"
        limit = 1000
        
        # Determine start timestamp
        if start_time is not None:
            start_ts = int(start_time)
        else:
            # Use history requirements if available
            days = getattr(self.settings, 'HISTORY_REQUIREMENTS', {}).get(timeframe, 30)
            if days >= 3650:
                # Use Bitcoin genesis date for very long history
                dt0 = datetime(2017, 1, 1, tzinfo=timezone.utc)
                start_ts = int(dt0.timestamp() * 1000)
            else:
                dt0 = datetime.now(timezone.utc) - timedelta(days=days)
                start_ts = int(dt0.timestamp() * 1000)

        all_klines = []
        current_start = start_ts
        max_retries = 3
        
        while True:
            params = {
                "symbol": symbol,
                "interval": timeframe,
                "limit": limit
            }
            if current_start is not None:
                params["startTime"] = current_start

            # Retry logic for API calls
            for retry in range(max_retries):
                try:
                    resp = requests.get(url, params=params, timeout=self.timeout)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    if retry == max_retries - 1:
                        self.logger.error(f"Failed to fetch {symbol}@{timeframe} after {max_retries} retries: {e}")
                        return pd.DataFrame(columns=_KLINES_COLS)
                    time.sleep(1)  # Wait before retry

            if not data:
                break

            all_klines.extend(data)
            
            # Set next start time for pagination
            last_close = data[-1][6]  # close_time
            current_start = last_close + 1

            if len(data) < limit:
                break
                
            # Small delay to respect rate limits
            time.sleep(0.1)

        # Return empty DataFrame if no data
        if not all_klines:
            return pd.DataFrame(columns=_KLINES_COLS)

        # Create DataFrame with proper column names
        df = pd.DataFrame(all_klines, columns=_KLINES_COLS)
        
        # Convert timestamp columns
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        
        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume",
                       "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df["number_of_trades"] = df["number_of_trades"].astype(int, errors="ignore")
        self.logger.info(f"Fetched {len(df)} records for {symbol}@{timeframe}")
        return df
    
    def update_data(self, symbol: str, timeframe: str = None, deep_train: bool = False) -> pd.DataFrame:
        """
        CONSOLIDATED data update method - removes all duplicates
        Intelligently updates data with smart caching and timing
        """
        symbol = symbol.upper()
        quote = getattr(self.settings, "DEFAULT_QUOTE_BINANCE", "USDT")
        if not symbol.endswith(quote):
            symbol = f"{symbol}{quote}"

        intervals = [timeframe] if timeframe else list(HISTORY_REQUIREMENTS.keys())
        result_frames = []

        for tf in intervals:
            self.logger.info(f"Updating {symbol}@{tf}")
            path = os.path.join(self.data_dir, f"{symbol}@{tf}.pkl.gz")

            # Load existing data and determine start time
            df_old = pd.DataFrame()
            start_time = None
            
            if os.path.isfile(path) and not deep_train:
                try:
                    with gzip.open(path, "rb") as f:
                        df_old = pickle.load(f)
                    
                    if not df_old.empty and "close_time" in df_old.columns:
                        last_ts = pd.to_datetime(df_old["close_time"].iloc[-1], utc=True)
                        
                        # Check if enough time has passed for new candle
                        qty, unit = int(tf[:-1]), tf[-1]
                        if unit == "m":
                            wait = timedelta(minutes=qty)
                        elif unit == "h":
                            wait = timedelta(hours=qty)
                        elif unit == "d":
                            wait = timedelta(days=qty)
                        else:
                            wait = timedelta(hours=1)  # default
                        
                        now = datetime.now(timezone.utc)
                        if now < last_ts + wait:
                            self.logger.info(f"{symbol}@{tf}: No new candle expected yet")
                            df_new = pd.DataFrame()
                        else:
                            start_time = int(last_ts.timestamp() * 1000) + 1
                            df_new = self.fetch_klines(symbol, tf, start_time)
                except Exception as e:
                    self.logger.warning(f"Error reading cache for {tf}: {e}, fetching full history")
                    try:
                        os.remove(path)
                    except:
                        pass
                    df_new = self.fetch_klines(symbol, tf, None)
            else:
                # First time or deep training
                days = HISTORY_REQUIREMENTS.get(tf, 30 if not deep_train else 365)
                start_dt = datetime.now(timezone.utc) - timedelta(days=days)
                start_time = int(start_dt.timestamp() * 1000)
                self.logger.info(f"{'Deep training' if deep_train else 'First time'} fetch from {start_dt}")
                df_new = self.fetch_klines(symbol, tf, start_time)

            # Combine old and new data
            if df_new.empty:
                self.logger.info(f"{symbol}@{tf}: No new data")
                df_tf = df_old.copy()
            else:
                if not df_old.empty and not deep_train:
                    df_tf = pd.concat([df_old, df_new], ignore_index=True)\
                            .drop_duplicates(subset="close_time")\
                            .sort_values("close_time")
                else:
                    df_tf = df_new.copy()

                # Save updated data
                try:
                    with gzip.open(path, "wb") as f:
                        pickle.dump(df_tf, f)
                    self.logger.info(f"Saved {len(df_tf)} rows to {path}")
                except Exception as e:
                    self.logger.error(f"Error saving to {path}: {e}")

            result_frames.append(df_tf)

        # Combine all timeframes
        combined = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
        
        if not combined.empty:
            try:
                combined.to_csv(os.path.join(self.data_dir, f"{symbol}_combined.csv"), index=False)
                self.logger.info(f"Saved combined CSV for {symbol}")
            except Exception as e:
                self.logger.error(f"Error saving combined CSV: {e}")

        self.logger.info(f"Completed data update for {symbol}")
        return combined

    # =============== UTILITY METHODS ===============
    
    def save_compressed(self, data, file_path):
        """Save data with gzip compression"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load_compressed(self, file_path):
        """Load compressed data"""
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)

# =============== CONSOLIDATED MARKET ANALYSIS CLASS ===============

class MarketAnalysis:
    """
    CONSOLIDATED Market Analysis Class - All market sentiment and external data analysis
    
    Features consolidated from multiple duplicate implementations:
    - Sentiment analysis (Fear & Greed Index)
    - On-chain data analysis
    - Order book analysis
    - Intermarket correlation analysis
    
    Removed duplications:
    - Multiple get_market_sentiment implementations
    - Redundant on-chain data fetching
    - Duplicate order book analysis functions
    """
    
    def __init__(self):
        """Initialize consolidated market analysis"""
        self.logger = get_logger('market_analysis')
    
    # =============== SENTIMENT ANALYSIS METHODS ===============
    
    def get_market_sentiment(self):
        """
        Get crypto market sentiment (Fear & Greed Index)
        Returns dict with 'fear_greed_index' (int) and 'classification' (str)
        """
        try:
            url = "https://api.alternative.me/fng/?limit=1&format=json"
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                self.logger.error(f"Fear & Greed API error (HTTP {response.status_code}): {response.text}")
                return {"fear_greed_index": None, "classification": None}

            data = response.json()
            if data.get("data"):
                entry = data["data"][0]
                value = int(entry.get("value", 0))
                classification = entry.get("value_classification", "")
                return {"fear_greed_index": value, "classification": classification}
            else:
                return {"fear_greed_index": None, "classification": None}

        except Exception as e:
            self.logger.error(f"Error fetching Fear & Greed Index: {e}")
            return {"fear_greed_index": None, "classification": None}

    # =============== ON-CHAIN ANALYSIS METHODS ===============
    
    def get_onchain_data(self):
        """
        Get Bitcoin on-chain statistics via blockchain.info
        Returns dict with metrics: tx_count_24h, btc_sent_24h, mempool_tx, hash_rate, market_cap_usd, btc_price, total_btc
        """
        metrics = {}
        try:
            urls = {
                "tx_count_24h": "https://blockchain.info/q/24hrtransactioncount",
                "btc_sent_24h":  "https://blockchain.info/q/24hrbtcsent",
                "mempool_tx":    "https://blockchain.info/q/unconfirmedcount",
                "hash_rate":     "https://blockchain.info/q/hashrate",
                "market_cap_usd":"https://blockchain.info/q/marketcap",
                "btc_price":     "https://blockchain.info/q/24hrprice",
                "total_btc":     "https://blockchain.info/q/totalbc"
            }

            for key, url in urls.items():
                try:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code != 200:
                        metrics[key] = None
                        self.logger.warning(f"HTTP {resp.status_code} error for {key}: {resp.text}")
                        continue

                    text = resp.text.strip()
                    if text == "":
                        metrics[key] = None
                    else:
                        if key in ["tx_count_24h", "mempool_tx"]:
                            metrics[key] = int(float(text))
                        elif key in ["btc_sent_24h", "total_btc"]:
                            val = float(text)
                            metrics[key] = val / 1e8  # satoshi -> BTC
                        else:
                            try:
                                metrics[key] = float(text)
                            except ValueError:
                                metrics[key] = text
                except Exception as e:
                    metrics[key] = None
                    self.logger.warning(f"Connection error for {key}: {e}")

        except Exception as e:
            self.logger.error(f"General error fetching on-chain data: {e}")

        return metrics

    # =============== ORDER BOOK ANALYSIS METHODS ===============
    
    def analyze_orderbook(self, symbol="BTCUSDT"):
        """
        Analyze order book for a trading pair
        Returns dict with: best_bid, best_ask, spread, spread_percent, bid_volume, ask_volume, imbalance
        """
        result = {}
        try:
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=50"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in data.get("asks", [])]

            if not bids or not asks:
                return result

            best_bid_price, best_bid_qty = max(bids, key=lambda x: x[0])
            best_ask_price, best_ask_qty = min(asks, key=lambda x: x[0])
            spread = best_ask_price - best_bid_price
            mid_price = (best_ask_price + best_bid_price) / 2.0
            total_bid_vol = sum(q for _, q in bids)
            total_ask_vol = sum(q for _, q in asks)
            imbalance = 0.0
            if total_bid_vol + total_ask_vol:
                imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

            result = {
                "best_bid": best_bid_price,
                "best_ask": best_ask_price,
                "spread": spread,
                "spread_percent": (spread / mid_price * 100.0) if mid_price else None,
                "bid_volume": total_bid_vol,
                "ask_volume": total_ask_vol,
                "imbalance": imbalance
            }

        except Exception as e:
            self.logger.error(f"Error analyzing order book for {symbol}: {e}")

        return result

    # =============== INTERMARKET ANALYSIS METHODS ===============
    
    def get_intermarket_data(self):
        """
        Get price and change % for SPY (S&P500), GLD (Gold), DX=F (DXY)
        Returns dict with keys: sp500_price, sp500_change_pct, gold_price, gold_change_pct, dxy_price, dxy_change_pct
        """
        data = {}
        symbols = {
            "sp500": "SPY",
            "gold":  "GLD",
            "dxy":   "DX=F"
        }

        try:
            for key, symbol in symbols.items():
                url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
                resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code != 200:
                    self.logger.warning(f"Yahoo Finance API error for {symbol}: HTTP {resp.status_code}")
                    continue

                result = resp.json().get("quoteResponse", {}).get("result", [])
                if result:
                    item = result[0]
                    price = item.get("regularMarketPrice")
                    change_pct = item.get("regularMarketChangePercent")
                    data[f"{key}_price"] = price
                    data[f"{key}_change_percent"] = change_pct

        except Exception as e:
            self.logger.error(f"Error fetching intermarket data: {e}")

        return data

# =============== CONSOLIDATED NEWS ANALYSIS CLASS ===============

class NewsAnalysis:
    """
    CONSOLIDATED News Analysis Class - All news collection, analysis, and sentiment processing
    
    Features consolidated from multiple duplicate implementations:
    - RSS feed collection from multiple sources
    - News summarization and translation
    - Sentiment analysis of news content
    - Intelligent caching with incremental updates
    
    Removed duplications:
    - Multiple get_news implementations
    - Redundant RSS feed processing
    - Duplicate sentiment analysis functions
    - Multiple translation utilities
    """
    
    def __init__(self):
        """Initialize consolidated news analysis"""
        self.logger = get_logger('news_analysis')
        self.news_cache_path = os.path.join(DATA_DIR, "news_cache.csv")
        
        # Initialize sentiment analyzer
        try:
            if NLP_AVAILABLE and SentimentIntensityAnalyzer:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            else:
                self.sentiment_analyzer = None
                self.logger.warning("Sentiment analyzer not available")
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment analyzer: {e}")
            self.sentiment_analyzer = None
    
    # =============== NEWS COLLECTION METHODS ===============
    
    def get_news(self, symbol, max_items=10, extra_sources=None):
        """
        CONSOLIDATED news collection with intelligent caching
        Get news about a cryptocurrency with sentiment analysis and translation
        """
        # Load existing cache
        if os.path.exists(self.news_cache_path):
            try:
                cached = pd.read_csv(self.news_cache_path, parse_dates=['publishedAt'])
                last_cached = cached['publishedAt'].max()
            except Exception as e:
                self.logger.warning(f"Error loading news cache: {e}")
                cached = pd.DataFrame()
                last_cached = None
        else:
            cached = pd.DataFrame()
            last_cached = None

        try:
            # Fetch new news data
            news_df = self._fetch_rss_feeds(max_items, extra_sources)
            
            # Filter for new news only
            if last_cached is not None:
                news_df = news_df[news_df['publishedAt'] > last_cached]
            
            # Filter by symbol keyword
            symbol_keyword = symbol.replace("USDT", "").replace("BTC", "Bitcoin").replace("ETH", "Ethereum")
            news_df_filtered = news_df[
                news_df['title'].str.contains(symbol_keyword, case=False, na=False) |
                news_df['summary'].str.contains(symbol_keyword, case=False, na=False)
            ]

            summaries = []
            if not news_df_filtered.empty:
                # Process news with parallel execution
                with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
                    futures = {}
                    for _, row in news_df_filtered.iterrows():
                        future = executor.submit(self._process_news_item, row)
                        futures[future] = row

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result:
                                summaries.append(result)
                        except Exception as e:
                            self.logger.error(f"Error processing news item: {e}")

            # Update cache
            if summaries:
                all_summaries = pd.concat([cached, pd.DataFrame(summaries)], ignore_index=True)\
                                 .drop_duplicates(subset=['url'])\
                                 .sort_values('publishedAt', ascending=False)
                
                try:
                    all_summaries.to_csv(self.news_cache_path, index=False)
                    self.logger.info(f"Updated news cache with {len(summaries)} new items")
                except Exception as e:
                    self.logger.error(f"Error saving news cache: {e}")

            return sorted(summaries, key=lambda x: x['publishedAt'], reverse=True)

        except Exception as e:
            self.logger.error(f"Error in get_news: {e}")
            return []
    
    def _fetch_rss_feeds(self, max_items=10, extra_sources=None):
        """Fetch news from RSS feeds"""
        rss_feeds = [
            ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
            ("Cointelegraph", "https://cointelegraph.com/rss"),
            ("CryptoNews", "https://cryptonews.com/news/feed"),
            ("Bitcoin.com", "https://news.bitcoin.com/feed/"),
            ("NewsBTC", "https://www.newsbtc.com/feed/"),
            ("CoinGape", "https://coingape.com/feed/"),
            ("CryptoSlate", "https://cryptoslate.com/feed/"),
            ("BeInCrypto", "https://beincrypto.com/feed/"),
            ("U.Today", "https://u.today/rss"),
            ("Decrypt", "https://decrypt.co/feed"),
            ("The Block", "https://www.theblockcrypto.com/rss.xml"),
            ("CoinJournal", "https://coinjournal.net/feed/"),
            ("AMBCrypto", "https://ambcrypto.com/feed/"),
            ("CryptoDaily", "https://cryptodaily.co.uk/feed"),
            ("Bitcoin Magazine", "https://bitcoinmagazine.com/.rss/full/"),
            ("TokenPost", "https://tokenpost.com/rss"),
            ("Crypto Economy", "https://crypto-economy.com/feed/"),
            ("Coinpedia", "https://coinpedia.org/feed/"),
            ("Blockchain News", "https://www.the-blockchain.com/feed/"),
            ("Coin Rivet", "https://coinrivet.com/feed/"),
            ("CryptoNewsZ", "https://www.cryptonewsz.com/feed/"),
            ("CoinStaker", "https://www.coinstaker.com/feed/"),
            ("CryptoSlate Press", "https://cryptoslate.com/press-releases/feed/")
        ]
        
        if extra_sources:
            rss_feeds.extend(extra_sources)

        all_articles = []
        
        if not WEB_SCRAPING_AVAILABLE or not feedparser:
            self.logger.warning("RSS feed parsing not available")
            return pd.DataFrame()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self._fetch_single_feed, name, url, max_items): name 
                      for name, url in rss_feeds}
            
            for future in as_completed(futures):
                feed_name = futures[future]
                try:
                    articles = future.result()
                    if articles:
                        all_articles.extend(articles)
                except Exception as e:
                    self.logger.warning(f"Error fetching from {feed_name}: {e}")

        if all_articles:
            df = pd.DataFrame(all_articles)
            df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
            df = df.dropna(subset=['publishedAt']).sort_values('publishedAt', ascending=False)
            return df.reset_index(drop=True)
        else:
            return pd.DataFrame()
    
    def _fetch_single_feed(self, name, url, max_items):
        """Fetch articles from a single RSS feed"""
        try:
            feed = feedparser.parse(url)
            articles = []
            
            for entry in feed.entries[:max_items]:
                try:
                    # Parse publication date
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                    else:
                        pub_date = datetime.now(timezone.utc)
                    
                    article = {
                        'source': name,
                        'title': entry.get('title', ''),
                        'url': entry.get('link', ''),
                        'summary': entry.get('summary', entry.get('description', '')),
                        'publishedAt': pub_date
                    }
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing entry from {name}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            self.logger.warning(f"Error fetching RSS feed {name}: {e}")
            return []
    
    def _process_news_item(self, row):
        """Process individual news item with summarization, translation, and sentiment"""
        try:
            # Clean HTML content
            cleaned_summary = self._clean_html_content(row.get('summary', ''))
            
            # Summarize content
            summarized_content = self._summarize_text(cleaned_summary)
            
            # Analyze sentiment
            sentiment_score = 0.0
            if self.sentiment_analyzer and row.get('title'):
                try:
                    sentiment_score = self.sentiment_analyzer.polarity_scores(row['title'])['compound']
                except Exception as e:
                    self.logger.warning(f"Sentiment analysis error: {e}")
            
            # Translate to Vietnamese
            title_vi = self._translate_text(row.get('title', ''))
            summary_vi = self._translate_text(summarized_content)
            
            return {
                'source': row.get('source', ''),
                'title': row.get('title', ''),
                'title_vi': title_vi,
                'publishedAt': row.get('publishedAt', ''),
                'url': row.get('url', ''),
                'summary': summarized_content,
                'summary_vi': summary_vi,
                'sentiment': sentiment_score
            }
            
        except Exception as e:
            self.logger.error(f"Error processing news item: {e}")
            return None
    
    # =============== UTILITY METHODS ===============
    
    def _clean_html_content(self, html_content):
        """Clean HTML content from news summaries"""
        if not WEB_SCRAPING_AVAILABLE or not BeautifulSoup:
            return html_content
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(separator=' ').strip()
        except Exception as e:
            self.logger.warning(f"HTML cleaning error: {e}")
            return html_content
    
    def _summarize_text(self, text, max_length=150):
        """Summarize text content"""
        if not NLP_AVAILABLE:
            # Simple truncation if ML not available
            return text[:max_length] + "..." if len(text) > max_length else text
        
        try:
            # Use transformers pipeline if available
            if hasattr(self, '_summarizer'):
                summarizer = self._summarizer
            else:
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                self._summarizer = summarizer
            
            # Ensure text is appropriate length for summarization
            if len(text) < 50:
                return text
            elif len(text) > 1024:
                text = text[:1024]
            
            summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']
            
        except Exception as e:
            self.logger.warning(f"Summarization error: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def _translate_text(self, text, target="vi"):
        """Translate text to target language"""
        if not text or not WEB_SCRAPING_AVAILABLE:
            return text
            
        try:
            if GoogleTranslator:
                translator = GoogleTranslator(source="auto", target=target)
                max_len = 5000
                if len(text) <= max_len:
                    return translator.translate(text)
                else:
                    # Split long text into chunks
                    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
                    translated_chunks = [translator.translate(chunk) for chunk in chunks]
                    return " ".join(translated_chunks)
            else:
                return text
                
        except Exception as e:
            self.logger.warning(f"Translation error: {e}")
            return text

# =============== CONSOLIDATED DATA PROCESSING CLASS ===============

class DataProcessing:
    """
    CONSOLIDATED Data Processing Class - All feature engineering and data transformation
    
    Features consolidated from multiple duplicate implementations:
    - Technical indicator calculation
    - Pattern recognition and labeling
    - Feature engineering and preparation
    - Data validation and cleaning
    - Spike detection and labeling
    
    Removed duplications:
    - Multiple compute_indicators implementations
    - Redundant feature engineering functions
    - Duplicate validation methods
    - Multiple pattern detection functions
    """
    
    def __init__(self):
        """Initialize consolidated data processing"""
        self.logger = get_logger('data_processing')
        self.settings = settings
        self.data_dir = DATA_DIR
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
    # =============== MAIN DATA PROCESSING FUNCTIONS ===============
    
    def load_and_process_data(self, symbol: str, timeframe: str, apply_indicators: bool = True,
                             apply_patterns: bool = True, apply_labels: bool = True,
                             horizon: int = 12, threshold: float = 0.01,
                             force_reload: bool = False) -> Optional[pd.DataFrame]:
        """
        CONSOLIDATED data loading and processing with all preprocessing steps
        """
        try:
            # Create cache file path
            cache_file = os.path.join(self.data_dir, f"{symbol}_{timeframe}_processed.pkl.gz")
            
            if not force_reload and os.path.exists(cache_file):
                try:
                    with gzip.open(cache_file, 'rb') as f:
                        df = pickle.load(f)
                    self.logger.info(f"Loaded processed data from cache: {len(df)} rows")
                    return df
                except Exception as e:
                    self.logger.warning(f"Failed to load cache: {e}")

            # Load raw data using DataCollection
            data_collection = DataCollection()
            df = data_collection.fetch_klines(symbol, timeframe)
            if df is None or df.empty:
                self.logger.error(f"No data available for {symbol} {timeframe}")
                return None

            # Validate and clean data
            df, is_valid = self.validate_data(df, timeframe)
            if not is_valid:
                self.logger.error(f"Data validation failed for {symbol} {timeframe}")
                return None

            # Apply labels if requested
            if apply_labels:
                self.logger.info(f"Applying labels with horizon={horizon}, threshold={threshold}")
                df = self.label_spikes(df, horizon=horizon, threshold=threshold)

            # Apply technical indicators
            if apply_indicators:
                self.logger.info("Computing technical indicators")
                df = self.compute_indicators(df)
                gc.collect()

            # Apply pattern recognition
            if apply_patterns:
                self.logger.info("Detecting chart patterns")
                df = self._apply_pattern_recognition(df)
                gc.collect()

            # Save processed data to cache
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                self.logger.info("Saved processed data to cache")
            except Exception as e:
                self.logger.warning(f"Failed to save cache: {e}")

            return df

        except Exception as e:
            self.logger.error(f"Error in load_and_process_data: {e}", exc_info=True)
            return None
    
    def validate_data(self, df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, bool]:
        """
        CONSOLIDATED data validation and cleaning
        """
        if df is None or df.empty:
            self.logger.warning("Empty dataframe")
            return df, False
            
        df = df.copy()
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return df, False
        
        # Handle timestamp column
        if 'timestamp' not in df.columns:
            if 'close_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['close_time'], unit='ms', utc=True, errors='coerce')
            else:
                self.logger.error("Missing timestamp and close_time columns")
                return df, False
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

        # Clean data
        df.dropna(subset=['timestamp'], inplace=True)
        df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        if df.empty:
            return df, False

        # Remove rows with invalid data
        df = df.dropna(subset=required_cols)
        df = df[(df['high'] >= df['low']) & (df['high'] >= df['open']) & 
                (df['high'] >= df['close']) & (df['low'] <= df['open']) & 
                (df['low'] <= df['close']) & (df['volume'] >= 0)]
        
        # Filter positive prices
        for col in required_cols[:-1]:  # exclude volume
            if col in df.columns:
                df = df[df[col] > 0]
        
        if len(df) < 100:
            self.logger.warning(f"Insufficient data after cleaning: {len(df)} rows")
            return df, False
            
        return df, True
    
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CONSOLIDATED technical indicators computation using FeatureEngineer
        """
        # Validate required columns
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise ValueError(f"Missing column {col}")

        # Sort by time if available
        if "close_time" in df.columns:
            df = df.sort_values("close_time").reset_index(drop=True)
        elif "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        # Apply feature engineering
        try:
            features = self.feature_engineer.transform(df)
            
            # Merge features with original data
            common_cols = df.columns.intersection(features.columns)
            feature_only_cols = features.columns.difference(common_cols)
            
            result = df.copy()
            for col in feature_only_cols:
                result[col] = features[col]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error computing indicators: {e}")
            return df
            self.logger.warning(f"Lỗi HTTP {resp.status_code} khi lấy {key}: {resp.text}")
                continue

                text = resp.text.strip()
                if text == "":
                        metrics[key] = None
                    else:
                        if key in ["tx_count_24h", "mempool_tx"]:
                            metrics[key] = int(float(text))
                        elif key in ["btc_sent_24h", "total_btc"]:
                            val = float(text)
                            metrics[key] = val / 1e8  # satoshi -> BTC
                        else:
                            try:
                                metrics[key] = float(text)
                            except ValueError:
                                metrics[key] = text
                except Exception as e:
                    metrics[key] = None
                    self.logger.warning(f"Lỗi kết nối khi lấy {key}: {e}")

        except Exception as e:
            self.logger.error("Lỗi chung khi lấy on-chain data:", e)

        return metrics

    # Order Book Analysis Methods
    def analyze_orderbook(self, symbol="BTCUSDT"):
        """
        Trả về dict với: best_bid, best_ask, spread, spread_percent,
        bid_volume, ask_volume, imbalance.
        """
        result = {}
        try:
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=50"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in data.get("asks", [])]

            if not bids or not asks:
                return result

            best_bid_price, best_bid_qty = max(bids, key=lambda x: x[0])
            best_ask_price, best_ask_qty = min(asks, key=lambda x: x[0])
            spread = best_ask_price - best_bid_price
            mid_price = (best_ask_price + best_bid_price) / 2.0
            total_bid_vol = sum(q for _, q in bids)
            total_ask_vol = sum(q for _, q in asks)
            imbalance = 0.0
            if total_bid_vol + total_ask_vol:
                imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

            result = {
                "best_bid": best_bid_price,
                "best_ask": best_ask_price,
                "spread": spread,
                "spread_percent": (spread / mid_price * 100.0) if mid_price else None,
                "bid_volume": total_bid_vol,
                "ask_volume": total_ask_vol,
                "imbalance": imbalance
            }

        except Exception as e:
            self.logger.error(f"Lỗi khi lấy order book cho {symbol}: {e}")

        return result

    # Intermarket Analysis Methods
    def get_intermarket_data(self):
        """
        Lấy giá và biến động (%) cho SPY (S&P500), GLD (vàng), DX=F (DXY).
        Trả về dict với keys: sp500_price, sp500_change_pct, gold_price, gold_change_pct, dxy_price, dxy_change_pct.
        """
        data = {}
        symbols = {
            "sp500": "SPY",
            "gold":  "GLD",
            "dxy":   "DX=F"
        }

        try:
            for key, symbol in symbols.items():
                url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
                resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code != 200:
                    self.logger.warning(f"Lỗi Yahoo Finance API cho {symbol}: HTTP {resp.status_code}")
                    continue

                result = resp.json().get("quoteResponse", {}).get("result", [])
                if result:
                    item = result[0]
                    price = item.get("regularMarketPrice")
                    change_pct = item.get("regularMarketChangePercent")
                    data[f"{key}_price"] = price
                    data[f"{key}_change_percent"] = change_pct

        except Exception as e:
            self.logger.error("Lỗi khi lấy dữ liệu liên thị trường:", e)

        return data

class NewsAnalysis:
    """
    News Analysis Class - News collection, summarization, and sentiment analysis
    
    Handles all news-related operations including:
    - RSS feed collection
    - News summarization
    - Sentiment analysis of news
    - Translation of news content
    """
    
    def __init__(self):
        """Initialize news analysis with global instances"""
        self.logger = get_logger('news_analysis')
        self.news_cache_path = os.path.join(DATA_DIR, "news_cache.csv")
        
    def load_summarizer(self):
        """Load the summarization model"""
        return pipeline("summarization", model="facebook/bart-large-cnn", framework='pt')
        
    def translate_text(self, text, target="vi"):
        """Translate text to target language"""
        try:
            translator = GoogleTranslator(source="auto", target=target)
            max_len = 5000
            if len(text) <= max_len:
                return translator.translate(text)
            chunks = text.wrap(text, max_len)
            translated = [translator.translate(chunk) for chunk in chunks]
            return "".join(translated)
        except Exception as e:
            self.logger.warning(f"Lỗi dịch văn bản: {e}")
            return text
            
    def clean_html_content(self, html_content):
        """Clean HTML content from news summaries"""
        if not BeautifulSoup:
            return html_content
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(separator=' ')
        except Exception as e:
            self.logger.warning(f"Lỗi khi xử lý HTML: {e}")
            return html_content
            
    def get_news(self, symbol, max_items=10, extra_sources=None):
        """Get news about a cryptocurrency"""
        # 1) Đường dẫn cache
        # 2) Load cache cũ nếu có, xác định thời điểm mới nhất
        if os.path.exists(self.news_cache_path):
            cached = pd.read_csv(self.news_cache_path, parse_dates=['publishedAt'])
            last_cached = cached['publishedAt'].max()
        else:
            cached = pd.DataFrame()
            last_cached = None   
        news_source = NewsDataSource(max_items=max_items, sources=extra_sources)
        summarizer = NewsSummarizer()

        try:
            news_df = news_source.fetch()
        except Exception as e:
            self.logger.error(f"Lỗi lấy tin tức RSS: {e}")
            return []
        # 3) Chỉ giữ những bản tin mới hơn last_cached
        if last_cached is not None:
            news_df = news_df[news_df['publishedAt'] > last_cached]
        # 4) Lọc theo keyword symbol
        symbol_keyword = symbol.replace("USDT", "")
        news_df_filtered = news_df[news_df['title'].str.contains(symbol_keyword, case=False)]

        summaries = []
        if not news_df_filtered.empty:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = {}
                for _, row in news_df_filtered.iterrows():
                    cleaned_summary = self.clean_html_content(row.get('summary', ''))
                    futures[executor.submit(summarizer.summarize, cleaned_summary)] = row

                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        summarized_content = future.result()
                        sentiment_score = analyzer.polarity_scores(row['title'])['compound'] if analyzer else 0.0

                        # Dịch sang tiếng Việt
                        title_vi = self.translate_text(row.get('title', ''))
                        summary_vi = self.translate_text(summarized_content)

                        summaries.append({
                            'source': row.get('source', ''),
                            'title': row.get('title', ''),
                            'title_vi': title_vi,
                            'publishedAt': row.get('publishedAt', ''),
                            'url': row.get('url', ''),
                            'summary': summarized_content,
                            'summary_vi': summary_vi,
                            'sentiment': sentiment_score
                        })
                    except Exception as e:
                        self.logger.error(f"Lỗi xử lý tin: {e}")

        # Lưu cache nếu cần
        # 5) Ghép vào cache cũ và lưu lại
        all_summaries = (
            pd.concat([cached, pd.DataFrame(summaries)], ignore_index=True)
            .drop_duplicates(subset=['url'])
            .sort_values('publishedAt', ascending=False)
        )
        all_summaries.to_csv(self.news_cache_path, index=False)

        # 6) Trả về chỉ danh sách tin mới (nếu muốn), hoặc .to_dict(orient='records') để trả toàn bộ
        return sorted(summaries, key=lambda x: x['publishedAt'], reverse=True)
        """Initialize market analysis with global instances"""
        self.logger = get_logger('market_analysis')
    
    # Sentiment Analysis Methods
    def get_market_sentiment(self):
        """
        Lấy chỉ số tâm lý thị trường crypto (ví dụ: Crypto Fear & Greed Index).
        Trả về dict chứa 'fear_greed_index' (int) và 'classification' (str).
        """
        try:
            url = "https://api.alternative.me/fng/?limit=1&format=json"
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                self.logger.error(f"Lỗi API Fear & Greed (HTTP {response.status_code}): {response.text}")
                return {"fear_greed_index": None, "classification": None}

            data = response.json()
            if data.get("data"):
                entry = data["data"][0]
                value = int(entry.get("value", 0))
                classification = entry.get("value_classification", "")
                return {"fear_greed_index": value, "classification": classification}
            else:
                return {"fear_greed_index": None, "classification": None}

        except Exception as e:
            self.logger.error("Lỗi khi lấy chỉ số Fear & Greed:", e)
            return {"fear_greed_index": None, "classification": None}

    # On-chain Analysis Methods
    def get_onchain_data(self):
        """
        Lấy một số thống kê on-chain của Bitcoin qua blockchain.info.
        Trả về dict với các chỉ số: tx_count_24h, btc_sent_24h, mempool_tx, hash_rate, market_cap_usd, btc_price, total_btc.
        """
        metrics = {}
        try:
            urls = {
                "tx_count_24h": "https://blockchain.info/q/24hrtransactioncount",
                "btc_sent_24h":  "https://blockchain.info/q/24hrbtcsent",
                "mempool_tx":    "https://blockchain.info/q/unconfirmedcount",
                "hash_rate":     "https://blockchain.info/q/hashrate",
                "market_cap_usd":"https://blockchain.info/q/marketcap",
                "btc_price":     "https://blockchain.info/q/24hrprice",
                "total_btc":     "https://blockchain.info/q/totalbc"
            }

            for key, url in urls.items():
                try:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code != 200:
                        metrics[key] = None
                        self.logger.warning(f"Lỗi HTTP {resp.status_code} khi lấy {key}: {resp.text}")
                        continue

                    text = resp.text.strip()
                    if text == "":
                        metrics[key] = None
                    else:
                        if key in ["tx_count_24h", "mempool_tx"]:
                            metrics[key] = int(float(text))
                        elif key in ["btc_sent_24h", "total_btc"]:
                            val = float(text)
                            metrics[key] = val / 1e8  # satoshi -> BTC
                        else:
                            try:
                                metrics[key] = float(text)
                            except ValueError:
                                metrics[key] = text
                except Exception as e:
                    metrics[key] = None
                    self.logger.warning(f"Lỗi kết nối khi lấy {key}: {e}")

        except Exception as e:
            self.logger.error("Lỗi chung khi lấy on-chain data:", e)

        return metrics

    # Order Book Analysis Methods
    def analyze_orderbook(self, symbol="BTCUSDT"):
        """
        Trả về dict với: best_bid, best_ask, spread, spread_percent,
        bid_volume, ask_volume, imbalance.
        """
        result = {}
        try:
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=50"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in data.get("asks", [])]

            if not bids or not asks:
                return result

            best_bid_price, best_bid_qty = max(bids, key=lambda x: x[0])
            best_ask_price, best_ask_qty = min(asks, key=lambda x: x[0])
            spread = best_ask_price - best_bid_price
            mid_price = (best_ask_price + best_bid_price) / 2.0
            total_bid_vol = sum(q for _, q in bids)
            total_ask_vol = sum(q for _, q in asks)
            imbalance = 0.0
            if total_bid_vol + total_ask_vol:
                imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

            result = {
                "best_bid": best_bid_price,
                "best_ask": best_ask_price,
                "spread": spread,
                "spread_percent": (spread / mid_price * 100.0) if mid_price else None,
                "bid_volume": total_bid_vol,
                "ask_volume": total_ask_vol,
                "imbalance": imbalance
            }

        except Exception as e:
            self.logger.error(f"Lỗi khi lấy order book cho {symbol}: {e}")

        return result

    # Intermarket Analysis Methods
    def get_intermarket_data(self):
        """
        Lấy giá và biến động (%) cho SPY (S&P500), GLD (vàng), DX=F (DXY).
        Trả về dict với keys: sp500_price, sp500_change_pct, gold_price, gold_change_pct, dxy_price, dxy_change_pct.
        """
        data = {}
        symbols = {
            "sp500": "SPY",
            "gold":  "GLD",
            "dxy":   "DX=F"
        }

        try:
            for key, symbol in symbols.items():
                url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
                resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code != 200:
                    self.logger.warning(f"Lỗi Yahoo Finance API cho {symbol}: HTTP {resp.status_code}")
                    continue

                result = resp.json().get("quoteResponse", {}).get("result", [])
                if result:
                    item = result[0]
                    price = item.get("regularMarketPrice")
                    change_pct = item.get("regularMarketChangePercent")
                    data[f"{key}_price"] = price
                    data[f"{key}_change_percent"] = change_pct

        except Exception as e:
            self.logger.error("Lỗi khi lấy dữ liệu liên thị trường:", e)

        return data
    
    def __init__(self):
        self.logger = get_logger('data_collection')
        self.base_url = BINANCE_BASE_URL
        self.data_dir = DATA_DIR
        
    # 1. Fetch data from external sources    # fetch_klines method moved to DataCollection class - use data_collection.fetch_klines() instead
          # fetch_new_data_from_binance method moved to DataCollection class - use data_collection.fetch_new_data_from_binance() instead
    
    # 2. Save/Load data utilities
    def save_compressed(self, data, file_path):
        """Bước 2: Lưu dữ liệu nén"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load_compressed(self, file_path):
        """Bước 3: Tải dữ liệu đã nén"""
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    
    # 3. Main data update function
    def update_data(self, symbol: str, timeframe: str = None, deep_train: bool = False) -> pd.DataFrame:
        """Update data for a symbol and timeframe"""
        self.ensure_data_dir()
        symbol = symbol.upper()
        quote = getattr(self.settings, "DEFAULT_QUOTE_BINANCE", "USDT")
        if not symbol.endswith(quote):
            symbol = f"{symbol}{quote}"

        intervals = [timeframe] if timeframe else list(self.settings.HISTORY_REQUIREMENTS.keys())
        result_frames = []

        for tf in intervals:
            self.logger.info(f"[update_data] Bắt đầu cập nhật {symbol}@{tf}")
            path = os.path.join(self.data_dir, f"{symbol}@{tf}.pkl.gz")

            # —————— LOAD DỮ LIỆU CŨ & TÍNH THỜI ĐIỂM BẮT ĐẦU ——————  
            df_old = pd.DataFrame()  
            start_time = None  
            if os.path.isfile(path) and not deep_train:  
                try:  
                    with gzip.open(path, "rb") as f:  
                        df_old = pickle.load(f)  
                    
                    if not df_old.empty and "close_time" in df_old.columns:  
                        last_ts = pd.to_datetime(df_old["close_time"].iloc[-1], utc=True)  
                        # đủ thời gian → bắt đầu fetch từ last_ts + 1ms  
                        start_time = int(last_ts.timestamp() * 1000) + 1  
                        self.logger.info(f"[update_data] Đọc {len(df_old)} dòng, tải từ {last_ts}")  
                        df_new = self.fetch_klines(symbol, tf, start_time)  
                except Exception as e:  
                    self.logger.warning(f"[update_data] Lỗi đọc khung {tf} tại {path}: {e}, sẽ tải full lịch sử.")

                    try: os.remove(path)  
                    except: pass  
                    df_new = self.fetch_klines(symbol, tf, None)  
            else:  
                # lần đầu hoặc deep_train → tải theo HISTORY_REQUIREMENTS  
                days = self.settings.HISTORY_REQUIREMENTS.get(tf, 30 if not deep_train else self.settings.HISTORY_REQUIREMENTS.get(tf,30))  
                start_dt = datetime.now(timezone.utc) - timedelta(days=days)  
                start_time = int(start_dt.timestamp() * 1000)  
                self.logger.info(f"[update_data] {'Deep train' if deep_train else 'Mới'} tải từ {start_dt}")  
                df_new = self.fetch_klines(symbol, tf, start_time)

            if df_new.empty:
                self.logger.info(f"[update_data] {symbol}@{tf}: Không có dữ liệu mới.")
                df_tf = df_old.copy()
            else:
                if not df_old.empty and not deep_train:
                    df_tf = pd.concat([df_old, df_new], ignore_index=True)\
                            .drop_duplicates(subset="close_time")\
                            .sort_values("close_time")
                else:
                    df_tf = df_new.copy()

                try:
                    with gzip.open(path, "wb") as f:
                        pickle.dump(df_tf, f)
                    self.logger.info(f"[update_data] Lưu {len(df_tf)} dòng vào {path}")
                except Exception as e:
                    self.logger.error(f"[update_data] Lỗi lưu pickle {path}: {e}")

            result_frames.append(df_tf)
            self.logger.info(f"[update_data] Hoàn thành {symbol}@{tf}")

        # Kết hợp tất cả timeframes
        combined = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
        
        try:
            combined.to_csv(os.path.join(self.data_dir, f"{symbol}_combined.csv"), index=False)
            self.logger.info(f"[update_data] Đã lưu CSV tổng hợp cho {symbol}")
        except Exception as e:
            self.logger.error(f"[update_data] Lỗi lưu CSV tổng hợp: {e}")

        self.logger.info(f"[update_data] Hoàn tất cập nhật dữ liệu {symbol}")
        return combined
        """Bước 4: Cập nhật dữ liệu chính - function chính của DataCollection"""
        # Implementation for updating data
        pass    # 4. News data collection - methods should be moved to NewsAnalysis class later
        """Lấy, tóm tắt, phân tích cảm xúc và dịch tin tức từ RSS về coin"""

        # 1) Đường dẫn cache
        news_cache_path = os.path.join("data", "news_cache.csv")
        # 2) Load cache cũ nếu có, xác định thời điểm mới nhất
        if os.path.exists(news_cache_path):
            cached = pd.read_csv(news_cache_path, parse_dates=['publishedAt'])
            last_cached = cached['publishedAt'].max()
        else:
            cached = pd.DataFrame()
            last_cached = None   
        news_source = NewsDataSource(max_items=max_items, sources=extra_sources)
        summarizer = NewsSummarizer()

        try:
            news_df = news_source.fetch()
        except Exception as e:
            logging.error(f"Lỗi lấy tin tức RSS: {e}")
            return []
        # 3) Chỉ giữ những bản tin mới hơn last_cached
        if last_cached is not None:
            news_df = news_df[news_df['publishedAt'] > last_cached]
        # 4) Lọc theo keyword symbol
        symbol_keyword = symbol.replace("USDT", "")
        news_df_filtered = news_df[news_df['title'].str.contains(symbol_keyword, case=False)]

        summaries = []
        if not news_df_filtered.empty:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = {}
                for _, row in news_df_filtered.iterrows():
                    cleaned_summary = clean_html_content(row.get('summary', ''))
                    futures[executor.submit(summarizer.summarize, cleaned_summary)] = row

                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        summarized_content = future.result()
                        sentiment_score = analyzer.polarity_scores(row['title'])['compound'] if analyzer else 0.0

                        # Dịch sang tiếng Việt
                        title_vi = translate_text(row.get('title', ''))
                        summary_vi = translate_text(summarized_content)

                        summaries.append({
                            'source': row.get('source', ''),
                            'title': row.get('title', ''),
                            'title_vi': title_vi,
                            'publishedAt': row.get('publishedAt', ''),
                            'url': row.get('url', ''),
                            'summary': summarized_content,
                            'summary_vi': summary_vi,
                            'sentiment': sentiment_score
                        })
                    except Exception as e:
                        logging.error(f"Lỗi xử lý tin: {e}")

        # Lưu cache nếu cần
        # 5) Ghép vào cache cũ và lưu lại
        all_summaries = (
            pd.concat([cached, pd.DataFrame(summaries)], ignore_index=True)
            .drop_duplicates(subset=['url'])
            .sort_values('publishedAt', ascending=False)
        )
        all_summaries.to_csv(news_cache_path, index=False)

        # 6) Trả về chỉ danh sách tin mới (nếu muốn), hoặc .to_dict(orient='records') để trả toàn bộ
        return sorted(summaries, key=lambda x: x['publishedAt'], reverse=True)
    # -*- coding: utf-8 -*-



    # Thư viện snscrape để lấy tin Twitter (nếu chưa có thì bỏ qua)
    sntwitter = None
    # Market analysis methods - should be moved to MarketAnalysis class later
    # Sentiment, on-chain, orderbook, and intermarket analysis functions need to be reorganized    # Removed: load_summarizer function moved to NewsAnalysis class
    
    # Removed: translate_text function moved to NewsAnalysis class

    def __init__(self):
        """Initialize data collection with global instances"""
        self.settings = settings
        self.data_dir = DATA_DIR
        self.logger = get_logger('data_collection')
        
        # News data source initialization
        self.news_data_source = None
        self.news_cache_path = os.path.join(self.data_dir, "news_cache.csv")
        
        # Exchange data configuration
        self.binance_base_url = BINANCE_BASE_URL
        self.default_symbol = DEFAULT_symbol
        self.default_quote = DEFAULT_QUOTE_BINANCE
        self.timeout = DEFAULT_TIMEOUT
          # fetch_klines method moved to DataCollection class - use data_collection.fetch_klines() instead
    
    # fetch_new_data_from_binance method moved to DataCollection class - use data_collection.fetch_new_data_from_binance() instead
    
    def update_data(self, symbol: str, timeframe: str = None, deep_train: bool = False) -> pd.DataFrame:
        """Update data for a symbol and timeframe"""
        self.ensure_data_dir()
        symbol = symbol.upper()
        quote = getattr(self.settings, "DEFAULT_QUOTE_BINANCE", "USDT")
        if not symbol.endswith(quote):
            symbol = f"{symbol}{quote}"

        intervals = [timeframe] if timeframe else list(HISTORY_REQUIREMENTS.keys())
        result_frames = []

        for tf in intervals:
            self.logger.info(f"[update_data] Bắt đầu cập nhật {symbol}@{tf}")
            path = os.path.join(self.data_dir, f"{symbol}@{tf}.pkl.gz")

            # —————— LOAD DỮ LIỆU CŨ & TÍNH THỜI ĐIỂM BẮT ĐẦU ——————  
            df_old = pd.DataFrame()  
            start_time = None  
            if os.path.isfile(path) and not deep_train:  
                try:  
                    with gzip.open(path, "rb") as f:  
                        df_old = pickle.load(f)  
                    
                    if not df_old.empty and "close_time" in df_old.columns:  
                        last_ts = pd.to_datetime(df_old["close_time"].iloc[-1], utc=True)  
                        # đủ thời gian → bắt đầu fetch từ last_ts + 1ms  
                        start_time = int(last_ts.timestamp() * 1000) + 1  
                        self.logger.info(f"[update_data] Đọc {len(df_old)} dòng, tải từ {last_ts}")  
                        df_new = self.fetch_klines(symbol, tf, start_time)  
                except Exception as e:  
                    self.logger.warning(f"[update_data] Lỗi đọc khung {tf} tại {path}: {e}, sẽ tải full lịch sử.")

                    try: os.remove(path)  
                    except: pass  
                    df_new = self.fetch_klines(symbol, tf, None)  
            else:  
                # lần đầu hoặc deep_train → tải theo HISTORY_REQUIREMENTS  
                days = HISTORY_REQUIREMENTS.get(tf, 30 if not deep_train else self.settings.HISTORY_REQUIREMENTS.get(tf,30))  
                start_dt = datetime.now(timezone.utc) - timedelta(days=days)  
                start_time = int(start_dt.timestamp() * 1000)  
                self.logger.info(f"[update_data] {'Deep train' if deep_train else 'Mới'} tải từ {start_dt}")  
                df_new = self.fetch_klines(symbol, tf, start_time)

            if df_new.empty:
                self.logger.info(f"[update_data] {symbol}@{tf}: Không có dữ liệu mới.")
                df_tf = df_old.copy()
            else:
                if not df_old.empty and not deep_train:
                    df_tf = pd.concat([df_old, df_new], ignore_index=True)\
                            .drop_duplicates(subset="close_time")\
                            .sort_values("close_time")
                else:
                    df_tf = df_new.copy()

                try:
                    with gzip.open(path, "wb") as f:
                        pickle.dump(df_tf, f)
                    self.logger.info(f"[update_data] Lưu {len(df_tf)} dòng vào {path}")
                except Exception as e:
                    self.logger.error(f"[update_data] Lỗi lưu pickle {path}: {e}")

            result_frames.append(df_tf)
            self.logger.info(f"[update_data] Hoàn thành {symbol}@{tf}")

        # Kết hợp tất cả timeframes
        combined = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
        
        try:
            combined.to_csv(os.path.join(self.data_dir, f"{symbol}_combined.csv"), index=False)
            self.logger.info(f"[update_data] Đã lưu CSV tổng hợp cho {symbol}")
        except Exception as e:
            self.logger.error(f"[update_data] Lỗi lưu CSV tổng hợp: {e}")

        self.logger.info(f"[update_data] Hoàn tất cập nhật dữ liệu {symbol}")
        return combined

    def fetch_candles(self, symbol, interval, start_time=None, end_time=None) -> pd.DataFrame:
        """
        Lấy dữ liệu nến từ Binance (tối đa 1000 nến mỗi lần gọi).
        Trả về DataFrame với các cột open, high, low, close, volume (index thời gian UTC).
        """
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": 1000
        }
        if start_time:
            params["startTime"] = int(start_time)
        if end_time:
            params["endTime"] = int(end_time)

        try:
            resp = requests.get(f"{self.binance_base_url}/api/v3/klines", params=params, timeout=10)
            if resp.status_code != 200:
                self.logger.error(f"Lỗi Binance API (fetch_candles): {resp.status_code} - {resp.text}")
                return pd.DataFrame()
            data = resp.json()
        except Exception as e:
            self.logger.error(f"Lỗi kết nối Binance: {e}")
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "buy_volume", "buy_quote", "ignore"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df.set_index("time", inplace=True)
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    def fetch_binance_data(self, symbol: str = None, quote: str = None, interval: str = "1h",
                           start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        Tải toàn bộ dữ liệu lịch sử từ Binance cho cặp (symbol, quote) và khung thời gian `interval`.
        Tự động lặp nhiều lần để lấy đủ dữ liệu từ `start_time` đến `end_time`.
        Trả về DataFrame OHLCV (index thời gian UTC).
        """
        symbol = symbol or self.default_symbol
        quote = quote or self.default_quote
        pair = f"{symbol.upper()}{quote.upper()}"
        endpoint = f"{self.binance_base_url}/api/v3/klines"
        params = {
            "symbol": pair,
            "interval": interval,
            "limit": 1000
        }
        if start_time:
            params["startTime"] = int(start_time)
        if end_time:
            params["endTime"] = int(end_time)

        all_data = []
        incomplete = False

        while True:
            try:
                resp = requests.get(endpoint, params=params, timeout=10)
            except Exception as e:
                self.logger.error(f"Lỗi kết nối Binance: {e}")
                incomplete = True
                break
            if resp.status_code != 200:
                self.logger.error(f"Lỗi Binance API: {resp.status_code} - {resp.text}")
                incomplete = True
                break

            data = resp.json()
            if not data:
                break

            # Lưu dữ liệu vào DataFrame tạm và thêm vào danh sách
            df_batch = pd.DataFrame(data, columns=[
                "time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_volume", "taker_buy_quote_volume", "ignore"
            ])
            df_batch["time"] = pd.to_datetime(df_batch["time"], unit="ms", utc=True)
            df_batch.set_index("time", inplace=True)
            df_batch = df_batch[["open", "high", "low", "close", "volume"]].astype(float)
            all_data.append(df_batch)

            # Nếu số bản ghi trả về ít hơn limit, nghĩa là đã tới cuối khoảng dữ liệu
            if len(data) < params["limit"]:
                break

            # Cập nhật startTime cho lần gọi kế tiếp (tiếp tục sau thời điểm cuối cùng đã nhận)
            last_time = int(data[-1][0])
            params["startTime"] = last_time + 1  # +1ms để tránh trùng lặp
            time.sleep(0.2)  # nghỉ một chút để tránh vượt giới hạn tần suất

        # Kết thúc vòng lặp
        if not all_data or incomplete:
            # Nếu không thu được dữ liệu hoặc xảy ra lỗi, trả về DataFrame rỗng
            return pd.DataFrame()

        # Kết hợp tất cả các batch và sắp xếp theo thời gian tăng dần
        full_df = pd.concat(all_data).sort_index()
        return full_df

    def get_historical_data(self, symbol: str = None, interval: str = "1h",
                            start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        Hàm hỗ trợ lấy dữ liệu lịch sử Binance theo ký hiệu `symbol` (có thể bao gồm quote, ví dụ "BTCUSDT").
        Tự động tách base và quote. Trả về DataFrame kết quả giống fetch_binance_data.
        """
        symbol = (symbol or self.default_symbol).upper()
        quote = self.default_quote
        # Tách base và quote nếu symbol bao gồm quote
        if symbol.endswith("USDT") or symbol.endswith("BUSD") or symbol.endswith("USDC"):
            # Nếu symbol đã bao gồm một trong các quote phổ biến
            for q in ["USDT", "BUSD", "USDC"]:
                if symbol.endswith(q):
                    quote = q
                    base = symbol[:-len(q)]
                    break
        elif "-" in symbol:
            # Trường hợp ký hiệu dạng "BASE-QUOTE"
            base, quote = symbol.split("-", 1)
        else:
            # Mặc định sử dụng quote là USDT nếu không xác định được
            base = symbol
        return self.fetch_binance_data(base, quote, interval, start_time, end_time)

    def analyze_historical_events(self, symbol=None):
        """Analyze historical events impact on price"""
        # ...existing implementation will be moved here...    # Functions moved to MarketAnalysis class - these should be accessed via market_analysis instance
    # Functions moved to NewsAnalysis class - these should be accessed via news_analysis instance
                ("TokenPost", "https://tokenpost.com/rss"),
                ("Crypto Economy", "https://crypto-economy.com/feed/"),
                ("Coinpedia", "https://coinpedia.org/feed/"),
                ("Blockchain News", "https://www.the-blockchain.com/feed/"),
                ("Coin Rivet", "https://coinrivet.com/feed/"),
                ("CryptoNewsZ", "https://www.cryptonewsz.com/feed/"),
                ("CoinStaker", "https://www.coinstaker.com/feed/"),
                ("CryptoSlate (Press Releases)", "https://cryptoslate.com/press-releases/feed/")          
            ]
                

        
            self.sources = [RSSSource(name, url, max_items) for name, url in rss_feeds]

        def fetch(self) -> pd.DataFrame:
            dfs = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(source.fetch): source for source in self.sources}
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            dfs.append(df)
                    except Exception as e:
                        logging.error(f"Lỗi lấy tin từ {source.source_name}: {e}")

            if dfs:
                return pd.concat(dfs).sort_values("publishedAt", ascending=False).reset_index(drop=True)
            else:
                raise RuntimeError("Không lấy được tin tức từ bất kỳ nguồn nào.")
    logger = get_logger('binance_api')
      # Duplicate wrapper functions removed - use class instances directly
    
    # ===================== BIẾN ANALYZER TOÀN CỤC =====================
    try:
        analyzer = SentimentIntensityAnalyzer()
    except Exception as e:
        logger.error(f"Lỗi khởi tạo Sentiment Analyzer: {e}")
        analyzer = None
    
    # ===================== PLACEHOLDER CLASSES =====================
    # Các class khác sẽ được implement sau:
    # class DataProcessing: ...
    # class FeatureEngineer: ...
    # class ModelTrainer: ...
    # class TradingStrategy: ...

            # —————— LOAD DỮ LIỆU CŨ & TÍNH THỜI ĐIỂM BẮT ĐẦU ——————  
            df_old = pd.DataFrame()  
            start_time = None  
            if os.path.isfile(path) and not deep_train:  
                try:  
                    with gzip.open(path, "rb") as f:  
                        df_old = pickle.load(f)  
                    # chuẩn hóa cột timestamp  
                    df_old["timestamp"] = pd.to_datetime(  
                        df_old.get("timestamp", df_old.get("close_time")), utc=True  
                    )  
                    last_ts = df_old["timestamp"].max()  
                    # tính delay = 1 nến của timeframe  
                    qty, unit = int(tf[:-1]), tf[-1]  
                    wait = timedelta(minutes=qty) if unit=="m" else timedelta(hours=qty)  
                    now = datetime.now(timezone.utc)  
                    # nếu chưa đến giờ nến mới → bỏ fetch, dùng luôn df_old  
                    if now < last_ts + wait:  
                        logger.info(  
                            f"[update_data] {symbol}@{tf}: chưa đủ {tf} kể từ {last_ts}, bỏ fetch API."  
                        )  
                        df_new = pd.DataFrame()  
                    else:  
                        # đủ thời gian → bắt đầu fetch từ last_ts + 1ms  
                        start_time = int(last_ts.timestamp() * 1000) + 1  
                        logger.info(f"[update_data] Đọc {len(df_old)} dòng, tải từ {last_ts}")  
                        df_new = fetch_klines(symbol, tf, start_time)  
                except Exception as e:  
                    logger.warning(f"[update_data] Lỗi đọc khung {tf} tại {path}: {e}, sẽ tải full lịch sử.")

                    try: os.remove(path)  
                    except: pass  
                    df_new = fetch_klines(symbol, tf, None)  
            else:  
                # lần đầu hoặc deep_train → tải theo HISTORY_REQUIREMENTS  
                days = HISTORY_REQUIREMENTS.get(tf, 30 if not deep_train else settings.HISTORY_REQUIREMENTS.get(tf,30))  
                start_dt = datetime.now(timezone.utc) - timedelta(days=days)  
                start_time = int(start_dt.timestamp() * 1000)  
                logger.info(f"[update_data] {'Deep train' if deep_train else 'Mới'} tải từ {start_dt}")  
                df_new = fetch_klines(symbol, tf, start_time)

            if df_new.empty:
                logger.info(f"[update_data] {symbol}@{tf}: Không có dữ liệu mới.")
                df_tf = df_old.copy()
            else:
                if not df_old.empty and not deep_train:
                    df_tf = pd.concat([df_old, df_new], ignore_index=True)\
                            .drop_duplicates(subset="close_time")\
                            .sort_values("close_time")
                else:
                    df_tf = df_new.copy()

                try:
                    with gzip.open(path, "wb") as f:
                        pickle.dump(df_tf, f)
                    logger.info(f"[update_data] Lưu {len(df_tf)} dòng vào {path}")
                except Exception as e:
                    logger.error(f"[update_data] Lỗi khi lưu dữ liệu vào {path}: {e}")

            df_ai = df_tf[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            df_valid, _ = validate_data(df_ai, tf)
            if df_valid.empty:
                logger.warning(f"[update_data] {symbol}@{tf}: Rỗng sau validate, bỏ qua.")
                continue

            df_valid["timeframe"] = tf
            result_frames.append(df_valid)

        if not result_frames:
            logger.warning(f"[update_data] Không có khung nào cập nhật thành công cho {symbol}")
            return pd.DataFrame()

        combined = pd.concat(result_frames)
        combined.reset_index(drop=True, inplace=True)

        try:
            combined.to_csv(os.path.join(DATA_DIR, f"{symbol}_combined.csv"), index=False)
            logger.info(f"[update_data] Đã lưu CSV tổng hợp cho {symbol}")
        except Exception as e:
            logger.error(f"[update_data] Lỗi lưu CSV tổng hợp: {e}")
        logger.info(f"[update_data] Hoàn tất cập nhật dữ liệu {symbol}")
        return combined

    """
    Hàm helper để load và xử lý dữ liệu một cách đồng bộ, đảm bảo rằng
    tất cả các bước xử lý (indicators, pattern recognition, labeling, etc.)
    được áp dụng ngay khi dữ liệu được load.
    """
    
    def fetch_candles(symbol, interval, start_time=None, end_time=None) -> pd.DataFrame:
        """Wrapper function - redirects to DataCollection.fetch_candles()"""
        data_collection = DataCollection()
        return data_collection.fetch_candles(symbol, interval, start_time, end_time)

    def get_historical_data(symbol: str = DEFAULT_symbol, interval: str = "1h",
                            start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """Wrapper function - redirects to DataCollection.get_historical_data()"""
        data_collection = DataCollection()
        return data_collection.get_historical_data(symbol, interval, start_time, end_time)

    logger = get_logger('data_processor')

    # data_processing/data_loader.py (chỉ sửa hàm update_data)    # Moved to DataCollection.update_data() method    # Moved to DataProcessing.load_and_process_data() method
    
    # Data validation functions
    # Moved to DataProcessing.validate_data() methoddef validate_data(df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, bool]:
        """Wrapper function - redirects to DataProcessing.validate_data()"""
        data_processing = DataProcessing()
        return data_processing.validate_data(df, timeframe)
    # File: AI_Crypto_Project/data_processing/event_labeler.py
    # Nhiệm vụ: Gắn nhãn sự kiện “spike” trên dữ liệu OHLCV để AI training



    # Moved to DataProcessing.label_spikes() method  
    d

    def collect_spike_events(symbols=None,
                            timeframe: str = '15m',
                            pct_threshold: float = 0.05) -> pd.DataFrame:
        """
        - Input:
        • symbols: str hoặc list[str]; nếu None thì lấy [DEFAULT_symbol]
        • timeframe: khung thời gian (VD '15m')
        • pct_threshold: ngưỡng spike (VD 0.05 = 5%)
        - Output: DataFrame các sự kiện spike với cột
        ['symbol','timestamp','close','pct_change','spike']
        - Logic:
        1. Chuẩn hóa symbols thành list
        2. Với mỗi symbol:
            a) fetch data bằng update_data
            b) gắn nhãn spike qua label_spikes
            c) lọc các dòng spike ≠ 0
        3. Gộp tất cả vào một DataFrame duy nhất
        """
        # Chuẩn hóa danh sách symbols
        if symbols is None:
        symbol_list = [DEFAULT_symbol]
        elif isinstance(symbols, str):
        symbol_list = [symbols]
        else:
        symbol_list = list(symbols)

        all_events = []
        for symbol in symbol_list:
            df = update_data(symbol, timeframe)
            if df is None or df.empty:
                continue
            labeled = label_spikes(df, threshold=pct_threshold)
            spikes = labeled[labeled['spike'] != 0][
                ['timestamp','close','pct_change','spike']
            ].copy()
            if not spikes.empty:
                spikes['symbol'] = symbol
                all_events.append(spikes)

        if not all_events:
            return pd.DataFrame(columns=['symbol','timestamp','close','pct_change','spike'])
        return pd.concat(all_events, ignore_index=True)
class DataProcessing:
    """
    Data Processing Class - Feature engineering and data transformation
    """
    
    def __init__(self):
        self.logger = get_logger('data_processing')
        
    # Feature engineering methods
    def FeatureEngineer(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Main feature engineering function"""
        # Implementation moved here
        
    def label_spikes(self, df: pd.DataFrame, price_col: str = 'close',
                    horizon: int = 12, threshold: float = 0.01) -> pd.DataFrame:
        """Label spike events based on future price movements"""
        try:
            if df.empty or price_col not in df.columns:
                self.logger.warning(f"Empty DataFrame or missing column {price_col}")
                return df
                
            df = df.copy()
            prices = df[price_col].values
            
            # Calculate future returns
            future_changes = []
            for i in range(len(prices)):
                if i + horizon < len(prices):
                    future_price = prices[i + horizon]
                    current_price = prices[i]
                    pct_change = (future_price - current_price) / current_price
                    future_changes.append(pct_change)
                else:
                    future_changes.append(0)  # Default for end of series
            
            df['future_pct_change'] = future_changes
            
            # Create binary labels based on threshold
            df['spike'] = (abs(df['future_pct_change']) > threshold).astype(int)
            df['label'] = df['spike']  # Alias for compatibility
            
            spike_count = df['spike'].sum()
            total_count = len(df)
            spike_ratio = spike_count / total_count if total_count > 0 else 0
            
            self.logger.info(f"Labeled {spike_count}/{total_count} ({spike_ratio:.2%}) spikes with threshold={threshold}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in label_spikes: {e}")
            return df
          def prepare_features(self, df: pd.DataFrame, symbol: str = None, training: bool = True) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        """Prepare features for ML models"""
        # Apply feature engineering
        df = self.compute_indicators(df)
        df_fe = self.feature_engineer.transform(df.copy())
        df_fe.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_fe.dropna(axis=1, how='all', inplace=True)

        if training:
            self.logger.info(f"Training data shape: {df_fe.shape}")
            
            if df_fe.empty:
                self.logger.error("Empty features after processing")
                return None, None

            # Get target labels
            if 'spike' in df_fe.columns:
                y = df_fe['spike']
            elif 'label' in df_fe.columns:
                y = df_fe['label']
            else:
                self.logger.error("No target labels found")
                return None, None

            # Remove non-feature columns
            feature_cols = df_fe.select_dtypes(include=[np.number]).columns
            exclude_cols = ['spike', 'label', 'future_change', 'future_pct_change']
            feature_cols = [col for col in feature_cols if col not in exclude_cols]
            
            X = df_fe[feature_cols]
            self.logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
            return X, y
        else:
            feature_cols = df_fe.select_dtypes(include=[np.number]).columns
            exclude_cols = ['spike', 'label', 'future_change', 'future_pct_change']
            feature_cols = [col for col in feature_cols if col not in exclude_cols]
            return df_fe[feature_cols]
        
    def prepare_df(self, symbol: str, tf: str, quick: bool) -> pd.DataFrame:
        """Prepare DataFrame for specific symbol and timeframe"""
        # Implementation moved here
        
    # Technical analysis methods
    def calculate_correlation(self, main_df: pd.DataFrame, other_assets: dict) -> dict:
        """Calculate correlation between assets"""
        if main_df is None or main_df.empty:
            return {}
        main_df['timestamp'] = pd.to_datetime(main_df['timestamp'], utc=True)
        main_series = main_df.set_index('timestamp')['close'].pct_change().dropna()
        corr = {}
        for name, df in other_assets.items():
            # Implementation for correlation calculation
            pass
        return corr

    def __init__(self):
        """Initialize data processing with global instances"""
        self.feature_engineer = FeatureEngineer()
        self.logger = get_logger('data_processing')
        self.settings = settings
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        
    # =============== MAIN DATA PROCESSING FUNCTIONS ===============
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature DataFrame with indicators and patterns"""
        df_feat = self.compute_indicators(df.copy())
        
        # Detect patterns and add pattern indicators
        pattern_list = self.detect_patterns(df_feat)
        for pat in pattern_list:
            pat_type = pat['type']
            pat_index = pat.get('end')
            col_name = f"pat_{pat_type.replace(' ', '_').replace('&', 'and')}"
            if col_name not in df_feat.columns:
                df_feat[col_name] = 0
            if pat_index is not None and pat_index < len(df_feat):
                df_feat.at[pat_index, col_name] = 1

        # Ensure all pattern columns exist
        pattern_types = [
            "Head and Shoulders", "Inverse Head and Shoulders", "Double Top", "Double Bottom",
            "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle",
            "Bullish Flag", "Bearish Flag", "Rising Wedge", "Falling Wedge"
        ]
        for pat_type in pattern_types:
            col_name = f"pat_{pat_type.replace(' ', '_').replace('&', 'and')}"
            if col_name not in df_feat.columns:
                df_feat[col_name] = 0

        # Remove time columns
        if 'open_time' in df_feat.columns:
            df_feat.drop(columns=['open_time'], inplace=True)
        if 'close_time' in df_feat.columns:
            df_feat.drop(columns=['close_time'], inplace=True)
            
        df_feat.dropna(inplace=True)
        return df_feat
    
    def validate_data(self, df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, bool]:
        """Validate and clean price data"""
        if df is None or df.empty:
            self.logger.warning("Empty dataframe")
            return df, False
            
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return df, False
            
        # Remove rows with invalid data
        df = df.dropna(subset=required_cols)
        df = df[(df['high'] >= df['low']) & (df['high'] >= df['open']) & 
                (df['high'] >= df['close']) & (df['low'] <= df['open']) & 
                (df['low'] <= df['close']) & (df['volume'] >= 0)]
        
        if len(df) < 100:
            self.logger.warning(f"Insufficient data after cleaning: {len(df)} rows")
            return df, False
            
        return df, True
        
    def load_and_process_data(self, symbol: str, timeframe: str, apply_indicators: bool = True,
                             apply_patterns: bool = True, apply_labels: bool = True,
                             horizon: int = 12, threshold: float = 0.01,
                             force_reload: bool = False) -> Optional[pd.DataFrame]:
        """
        Load and process data with all preprocessing steps
        """
        try:
            # Tạo đường dẫn file cache
            cache_file = os.path.join(self.data_dir, f"{symbol}_{timeframe}_processed.pkl.gz")
            
            if not force_reload and os.path.exists(cache_file):
                try:
                    with gzip.open(cache_file, 'rb') as f:
                        df = pickle.load(f)
                    self.logger.info(f"Loaded processed data from cache: {len(df)} rows")
                    return df
                except Exception as e:
                    self.logger.warning(f"Failed to load cache: {e}")

            # Load raw data 
            df = fetch_klines(symbol, timeframe)
            if df is None or df.empty:
                self.logger.error(f"No data available for {symbol} {timeframe}")
                return None

            # Validate and clean data
            df, is_valid = self.validate_data(df, timeframe)
            if not is_valid:
                self.logger.error(f"Data validation failed for {symbol} {timeframe}")
                return None

            # Apply labels if requested
            if apply_labels:
                self.logger.info(f"Applying labels with horizon={horizon}, threshold={threshold}")
                df = self.label_spikes(df, horizon=horizon, threshold=threshold)

            # Apply technical indicators
            if apply_indicators:
                self.logger.info("Computing technical indicators")
                df = self.compute_indicators(df)
                gc.collect()

            # Apply pattern recognition
            if apply_patterns:
                self.logger.info("Detecting chart patterns")
                pattern_functions = [
                    self.add_head_and_shoulders, self.add_double_top, self.add_double_bottom,
                    self.add_triangle, self.add_flag, self.add_pennant, self.add_rising_wedge,
                    self.add_falling_wedge, self.add_triple_top, self.add_triple_bottom, self.add_rectangle
                ]
                
                for pattern_fn in pattern_functions:
                    try:
                        df = pattern_fn(df)
                    except Exception as e:
                        self.logger.warning(f"Pattern {pattern_fn.__name__} failed: {e}")
                    gc.collect()

            # Save processed data to cache
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                self.logger.info(f"Saved processed data to cache")
            except Exception as e:
                self.logger.warning(f"Failed to save cache: {e}")

            return df

        except Exception as e:
            self.logger.error(f"Error in load_and_process_data: {e}", exc_info=True)
            return None

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators using FeatureEngineer"""
        # Validate required columns
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise ValueError(f"Missing column {col}")

        # Sort by time if available
        if "close_time" in df.columns:
            df = df.sort_values("close_time").reset_index(drop=True)

        # Apply feature engineering
        features = self.feature_engineer.transform(df)
        common = df.columns.intersection(features.columns)
        feat_only = features.drop(columns=common)
        
        result = df.join(feat_only, how="inner")
        return result

    def prepare_features(self, df: pd.DataFrame, symbol: str = None, training: bool = False):
        """Prepare features for ML models"""
        # Apply feature engineering
        df = self.compute_indicators(df)
        df_fe = self.feature_engineer.transform(df.copy())
        df_fe.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_fe.dropna(axis=1, how='all', inplace=True)

        if training:
            self.logger.info(f"Training data shape: {df_fe.shape}")
            
            if df_fe.empty:
                self.logger.error("Empty features after processing")
                return None, None

            # Get target labels
            if 'spike' in df_fe.columns:
                y = df_fe['spike']
            elif 'label' in df_fe.columns:
                y = df_fe['label']
            else:
                self.logger.error("No target labels found")
                return None, None

            # Remove non-feature columns
            feature_cols = df_fe.select_dtypes(include=[np.number]).columns
            exclude_cols = ['spike', 'label', 'future_change', 'future_pct_change']
            feature_cols = [col for col in feature_cols if col not in exclude_cols]
            
            X = df_fe[feature_cols]
            self.logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
            return X, y
        else:
            feature_cols = df_fe.select_dtypes(include=[np.number]).columns
            exclude_cols = ['spike', 'label', 'future_change', 'future_pct_change']
            feature_cols = [col for col in feature_cols if col not in exclude_cols]
            return df_fe[feature_cols]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature DataFrame with indicators and patterns"""
        df_feat = self.compute_indicators(df.copy())
        
        # Detect patterns and add pattern indicators
        pattern_list = self.detect_patterns(df_feat)
        for pat in pattern_list:
            pat_type = pat['type']
            pat_index = pat.get('end')
            col_name = f"pat_{pat_type.replace(' ', '_').replace('&', 'and')}"
            if col_name not in df_feat.columns:
                df_feat[col_name] = 0
            if pat_index is not None and pat_index < len(df_feat):
                df_feat.at[pat_index, col_name] = 1

        # Ensure all pattern columns exist
        pattern_types = [
            "Head and Shoulders", "Inverse Head and Shoulders", "Double Top", "Double Bottom",
            "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle",
            "Bullish Flag", "Bearish Flag", "Rising Wedge", "Falling Wedge"
        ]
        for pat_type in pattern_types:
            col_name = f"pat_{pat_type.replace(' ', '_').replace('&', 'and')}"
            if col_name not in df_feat.columns:
                df_feat[col_name] = 0

        # Remove time columns
        if 'open_time' in df_feat.columns:
            df_feat.drop(columns=['open_time'], inplace=True)
        if 'close_time' in df_feat.columns:
            df_feat.drop(columns=['close_time'], inplace=True)
            
        df_feat.dropna(inplace=True)
        return df_feat

    def detect_patterns(self, df: pd.DataFrame) -> list:
        """Detect chart patterns from price data"""
        patterns = []
        highs = df['high']
        lows = df['low']
        n = len(df)
        
        # Identify peaks and troughs
        peak_indices = []
        trough_indices = []
        for i in range(1, n-1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peak_indices.append(i)
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                trough_indices.append(i)

        # Head and Shoulders
        for j in range(1, len(peak_indices)-1):
            left_idx = peak_indices[j-1]
            mid_idx = peak_indices[j]
            right_idx = peak_indices[j+1]
            left_h = highs[left_idx]
            mid_h = highs[mid_idx]
            right_h = highs[right_idx]
            shoulders_avg = (left_h + right_h) / 2.0
            if shoulders_avg == 0:
                continue
            if mid_h > shoulders_avg * 1.05 and abs(left_h - right_h) / shoulders_avg < 0.03:
                patterns.append({"type": "Head and Shoulders", "end": right_idx})

        # Double Top
        for j in range(len(peak_indices)-1):
            i1 = peak_indices[j]
            i2 = peak_indices[j+1]
            h1 = highs[i1]
            h2 = highs[i2]
            if h1 == 0:
                continue
            if abs(h1 - h2) / h1 < 0.03:
                mid_low = lows[i1+1:i2].min() if i2 > i1 + 1 else lows[i1]
                if mid_low < min(h1, h2) * 0.95:
                    patterns.append({"type": "Double Top", "end": i2})

        # Double Bottom  
        for j in range(len(trough_indices)-1):
            i1 = trough_indices[j]
            i2 = trough_indices[j+1]
            l1 = lows[i1]
            l2 = lows[i2]
            if l1 == 0:
                continue
            if abs(l1 - l2) / l1 < 0.03:
                mid_high = highs[i1+1:i2].max() if i2 > i1 + 1 else highs[i1]
                if mid_high > max(l1, l2) * 1.05:
                    patterns.append({"type": "Double Bottom", "end": i2})

        return patterns

    def validate_data(self, df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, bool]:
        """Validate and clean price data"""
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            if 'close_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['close_time'], unit='ms', utc=True, errors='coerce')
            else:
                raise KeyError("Missing timestamp and close_time columns")
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

        df.dropna(subset=['timestamp'], inplace=True)
        df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        if df.empty:
            return df, False

        # Filter invalid prices
        for col in ('open', 'high', 'low', 'close', 'volume'):
            if col in df.columns:
                df = df[df[col] > 0]

        return df, True

    def label_spikes(self, df: pd.DataFrame, price_col: str = 'close',
                    horizon: int = 12, threshold: float = 0.01) -> pd.DataFrame:
        """Label spike events based on future price movements"""
        try:
            if df.empty or price_col not in df.columns:
                self.logger.warning(f"Empty DataFrame or missing column {price_col}")
                return df
                
            df = df.copy()
            prices = df[price_col].values
            
            # Calculate future returns
            future_changes = []
            for i in range(len(prices)):
                if i + horizon < len(prices):
                    future_price = prices[i + horizon]
                    current_price = prices[i]
                    pct_change = (future_price - current_price) / current_price
                    future_changes.append(pct_change)
                else:
                    future_changes.append(0)  # Default for end of series
            
            df['future_pct_change'] = future_changes
            
            # Create binary labels based on threshold
            df['spike'] = (abs(df['future_pct_change']) > threshold).astype(int)
            df['label'] = df['spike']  # Alias for compatibility
            
            spike_count = df['spike'].sum()
            total_count = len(df)
            self.logger.info(f"Labeled {spike_count}/{total_count} ({spike_count/total_count*100:.1f}%) as spikes")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in label_spikes: {e}")
            return df
        """Label spike events based on future price movements"""
        
        df = df.copy()
        
        # Find time column
        time_col = None
        for col_name in ['timestamp', 'close_time', 'time']:
            if col_name in df.columns:
                time_col = col_name
                break
        
        # Sort by time if available
        if time_col:
            df = df.sort_values(time_col).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        
        # Calculate future price movement
        future_price = df[price_col].shift(-horizon)
        df['future_change'] = (future_price - df[price_col]) / df[price_col]

        # Label spikes
        df['spike'] = 0
        df.loc[df['future_change'] >= threshold, 'spike'] = 1
        df.loc[df['future_change'] <= -threshold, 'spike'] = -1

        # Remove rows without future data
        df.dropna(subset=['future_change'], inplace=True)
        df['future_pct_change'] = df['future_change']
        df['label'] = df['spike']
        return df

    def collect_spike_events(self, symbols=None, timeframe: str = '15m',
                           pct_threshold: float = 0.05) -> pd.DataFrame:
        """Collect spike events across multiple symbols"""
        if symbols is None:
            symbol_list = [DEFAULT_symbol]
        elif isinstance(symbols, str):
            symbol_list = [symbols]
        else:
            symbol_list = list(symbols)

        all_events = []
        for symbol in symbol_list:
            df = update_data(symbol, timeframe)
            if df is None or df.empty:
                continue
            labeled = self.label_spikes(df, threshold=pct_threshold)
            spikes = labeled[labeled['spike'] != 0][
                ['timestamp', 'close', 'future_change', 'spike']
            ].copy()
            if not spikes.empty:
                spikes['symbol'] = symbol
                all_events.append(spikes)

        if not all_events:
            return pd.DataFrame(columns=['symbol', 'timestamp', 'close', 'future_change', 'spike'])
        return pd.concat(all_events, ignore_index=True)

    # Pattern detection methods
    def add_head_and_shoulders(self, df: pd.DataFrame, window: int = 30, tol: float = 0.03) -> pd.DataFrame:
        """Detect head and shoulders pattern"""
        hs = np.zeros(len(df))
        prices = df['close'].values
        for i in range(window, len(prices) - window):
            seg = prices[i-window:i+window]
            head = seg[window]
            left = seg[:window].max()
            right = seg[window+1:].max()
            if head > left and head > right and abs(left-right)/head < tol:
                hs[i] = 1
        df['pattern_head_shoulders'] = hs
        return df

    def add_double_top(self, df: pd.DataFrame, window: int = 50, tol: float = 0.02) -> pd.DataFrame:
        """Detect double top pattern"""
        dt = np.zeros(len(df))
        highs = df['high'].values
        peaks = (highs[1:-1] > highs[:-2]) & (highs[1:-1] > highs[2:])
        idx = np.where(peaks)[0] + 1
        for j in range(len(idx)-1):
            i1, i2 = idx[j], idx[j+1]
            if i2-i1 <= window and abs(highs[i1]-highs[i2])/highs[i1] < tol:
                dt[i2] = 1
        df['pattern_double_top'] = dt
        return df

    def add_double_bottom(self, df: pd.DataFrame, window: int = 50, tol: float = 0.02) -> pd.DataFrame:
        """Detect double bottom pattern"""
        db = np.zeros(len(df))
        lows = df['low'].values
        troughs = (lows[1:-1] < lows[:-2]) & (lows[1:-1] < lows[2:])
        idx = np.where(troughs)[0] + 1
        for j in range(len(idx)-1):
            i1, i2 = idx[j], idx[j+1]
            if i2-i1 <= window and abs(lows[i1]-lows[i2])/lows[i1] < tol:
                db[i2] = 1
        df['pattern_double_bottom'] = db
        return df

    def add_triangle(self, df: pd.DataFrame, window: int = 50, slope_thresh: float = 0.01) -> pd.DataFrame:
        """Detect triangle pattern"""
        from sklearn.linear_model import LinearRegression
        tri = np.zeros(len(df))
        for i in range(window, len(df)-window):
            seg = df['close'].iloc[i-window:i+window].values
            x = np.arange(len(seg)).reshape(-1, 1)
            lr = LinearRegression().fit(x, seg)
            if abs(lr.coef_[0]) < slope_thresh:
                tri[i] = 1
        df['pattern_triangle'] = tri
        return df

    def add_flag(self, df: pd.DataFrame, window: int = 20, slope_thresh: float = 0.02) -> pd.DataFrame:
        """Detect flag pattern"""
        flag = np.zeros(len(df))
        for i in range(window, len(df)):
            seg = df['close'].iloc[i-window:i].values
            slope = np.polyfit(np.arange(window), seg, 1)[0]
            if abs(slope) < slope_thresh:
                flag[i] = 1
        df['pattern_flag'] = flag
        return df

    def add_pennant(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Detect pennant pattern"""
        pen = np.zeros(len(df))
        for i in range(window*2, len(df)):
            seg = df['high'].iloc[i-window*2:i].values
            std1 = np.std(seg[:window])
            std2 = np.std(seg[window:])
            if std2 < std1:
                pen[i] = 1
        df['pattern_pennant'] = pen
        return df

    def add_rising_wedge(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """Detect rising wedge pattern"""
        rw = np.zeros(len(df))
        for i in range(window, len(df)):
            high_seg = df['high'].iloc[i-window:i].values
            low_seg = df['low'].iloc[i-window:i].values
            high_slope = np.polyfit(np.arange(window), high_seg, 1)[0]
            low_slope = np.polyfit(np.arange(window), low_seg, 1)[0]
            if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
                rw[i] = 1
        df['pattern_rising_wedge'] = rw
        return df

    def add_falling_wedge(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """Detect falling wedge pattern"""
        fw = np.zeros(len(df))
        for i in range(window, len(df)):
            high_seg = df['high'].iloc[i-window:i].values
            low_seg = df['low'].iloc[i-window:i].values
            high_slope = np.polyfit(np.arange(window), high_seg, 1)[0]
            low_slope = np.polyfit(np.arange(window), low_seg, 1)[0]
            if high_slope < 0 and low_slope < 0 and high_slope > low_slope:
                fw[i] = 1
        df['pattern_falling_wedge'] = fw
        return df

    def add_triple_top(self, df: pd.DataFrame, window: int = 50, tol: float = 0.02) -> pd.DataFrame:
        """Detect triple top pattern"""
        tt = np.zeros(len(df))
        highs = df['high'].values
        peaks = (highs[1:-1] > highs[:-2]) & (highs[1:-1] > highs[2:])
        idx = np.where(peaks)[0] + 1
        for j in range(len(idx)-2):
            i1, i2, i3 = idx[j], idx[j+1], idx[j+2]
            cond1 = i2-i1 <= window and i3-i2 <= window
            cond2 = abs(highs[i1]-highs[i2])<tol*highs[i1]
            cond3 = abs(highs[i2]-highs[i3])<tol*highs[i2]
            if cond1 and cond2 and cond3:
                tt[i3] = 1
        df['pattern_triple_top'] = tt
        return df

    def add_triple_bottom(self, df: pd.DataFrame, window: int = 50, tol: float = 0.02) -> pd.DataFrame:
        """Detect triple bottom pattern"""
        tb = np.zeros(len(df))
        lows = df['low'].values
        troughs = (lows[1:-1] < lows[:-2]) & (lows[1:-1] < lows[2:])
        idx = np.where(troughs)[0] + 1
        for j in range(len(idx)-2):
            i1, i2, i3 = idx[j], idx[j+1], idx[j+2]
            cond1 = i2-i1 <= window and i3-i2 <= window
            cond2 = abs(lows[i1]-lows[i2])<tol*lows[i1]
            cond3 = abs(lows[i2]-lows[i3])<tol*lows[i2]
            if cond1 and cond2 and cond3:
                tb[i3] = 1
        df['pattern_triple_bottom'] = tb
        return df

    def add_rectangle(self, df: pd.DataFrame, window: int = 30, tol: float = 0.01) -> pd.DataFrame:
        """Detect rectangle pattern"""
        rect = np.zeros(len(df))
        for i in range(window, len(df)-window):
            high_seg = df['high'].iloc[i-window:i+window]
            low_seg = df['low'].iloc[i-window:i+window]
            high_std = high_seg.std() / high_seg.mean()
            low_std = low_seg.std() / low_seg.mean()
            if high_std < tol and low_std < tol:
                rect[i] = 1
        df['pattern_rectangle'] = rect
        return df

    # 6. autoheal module - Tự động xử lý và bổ sung dữ liệu
    def autoheal_data(df, symbol=None, timeframe=None):
        """
        Làm sạch, nội suy, loại bỏ trùng, sửa giá trị bất thường và nếu cần tải bổ sung.
        """
        if df is None:
            return None

        data = df.copy()
        try:
            # 1) Nội suy các mốc thời gian
            if isinstance(data.index, pd.DatetimeIndex):
                freq = pd.infer_freq(data.index[:5])
                if freq:
                    full_idx = pd.date_range(data.index.min(), data.index.max(), freq=freq)
                    if len(full_idx) != len(data.index):
                        data = data.reindex(full_idx)
                        data.interpolate(method='time', inplace=True)
                        data.ffill(inplace=True)

            # 2) Loại bỏ trùng index
            if isinstance(data.index, pd.DatetimeIndex) and data.index.has_duplicates:
                data = data[~data.index.duplicated(keep='first')]

            # 3) Xử lý volume = 0
            if 'volume' in data.columns:
                zero_idx = data['volume'] == 0
                if zero_idx.any():
                    data.loc[zero_idx, 'volume'] = math.nan
                    data['volume'].interpolate(method='linear', inplace=True)

            # 4) Loại bỏ close <= 0 hoặc NaN
            if 'close' in data.columns:
                bad_idx = (data['close'] <= 0) | data['close'].isna()
                if bad_idx.any():
                    data = data[~bad_idx]

            # 5) Nếu cần, tải bổ sung dữ liệu cuối
            if symbol and timeframe and isinstance(data.index, pd.DatetimeIndex):
                last_time = data.index.max()
                now = datetime.datetime.utcnow()
                unit = timeframe[-1].lower()
                try:
                    val = int(timeframe[:-1])
                except:
                    val = 1
                if unit == 'm':
                    delta = datetime.timedelta(minutes=val)
                elif unit == 'h':
                    delta = datetime.timedelta(hours=val)
                elif unit == 'd':
                    delta = datetime.timedelta(days=val)
                else:
                    delta = datetime.timedelta(0)

                if now - last_time > 2 * delta:
                    print("Dữ liệu cần cập nhật bổ sung, đang tải thêm...")
                    try:
                        
                        new_df = load_data_for_symbol(symbol, timeframe)
                        if new_df is not None:
                            if 'time' in new_df.columns:
                                new_df.set_index('time', inplace=True)
                            data = data.combine_first(new_df)
                    except ImportError:
                        pass

        except Exception as e:
            print("Lỗi trong autoheal_data:", e)

        return data
    def augment(df: pd.DataFrame) -> pd.DataFrame:
        """
        Thực thi augmentation cho DataFrame giá:
        - ret_pct       : tỷ suất sinh lời phần trăm (close pct_change)
        - log_return    : log(1 + ret_pct)
        - volatility    : độ lệch chuẩn động của log_return (window=20)
        - lag_close_N   : giá đóng cửa trễ N chu kỳ (lag features)
        Trả về DataFrame đã loại bỏ NaN đầu chuỗi, chỉ chứa các cột numeric + timestamp, open, high, low, close, volume.
        """
        data = df.copy()

        # 1) Tính ret_pct và log_return
        data['ret_pct']    = data['close'].pct_change()
        data['log_return'] = np.log1p(data['ret_pct'])

        # 2) Tính volatility: rolling std của log_return (window=20)
        data['volatility'] = data['log_return'].rolling(window=20, min_periods=1).std()

        # 3) Tạo lag features cho giá đóng cửa
        for lag in (1, 2, 3):
            data[f'lag_close_{lag}'] = data['close'].shift(lag)

        # 4) Xóa NaN do pct_change/shift/rolling, reset index
        data = data.dropna().reset_index(drop=True)

        return data
# crypto/indicators/custom_indicator.py

    def apply_custom_indicator(df, func, col_name='Custom'):
        """
        Thêm một chỉ báo tùy chỉnh vào DataFrame:
        - df: DataFrame chứa dữ liệu OHLCV.
        - func: hàm do người dùng định nghĩa, nhận df và trả về Series hoặc list giá trị chỉ báo.
        - col_name: tên cột mới cho chỉ báo tùy chỉnh.
        Hàm sẽ thêm cột `col_name` vào df với giá trị do func tính toán.
        """
        if df is None or df.empty or not callable(func):
            return df
        try:
            result = func(df)
        except Exception as e:
            # Nếu hàm tùy chỉnh gặp lỗi, trả về df gốc không thay đổi
            print(f"Lỗi khi áp dụng chỉ báo tùy chỉnh: {e}")
            return df
        # Thêm cột chỉ báo mới vào DataFrame
        df[col_name] = result
        return df
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators and add them as columns to the DataFrame.
        Assumes df has columns: open, high, low, close, volume, and close_time (for VWAP grouping).
        """
            # Kiểm tra cột bắt buộc
        for col in ("open","high","low","close","volume"):
            if col not in df.columns:
                raise ValueError(f"[compute_indicators] Thiếu cột {col}")

        # Sort nếu có close_time để giữ thứ tự đúng
        if "close_time" in df.columns:
            df = df.sort_values("close_time").reset_index(drop=True)

        # Gọi FeatureEngineer.transform – nó sẽ tính toàn bộ indicators, interactions, PCA…
        features = _FE.transform(df)
        common = df.columns.intersection(features.columns)
        feat_only = features.drop(columns=common)
        # Nếu bạn cần merge các cột features trở lại df gốc, có thể:
        result = df.join(feat_only, how="inner")
        return result
    class FeatureEngineer:
        """
        Tính toàn bộ indicator cơ bản + nâng cao + chuyên sâu.
        Các tham số này sẽ được Optuna tune hoặc override qua Pipeline.set_params().
        """
        def __init__(self,
            # basic spans/windows
            ema_spans=(20,50,89),
            sma_windows=(20,50),
            # advanced
            hma_periods=(20,50),
            frama_windows=(20,),
            # core
            rsi_period=14,
            macd_params=(12,26,9),
            atr_period=14,
            cci_period=20,
            ichimoku_params=(9,26,52),
            adx_period=14,
            stoch_rsi_params=(14,3,3),
            wr_period=14,
            bollinger_params=(20,2),
            # additional
            cmf_period=20,
            mfi_period=14,
            psar_step=0.02,
            psar_maxstep=0.2,
            kc_atr_mult=1.5,
            vi_period=14,
            uo_p1=7, uo_p2=14, uo_p3=28,
            tsi_r=25, tsi_s=13,
            eom_period=14,
            trix_period=18,
            hurst_window=100,
            sampen_m=2, sampen_r=0.2,
            vol_window=20,
            fft_n=5,
                # ===== MỚI =====
            #: list of (col1, col2) to create product features col1_x_col2
            composite_interaction_pairs=None,
            #: degree for polynomial expansion (>=1)
            poly_degree=1,
            #: number of PCA components to extract (0 = disable)
            n_pca=0
            
        ):
            self.ema_spans=ema_spans; self.sma_windows=sma_windows
            self.hma_periods=hma_periods; self.frama_windows=frama_windows
            self.rsi_period=rsi_period; self.macd_params=macd_params
            self.atr_period=atr_period; self.cci_period=cci_period
            self.ichimoku_params=ichimoku_params; self.adx_period=adx_period
            self.stoch_rsi_params=stoch_rsi_params; self.wr_period=wr_period
            self.bollinger_params=bollinger_params
            self.cmf_period=cmf_period; self.mfi_period=mfi_period
            self.psar_step=psar_step; self.psar_maxstep=psar_maxstep
            self.kc_atr_mult=kc_atr_mult; self.vi_period=vi_period
            self.uo_p1=uo_p1; self.uo_p2=uo_p2; self.uo_p3=uo_p3
            self.tsi_r=tsi_r; self.tsi_s=tsi_s; self.eom_period=eom_period
            self.trix_period=trix_period; self.hurst_window=hurst_window
            self.sampen_m=sampen_m; self.sampen_r=sampen_r
            self.vol_window=vol_window; self.fft_n=fft_n
            # ===== INIT MỚI =====
            # nếu không muốn interaction thì để None hoặc []
            self.composite_interaction_pairs = composite_interaction_pairs or [
                ("rsi", "atr"),
                ("macd", "rsi"),
                ("bb_upper", "bb_lower"),
            ]
            self.poly_degree = poly_degree
            self.n_pca       = n_pca

        def get_indicator_columns(self, df):
            # sau khi transform(df) trả về bạn có thể lấy df.columns nếu cần
            return self.transform(df.copy()).columns.tolist()
        
        def fit(self, X, y=None):
            return self

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            
            
            # 0) Chuẩn bị
            df = df.copy().reset_index(drop=True)
            required = ["open", "high", "low", "close", "volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Thiếu cột bắt buộc: {missing}")
            o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

            # Fill NA cho tất cả
            for col in ["open","high","low","close"]:
                df[col] = df[col].ffill().bfill()
            df["volume"] = df["volume"].ffill().fillna(0)
            o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

            # 1) Basic & core indicators (raw names)
            # 1.1 EMA & SMA & HMA & FRAMA
            for p in self.ema_spans:
                df[f"ema_{p}"] = ema(c, p)
            for w in self.sma_windows:
                df[f"sma_{w}"] = sma(c, w)
            for p in self.hma_periods:
                df[f"hma_{p}"] = hma(c, p)
            for w in self.frama_windows:
                df[f"frama_{w}"] = frama(c, w)

            # 1.2 RSI
            df[f"rsi_{self.rsi_period}"] = rsi(c, self.rsi_period)

            # 1.3 MACD
            f, s, g = self.macd_params
            macd_line, macd_sig = macd(c, f, s, g)
            df[f"macd_{f}_{s}_{g}"]     = macd_line
            df[f"macd_sig_{f}_{s}_{g}"] = macd_sig

            # 1.4 ATR & CCI
            df[f"atr_{self.atr_period}"] = atr(h, l, c, self.atr_period)
            df[f"cci_{self.cci_period}"] = cci(h, l, c, self.cci_period)

            # 1.5 Ichimoku
            ten, kij, sa, sb, ch = ichimoku(h, l, c, *self.ichimoku_params)
            df["tenkan"], df["kijun"]  = ten, kij
            df["spanA"], df["spanB"]   = sa, sb
            df["chikou"]               = ch

            # 1.6 ADX
            df[f"adx_{self.adx_period}"] = adx(h, l, c, self.adx_period)

            # 1.7 Stochastic RSI
            sk, sd = stoch_rsi(c, *self.stoch_rsi_params)
            df["stoch_k"], df["stoch_d"] = sk, sd

            # 1.8 Williams %R
            df[f"wr_{self.wr_period}"] = williams_r(h, l, c, self.wr_period)

            # 1.9 Bollinger Bands
            ub, mb, lb = bollinger_bands(c, *self.bollinger_params)
            df["bb_up"], df["bb_mid"], df["bb_lo"] = ub, mb, lb

            # 1.10 Parabolic SAR
            df["psar"] = psar(h, l, c, self.psar_step, self.psar_maxstep)

            # 1.11 Keltner Channel
            du, mid, dn = keltner(h, l, c, v, self.atr_period, self.kc_atr_mult)
            df["kc_up"], df["kc_mid"], df["kc_lo"] = du, mid, dn

            # 1.12 Vortex Indicator
            vip, vin = vortex_indicator(h, l, c, self.vi_period)
            df["vi_plus"], df["vi_minus"] = vip, vin

            # 1.13 Ultimate Oscillator & TSI & EOM & TRIX
            df["uo"]       = ultimate_oscillator(h, l, c, self.uo_p1, self.uo_p2, self.uo_p3)
            df["tsi"]      = tsi(c, self.tsi_r, self.tsi_s)
            df[f"eom_{self.eom_period}"] = eom(h, l, c, v, self.eom_period)
            df[f"trix_{self.trix_period}"] = trix(c, self.trix_period)

        

            # 1.15 FFT coefficients — chỉ tính khi có đủ dữ liệu
            if len(c) > 0:
                try:
                    c_clean = c.ffill().bfill()
                    if len(c_clean.dropna()) > 0:
                        coeffs = fft.fft(c_clean.values)
                        for i in range(1, min(self.fft_n, len(coeffs)//2)):
                            df[f"fft_{i}"] = np.real(coeffs[i])
                    else:
                        logger.warning("Skipping FFT features: all values are NaN after cleaning")
                except ValueError as e:
                    logger.warning(f"Skipping FFT features: {e}")
            else:
                logger.warning("Skipping FFT features: input series is empty")        # 4) Hurst & Sample Entropy helper functions
            def _safe_hurst(series, w):
                try:
                    return hurst(series, w)
                except ZeroDivisionError:
                    return pd.Series(np.nan, index=series.index)

            def _safe_sampen(series, m, r):
                """
                Wrapper an toàn cho antropy.sample_entropy:
                - Chuyển về float array, tính trên log-returns
                - Dùng errstate ignore để che warn log(0)
                - nan/inf → 0
                - Trả Series cùng index (giá trị lặp đều)
            """
                x = np.asarray(series.values, dtype=float)
                # 1) Log-returns để tránh quá nhiều zeros
                with np.errstate(divide='ignore', invalid='ignore'):
                    ret = np.diff(np.log(x + EPS))
                if len(ret) == 0 or np.nanstd(ret) <= 0:
                    val = 0.0
                else:
                    std_ret = np.nanstd(ret)
                    r_eff = r if r is not None else 0.2 * std_ret
                    r_eff = max(r_eff, EPS)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        val = _antropy_sampen(ret, m, r_eff)
                    # ép nan/inf → 0
                    val = float(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0))
                # Trả Series full độ dài
                return pd.Series([val] * len(series), index=series.index)        
            # 1.14 Hurst & Sample Entropy
            df[f"hurst_{self.hurst_window}"] = _safe_hurst(c, self.hurst_window)

            max_len = 60000
            if len(c) <= max_len:
                df[f"sampen_{self.sampen_m}"] = _safe_sampen(c, self.sampen_m, self.sampen_r)
            else:
                logger.warning(f"Dữ liệu quá lớn ({len(c)}), chỉ tính Sample Entropy trên {max_len} dòng cuối.")
                # lấy 50000 dòng cuối
                tail = c.iloc[-max_len:]
                sampen_tail = _safe_sampen(tail, self.sampen_m, self.sampen_r)
                # tạo Series độ dài bằng c.index: phần đầu NaN, phần cuối gán giá trị sampen_tail
                padded = pd.Series(
                    [np.nan] * (len(c) - max_len) + sampen_tail.tolist(),
                    index=c.index
                )
                df[f"sampen_{self.sampen_m}"] = padded

            # 5) FFT coefficients đã được tính ở trên, bỏ qua phần duplicate này

            # 6) Đổi tên raw → standard (chỉ 1 lần duy nhất ở đây)
            rename_map = {
                f"rsi_{self.rsi_period}":               "rsi",
                f"atr_{self.atr_period}":               "atr",
                f"cci_{self.cci_period}":               "cci",
                f"adx_{self.adx_period}":               "adx",
                f"wr_{self.wr_period}":                 "willr",
                f"eom_{self.eom_period}":               "eom",
                f"macd_{f}_{s}_{g}":                    "macd",
                f"macd_sig_{f}_{s}_{g}":                "macd_signal",
                "bb_up":     "bb_upper",
                "bb_mid":    "bb_middle",
                "bb_lo":     "bb_lower",
                "stoch_k":   "stochk",
                "stoch_d":   "stochd",
                "uo":        "ult_osc",
                "PSAR":            "psar",
                "tsi":                      "tsi",
                "vi_minus":                 "vi_minus",
                "vi_plus":                  "vi_plus",
                "kc_up":                    "kc_up",
                "kc_mid":                   "kc_mid",
                "kc_lo":                    "kc_lo",
                f"mfi_{self.mfi_period}":   "mfi",
                "tenkan":                   "ichimoku_conv",
                "kijun":                    "ichimoku_base",
                "spanA":                    "ichimoku_a",
                "spanB":                    "ichimoku_b",
                "vwap":                     "vwap",
                "roc":                      "roc",
            }
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
                    inplace=True)

            # 7) Composite features: gọi luôn hàm trong COMPOSITE_REGISTRY
            for feat_name, func in COMPOSITE_REGISTRY.items():
                if feat_name in df.columns:
                    continue
                try:
                    df[feat_name] = func(df)
                except Exception as e:
                    logger.warning(f"Skipping composite feature {feat_name}: {e}")

            # 8) Polynomial expansion (nếu degree > 1)
            if self.poly_degree and self.poly_degree > 1:
                base = [col for col in df.columns if col not in required]
                poly = PolynomialFeatures(degree=self.poly_degree,
                                        interaction_only=False,
                                        include_bias=False)
                mat = df[base].fillna(0).values
                names = poly.get_feature_names_out(base)
                df_poly = pd.DataFrame(poly.fit_transform(mat),
                                    columns=names,
                                    index=df.index)
                df = pd.concat([df, df_poly], axis=1)

            # 9) PCA (nếu cần)
            if self.n_pca and self.n_pca > 0:
                pca_cols = [col for col in df.columns if col not in required]
                if len(pca_cols) >= self.n_pca:
                    mat = df[pca_cols].fillna(0).values
                    comps = PCA(n_components=self.n_pca).fit_transform(mat)
                    for i in range(self.n_pca):
                        df[f"pca_{i+1}"] = comps[:, i]

            # 10) Cleanup và trả về - ít nghiêm ngặt hơn
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Chỉ drop cột toàn NaN
            df.dropna(axis=1, how='all', inplace=True)
            
            # Thay vì dropna(how='any'), chỉ drop khi quá nhiều NaN
            # Giữ lại hàng nếu có ít nhất 50% cột hợp lệ
            min_valid_cols = max(1, int(df.shape[1] * 0.5))
            df.dropna(thresh=min_valid_cols, inplace=True)
            
            # Fill NaN còn lại bằng 0 hoặc forward fill
            # Thay vì dropna(how='any'), chỉ drop khi quá nhiều NaN
            min_valid_cols = max(1, int(df.shape[1] * 0.5))
            df.dropna(thresh=min_valid_cols, inplace=True)

            # —————————————— KHẮC PHỤC LỖI ——————————————
            # Loại bỏ cột trùng tên trước khi xác định numeric_cols
            df = df.loc[:, ~df.columns.duplicated()]

            # Fill NaN còn lại bằng 0 hoặc forward fill
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            # Dùng .loc để gán rõ ràng hàng và cột
            df.loc[:, numeric_cols] = df.loc[:, numeric_cols].ffill().fillna(0)

            return df.select_dtypes(include=[np.number])
        
    def ema(s, span):
        return s.ewm(span=span, adjust=False).mean()

    def sma(s, window):
        return s.rolling(window).mean()

    def hma(s, period):
        half = sma(s, period//2)
        full = sma(s, period)
        return sma(2*half - full, int(np.sqrt(period)))

    def frama(s, window, fc=4.6, sc=0.1):
        x = s.values
        N = window
        L = np.log(2) / np.log(N)
        if np.sum(np.abs(np.diff(x[:N]))) <= 0 \
        or np.sum(np.abs(np.diff(x[N:2*N]))) <= 0:
            D0, D1 = np.nan, np.nan
        else:
            D0 = np.log(np.sum(np.abs(np.diff(x[:N])))) / np.log(N)
            D1 = np.log(np.sum(np.abs(np.diff(x[N:2*N])))) / np.log(N)
        D     = (D0 + D1) / 2
        alpha = np.exp(-L * (D - sc) / (fc - sc))
        # clamp trực tiếp trên alpha để luôn 0 < alpha <= 1
        if alpha <= 0:
            alpha = EPS
        elif alpha > 1:
            alpha = 1.0
        return s.ewm(alpha=alpha, adjust=False).mean()

    def rsi(s, period=14):
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        # ══ tránh chia 0
        avg_loss = avg_loss.mask(avg_loss.abs() == 0, EPS)
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def macd(s, fast=12, slow=26, signal=9):
        fe = ema(s, fast)
        se = ema(s, slow)
        line = fe - se
        sig  = ema(line, signal)
        return line, sig

    def atr(high, low, close, period=14):
        hl = high - low
        hc = (high - close.shift()).abs()
        lc = (low  - close.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def cci(high, low, close, period=20):
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        denom = 0.015 * mad
        denom = denom.mask(denom.abs() ==0, EPS)
        return (tp - sma_tp) / denom

    def ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
        
        ten = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        kij = (high.rolling(kijun).max()  + low.rolling(kijun).min())  / 2
        sa  = ((ten + kij) / 2).shift(kijun)
        sb  = ((high.rolling(senkou).max()+ low.rolling(senkou).min()) / 2).shift(kijun)
        ch  = close.shift(-kijun)
        return ten, kij, sa, sb, ch

    def adx(high, low, close, period=14):
        tr = atr(high, low, close, period)
        tr = tr.mask(tr.abs() ==0, EPS)
        up = high.diff().clip(lower=0)
        dn = (-low.diff()).clip(lower=0)
        pdi = 100 * up.ewm(alpha=1/period, adjust=False).mean() / tr
        mdi = 100 * dn.ewm(alpha=1/period, adjust=False).mean() / tr
        dx = (pdi - mdi).abs() / (pdi + mdi) * 100
        return dx.ewm(alpha=1/period, adjust=False).mean()

    def stoch_rsi(s, period=14, k=3, d=3):
        r = rsi(s, period)
        minr = r.rolling(period).min()
        maxr = r.rolling(period).max()
        denom = (maxr - minr).mask((maxr - minr).abs() ==0, EPS)
        rs = 100 * (r - minr) / denom
        return rs.rolling(k).mean(), rs.rolling(k).mean().rolling(d).mean()

    def williams_r(high, low, close, period=14):
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        denom = (hh - ll).mask((hh - ll).abs() ==0, EPS)
        return -100 * (hh - close) / denom

    def bollinger_bands(s, period=20, std=2):
        m = s.rolling(period).mean()
        sd = s.rolling(period).std()
        return m + std*sd, m, m - std*sd

    def obv(close, volume):
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    def momentum(s, period=10):
        return s.diff(periods=period)

    def donchian(high, low, period=20):
        return high.rolling(period).max(), low.rolling(period).min()

    def vwap(high, low, close, volume, period=14):
        tp = (high + low + close) / 3
        num = (tp * volume).rolling(period).sum()
        denom = volume.rolling(period).sum().mask(volume.rolling(period).sum() ==0, EPS)
        return num / denom

    def trix(s, period=18):
        e1 = s.ewm(span=period, adjust=False).mean()
        e2 = e1.ewm(span=period, adjust=False).mean()
        e3 = e2.ewm(span=period, adjust=False).mean()
        return e3.pct_change() * 100

    def hurst(s, window):
        if len(s) < window:
            return pd.Series(np.nan, index=s.index)
        X = (s - s.rolling(window).mean()).dropna()
        Z = X.cumsum()
        R = Z.max() - Z.min()
        S = X.std()
        if S < EPS:
            return pd.Series(0.0, index=s.index)
        H = np.log(R/S) / np.log(len(X))
        return pd.Series(H, index=s.index)

    def mfi(high, low, close, volume, period=14):
        tp = (high+low+close)/3; mf = tp*volume
        pos = mf.where(tp>tp.shift(1),0); neg = mf.where(tp<tp.shift(1),0)
        num = pos.rolling(period).sum()
        denom = neg.rolling(period).sum().mask(neg.rolling(period).sum() ==0, EPS)
        mr = num / denom
        return 100-100/(1+mr)

    def cmf(high, low, close, volume, period=20):
        mf_mult = ((close - low) - (high - close)) / (high - low)
        mf_vol = mf_mult * volume
        num = mf_vol.rolling(period).sum()
        denom = volume.rolling(period).sum().mask(volume.rolling(period).sum() ==0, EPS)
        return num / denom

    # Parabolic SAR (simple)
    def psar(high, low, close, step=0.02, max_step=0.2):
        length = len(close)
        # Nếu không đủ điểm (dưới 2), trả về Series toàn NaN để tránh ps.iloc[0]=low.iloc[0] lỗi oob
        if length < 2:
            return pd.Series(np.nan, index=close.index)
        ps = close.copy()
        ps.iloc[0] = low.iloc[0]
        bull = True
        af = step
        ep = high.iloc[0]
        for i in range(1, length):
            ps.iloc[i] = ps.iloc[i-1] + af*(ep - ps.iloc[i-1])
            if bull:
                if low.iloc[i] < ps.iloc[i]:
                    bull = False; ps.iloc[i] = ep; af = step; ep = low.iloc[i]
                else:
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]; af = min(af+step, max_step)
            else:
                if high.iloc[i] > ps.iloc[i]:
                    bull = True; ps.iloc[i] = ep; af = step; ep = high.iloc[i]
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]; af = min(af+step, max_step)
        return ps

    # Keltner Channel
    def keltner(high, low, close, volume, period=20, atr_mult=1.5):
        ctr = ema(close, period)
        tr = atr(high, low, close, period)
        return ctr + atr_mult*tr, ctr, ctr - atr_mult*tr

    # Vortex Indicator
    def vortex_indicator(high, low, close, period=14):
        plus = (high - low.shift(1)).abs()
        minus= (low - high.shift(1)).abs()
        tr    = atr(high, low, close, period)
        tr    = tr.mask(tr.abs() ==0, EPS) 
        vip = plus.rolling(period).sum() / tr.rolling(period).sum()
        vin = minus.rolling(period).sum() / tr.rolling(period).sum()
        return vip, vin

    # Ultimate Oscillator
    def ultimate_oscillator(high, low, close, p1=7, p2=14, p3=28):
        bp = close - pd.concat([low, close.shift()], axis=1).max(axis=1)
        tr = pd.concat([high, close.shift()], axis=1).max(axis=1) - \
            pd.concat([low,  close.shift()], axis=1).min(axis=1)
        t1 = tr.rolling(p1).sum().mask(tr.rolling(p1).sum() ==0, EPS)
        t2 = tr.rolling(p2).sum().mask(tr.rolling(p2).sum() ==0, EPS)
        t3 = tr.rolling(p3).sum().mask(tr.rolling(p3).sum() ==0, EPS)
        avg1 = bp.rolling(p1).sum() / t1
        avg2 = bp.rolling(p2).sum() / t2
        avg3 = bp.rolling(p3).sum() / t3    
        return 100*(4*avg1 + 2*avg2 + avg3)/7

    # True Strength Index
    def tsi(s, r=25, s2=13):
        diff = s.diff()
        ema1 = ema(diff, r)
        ema2 = ema(ema1, s2)
        absema1 = ema(diff.abs(), r)
        absema2 = ema(absema1, s2)
        absema2 = absema2.mask(absema2.abs() ==0, EPS)
        return 100 * ema2 / absema2

    # Ease of Movement
    def eom(high, low, close, volume, period=14):
        vol_mid = (high + low) / 2
        prev_mid = vol_mid.shift(1)
        vol_eps = volume.mask(volume.abs() ==0, EPS)
        box = (high - low) / vol_eps 
        em = (vol_mid - prev_mid) / box
        return em.rolling(period).mean()
    # Moved to DataProcessing.prepare_features() method
    def prepare_features(df: pd.DataFrame, symbol: str = None, training: bool = False):
        """Wrapper function - redirects to DataProcessing.prepare_features()"""
        data_processing = DataProcessing()
        return data_processing.prepare_features(df, symbol, training)
    
    def label_spikes(df: pd.DataFrame,
                    price_col: str = 'close',
                    horizon: int = 12,
                    threshold: float = 0.01) -> pd.DataFrame:
        """Wrapper function - redirects to DataProcessing.label_spikes()"""
        data_processing = DataProcessing()
        return data_processing.label_spikes(df, price_col, horizon, threshold)
        
    def analyze_historical_events(symbol=None):
        """
        Phân tích tác động của các sự kiện lịch sử lớn đối với biến động giá của đồng coin (symbol) được chỉ định.
        Trả về danh sách kết quả cho từng sự kiện, bao gồm phần trăm thay đổi giá vào ngày sự kiện, ngày sau đó và sau 7 ngày.
        """
        if symbol is None:
            symbol = settings.DEFAULT_symbol if hasattr(settings, "DEFAULT_symbol") else "BTCUSDT"
        results = []  # danh sách kết quả cho mỗi sự kiện

        # Duyệt qua từng sự kiện lịch sử
        for event in HISTORICAL_EVENTS:
            event_date_str = event["date"]
            event_name = event["name"]
            event_dt = datetime.strptime(event_date_str, "%Y-%m-%d")
            symbol_data = None

            # Xác định cặp giao dịch và nguồn dữ liệu cho phù hợp
            # Ưu tiên Binance (nếu có dữ liệu), nếu không thì thử symbolbase
            use_binance = True
            api_symbol = symbol
            if "-" in symbol or symbol.endswith("USD") and not symbol.endswith("USDT"):
                # Nếu symbol dạng "COIN-USD" (Coinbase) thì chuyển sang dùng Coinbase API
                use_binance = False

            # Tính khoảng thời gian 7 ngày trước và 7 ngày sau sự kiện
            start_dt = event_dt - timedelta(days=8)  # lấy sớm hơn 1 ngày để đảm bảo có dữ liệu ngày trước
            end_dt = event_dt + timedelta(days=8)
            start_ts = int(start_dt.timestamp() * 1000)
            end_ts = int(end_dt.timestamp() * 1000)

            prices = {}  # lưu giá đóng cửa trước/sau sự kiện
            try:
                if use_binance:
                    # Gọi API lịch sử nến của Binance (interval 1 ngày)
                    url = (f"https://api.binance.com/api/v3/klines?symbol={api_symbol}"
                        f"&interval=1d&startTime={start_ts}&endTime={end_ts}")
                    response = requests.get(url, timeout=60)
                    data = response.json() if response.status_code == 200 else []
                    # Duyệt qua kết quả và lưu giá đóng cửa theo ngày
                    for candle in data:
                        # Mỗi candle: [open_time, open, high, low, close, volume, close_time, ...]
                        open_time = candle[0]
                        close_price = float(candle[4])
                        # Chuyển open_time (ms) về ngày UTC
                        date = datetime.utcfromtimestamp(open_time/1000.0).date()
                        prices[date] = close_price

                    # Nếu không có dữ liệu từ Binance (có thể do symbol không tồn tại hoặc quá xa), thử Coinbase
                    if len(prices) == 0:
                        use_binance = False  # chuyển sang coinbase
                if not use_binance:
                    # Sử dụng Coinbase API (nếu symbol dạng COIN-USD hoặc Binance không có dữ liệu)
                    # Đảm bảo format symbol cho Coinbase (ví dụ BTC-USD)
                    product = symbol if "-" in symbol else symbol[:-3] + "-" + symbol[-3:]
                    url = (f"https://api.pro.coinbase.com/products/{product}/candles"
                        f"?start={start_dt.strftime('%Y-%m-%d')}&end={end_dt.strftime('%Y-%m-%d')}&granularity=86400")
                    response = requests.get(url, timeout=5)
                    data = response.json() if response.status_code == 200 else []
                    # API Coinbase trả về mảng [time, low, high, open, close, volume] theo thứ tự giảm dần thời gian
                    # Sắp xếp lại theo thời gian tăng dần
                    data.sort(key=lambda x: x[0])
                    for candle in data:
                        time_sec = candle[0]
                        close_price = float(candle[4])
                        date = datetime.utcfromtimestamp(time_sec).date()
                        prices[date] = close_price
            except Exception as e:
                print(f"Lỗi khi lấy dữ liệu cho sự kiện {event_name}: {e}")

            # Tính toán tác động giá nếu có dữ liệu
            if len(prices) > 0:
                event_date = event_dt.date()
                prev_date = event_date - timedelta(days=1)
                next_date = event_date + timedelta(days=1)
                week_date = event_date + timedelta(days=7)

                # Lấy giá đóng cửa của ngày trước, ngày sự kiện, ngày sau và 7 ngày sau
                price_before = prices.get(prev_date)
                price_event = prices.get(event_date)
                price_next = prices.get(next_date)
                price_week = prices.get(week_date)

                # Chỉ tính % thay đổi nếu có đủ dữ liệu trước và sau sự kiện
                if price_before and price_event:
                    change_event = (price_event - price_before) / price_before * 100.0
                else:
                    change_event = None
                if price_event and price_next:
                    change_next = (price_next - price_event) / price_event * 100.0
                else:
                    change_next = None
                if price_event and price_week:
                    change_week = (price_week - price_event) / price_event * 100.0
                else:
                    change_week = None

                # Lưu kết quả vào danh sách
                results.append({
                    "event": event_name,
                    "date": event_date_str,
                    "price_before": price_before,
                    "price_on_event": price_event,
                    "price_next_day": price_next,
                    "price_after_7d": price_week,
                    "pct_change_event_day": round(change_event, 2) if change_event is not None else None,
                    "pct_change_next_day": round(change_next, 2) if change_next is not None else None,
                    "pct_change_after_7d": round(change_week, 2) if change_week is not None else None
                })

                # In ra kết quả phân tích cho sự kiện
                print(f"Sự kiện: {event_name} ({event_date_str})")
                if price_before and price_event:
                    print(f" - Giá trước sự kiện: {price_before:.4f}, giá ngày sự kiện: {price_event:.4f}, thay đổi: {change_event:.2f}%")
                if price_event and price_next:
                    print(f" - Giá ngày kế tiếp: {price_next:.4f}, thay đổi so với ngày sự kiện: {change_next:.2f}%")
                if price_event and price_week:
                    print(f" - Giá sau 7 ngày: {price_week:.4f}, thay đổi so với ngày sự kiện: {change_week:.2f}%")
            else:
                # Không có dữ liệu cho sự kiện này
                results.append({
                    "event": event_name,
                    "date": event_date_str,
                    "error": "No data available for this date/symbol"
                })
        print(f"Sự kiện: {event_name} ({event_date_str}) - Không có dữ liệu giá để phân tích.")

        return results
    
    def add_head_and_shoulders(df: pd.DataFrame, window: int = 30, tol: float = 0.03) -> pd.DataFrame:
        """Wrapper function - redirects to DataProcessing.add_head_and_shoulders()"""
        data_processing = DataProcessing()
        return data_processing.add_head_and_shoulders(df, window, tol)

    # Moved to DataProcessing.add_double_top() method
    def add_double_top(df: pd.DataFrame, window: int = 50, tol: float = 0.02) -> pd.DataFrame:
        """Wrapper function - redirects to DataProcessing.add_double_top()"""
        data_processing = DataProcessing()
        return data_processing.add_double_top(df, window, tol)
        dt    = np.zeros(len(df))
        highs = df['high'].values
        peaks = (highs[1:-1] > highs[:-2]) & (highs[1:-1] > highs[2:])
        idx   = np.where(peaks)[0] + 1
        for j in range(len(idx)-1):
            i1,i2 = idx[j], idx[j+1]
            if i2-i1 <= window and abs(highs[i1]-highs[i2])/highs[i1] < tol:
                dt[i2] = 1
        df['pattern_double_top'] = dt
        return df

# Moved to DataProcessing.add_double_bottom() method    def add_double_bottom(df: pd.DataFrame, window: int = 50, tol: float = 0.02) -> pd.DataFrame:
        """Wrapper function - redirects to DataProcessing.add_double_bottom()"""
        data_processing = DataProcessing()
        return data_processing.add_double_bottom(df, window, tol)

    def add_triangle(df: pd.DataFrame, window: int = 50, slope_thresh: float = 0.01) -> pd.DataFrame:
        tri = np.zeros(len(df))
        for i in range(window, len(df)-window):
            seg = df['close'].iloc[i-window:i+window].values
            x   = np.arange(len(seg)).reshape(-1,1)
            lr  = LinearRegression().fit(x, seg)
            if abs(lr.coef_[0]) < slope_thresh:
                tri[i] = 1
        df['pattern_triangle'] = tri
        return df

    def add_flag(df: pd.DataFrame, window: int = 20, slope_thresh: float = 0.02) -> pd.DataFrame:
        flag = np.zeros(len(df))
        for i in range(window, len(df)):
            seg   = df['close'].iloc[i-window:i].values
            slope = np.polyfit(np.arange(window), seg, 1)[0]
            if abs(slope) < slope_thresh:
                flag[i] = 1
        df['pattern_flag'] = flag
        return df

    def add_pennant(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        pen = np.zeros(len(df))
        for i in range(window*2, len(df)):
            seg   = df['high'].iloc[i-window*2:i].values
            std1  = np.std(seg[:window])
            std2  = np.std(seg[window:])
            if std2 < std1:
                pen[i] = 1
        df['pattern_pennant'] = pen
        return df

    def add_rising_wedge(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        rw = np.zeros(len(df))
        x  = np.arange(window).reshape(-1,1)
        for i in range(window, len(df)):
            seg_high = df['high'].iloc[i-window:i].values
            slope_h  = LinearRegression().fit(x, seg_high).coef_[0]
            seg_low  = df['low'].iloc[i-window:i].values
            slope_l  = LinearRegression().fit(x, seg_low ).coef_[0]
            if slope_h < 0 and slope_l > 0:
                rw[i] = 1
        df['pattern_rising_wedge'] = rw
        return df

    def add_falling_wedge(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        fw = np.zeros(len(df))
        x  = np.arange(window).reshape(-1,1)
        for i in range(window, len(df)):
            seg_high = df['high'].iloc[i-window:i].values
            slope_h  = LinearRegression().fit(x, seg_high).coef_[0]
            seg_low  = df['low'].iloc[i-window:i].values
            slope_l  = LinearRegression().fit(x, seg_low ).coef_[0]
            if slope_h > 0 and slope_l < 0:
                fw[i] = 1
        df['pattern_falling_wedge'] = fw
        return df

    def add_triple_top(df: pd.DataFrame, window: int = 50, tol: float = 0.02) -> pd.DataFrame:
        tt    = np.zeros(len(df))
        highs = df['high'].values
        peaks = (highs[1:-1] > highs[:-2]) & (highs[1:-1] > highs[2:])
        idx   = np.where(peaks)[0] + 1
        for j in range(len(idx)-2):
            i1,i2,i3 = idx[j], idx[j+1], idx[j+2]
            cond1 = i2-i1 <= window and i3-i2 <= window
            cond2 = abs(highs[i1]-highs[i2])<tol*highs[i1]
            cond3 = abs(highs[i2]-highs[i3])<tol*highs[i2]
            if cond1 and cond2 and cond3:
                tt[i3] = 1
        df['pattern_triple_top'] = tt
        return df

    def add_triple_bottom(df: pd.DataFrame, window: int = 50, tol: float = 0.02) -> pd.DataFrame:
        tb     = np.zeros(len(df))
        lows   = df['low'].values
        troughs= (lows[1:-1] < lows[:-2]) & (lows[1:-1] < lows[2:])
        idx    = np.where(troughs)[0] + 1
        for j in range(len(idx)-2):
            i1,i2,i3 = idx[j], idx[j+1], idx[j+2]
            cond1 = i2-i1 <= window and i3-i2 <= window
            cond2 = abs(lows[i1]-lows[i2])<tol*lows[i1]
            cond3 = abs(lows[i2]-lows[i3])<tol*lows[i2]
            if cond1 and cond2 and cond3:
                tb[i3] = 1
        df['pattern_triple_bottom'] = tb
        return df

    def add_rectangle(df: pd.DataFrame, window: int = 30, tol: float = 0.01) -> pd.DataFrame:
        rect = np.zeros(len(df))
        for i in range(window, len(df)):
            seg  = df['close'].iloc[i-window:i].values
            high = seg.max(); low = seg.min()
            if (high - low)/low < tol:
                rect[i] = 1
        df['pattern_rectangle'] = rect
        return df


    # List of pattern types to consider (for consistent feature columns in model features)
    PATTERN_TYPES = [
        "Head and Shoulders", "Inverse Head and Shoulders", "Double Top", "Double Bottom",
        "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle",
        "Bullish Flag", "Bearish Flag", "Rising Wedge", "Falling Wedge"
    ]
    _FE = FeatureEngineer()
    def detect_patterns(df: pd.DataFrame) -> list:
        """
        Detect chart patterns from price data.
        Returns a list of pattern occurrences as dicts with 'type' and 'end' index.
        """
        patterns = []
        highs = df['high']
        lows = df['low']
        n = len(df)
        # Identify local peaks and troughs
        peak_indices = []
        trough_indices = []
        for i in range(1, n-1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peak_indices.append(i)
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                trough_indices.append(i)
        # Head and Shoulders
        for j in range(1, len(peak_indices)-1):
            left_idx = peak_indices[j-1]
            mid_idx = peak_indices[j]
            right_idx = peak_indices[j+1]
            left_h = highs[left_idx]
            mid_h = highs[mid_idx]
            right_h = highs[right_idx]
            shoulders_avg = (left_h + right_h) / 2.0
            if shoulders_avg == 0:
                continue
            if mid_h > shoulders_avg * 1.05 and abs(left_h - right_h) / shoulders_avg < 0.03:
                patterns.append({"type": "Head and Shoulders", "end": right_idx})
        # Inverse Head and Shoulders
        for j in range(1, len(trough_indices)-1):
            left_idx = trough_indices[j-1]
            mid_idx = trough_indices[j]
            right_idx = trough_indices[j+1]
            left_l = lows[left_idx]
            mid_l = lows[mid_idx]
            right_l = lows[right_idx]
            shoulders_avg = (left_l + right_l) / 2.0
            if shoulders_avg == 0:
                continue
            if mid_l < shoulders_avg * 0.95 and abs(left_l - right_l) / shoulders_avg < 0.03:
                patterns.append({"type": "Inverse Head and Shoulders", "end": right_idx})
        # Double Top
        for j in range(len(peak_indices)-1):
            i1 = peak_indices[j]
            i2 = peak_indices[j+1]
            h1 = highs[i1]
            h2 = highs[i2]
            if h1 == 0:
                continue
            if abs(h1 - h2) / h1 < 0.03:
                mid_low = lows[i1+1:i2].min() if i2 > i1 + 1 else lows[i1]
                if mid_low < min(h1, h2) * 0.95:
                    patterns.append({"type": "Double Top", "end": i2})
        # Double Bottom
        for j in range(len(trough_indices)-1):
            i1 = trough_indices[j]
            i2 = trough_indices[j+1]
            l1 = lows[i1]
            l2 = lows[i2]
            if l1 == 0:
                continue
            if abs(l1 - l2) / l1 < 0.03:
                mid_high = highs[i1+1:i2].max() if i2 > i1 + 1 else highs[i1]
                if mid_high > max(l1, l2) * 1.05:
                    patterns.append({"type": "Double Bottom", "end": i2})
        # (Other patterns not explicitly detected for now)
        return patterns

    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the feature DataFrame with technical indicators and pattern features from raw price DataFrame.
        """
        df_feat = compute_indicators(df.copy())
        # Detect patterns and add pattern indicator columns
        pattern_list = detect_patterns(df_feat)
        for pat in pattern_list:
            pat_type = pat['type']
            pat_index = pat.get('end')
            col_name = f"pat_{pat_type.replace(' ', '_').replace('&', 'and')}"
            if col_name not in df_feat.columns:
                df_feat[col_name] = 0
            if pat_index is not None and pat_index < len(df_feat):
                df_feat.at[pat_index, col_name] = 1
        # Ensure all pattern columns exist even if not detected
        for pat_type in PATTERN_TYPES:
            col_name = f"pat_{pat_type.replace(' ', '_').replace('&', 'and')}"
            if col_name not in df_feat.columns:
                df_feat[col_name] = 0
        # Drop columns not needed as features
        if 'open_time' in df_feat.columns:
            df_feat.drop(columns=['open_time'], inplace=True)
        if 'close_time' in df_feat.columns:
            df_feat.drop(columns=['close_time'], inplace=True)
        # Drop any rows with NaN (initial periods without full indicator data)
        df_feat.dropna(inplace=True)
        return df_feat


    # Global GPU manager instance
    gpu_manager = GPUManager()


    # Initialize logger
    logger = get_logger('feature_engineering')

    # ─── 1) Indicator functions ─────────────────────────────────────────────────────
    EPS = 1e-8
    INDICATOR_REGISTRY = {
        # Basic & core
        **{f"ema_{p}":    (lambda df,p=p: ema(df["close"], p))          for p in (10,34,89)},
        **{f"sma_{w}":    (lambda df,w=w: sma(df["close"], w))          for w in (20,50)},
        **{f"rsi_{14}":   (lambda df:    rsi(df["close"], 14))          },
        **{f"macd_{12}_26_9":      (lambda df: macd(df["close"],12,26,9)[0])},
        **{f"macd_sig_12_26_9":    (lambda df: macd(df["close"],12,26,9)[1])},
        "atr_14":    (lambda df:    atr(df["high"], df["low"], df["close"], 14)),
        "cci_20":    (lambda df:    cci(df["high"], df["low"], df["close"], 20)),
        # Oscillators
        "adx_14":    (lambda df:    adx(df["high"], df["low"], df["close"], 14)),
        "stoch_k":   (lambda df:    stoch_rsi(df["close"], 14,3,3)[0]),
        "stoch_d":   (lambda df:    stoch_rsi(df["close"], 14,3,3)[1]),
        "willr_14":  (lambda df:    williams_r(df["high"], df["low"], df["close"], 14)),
        "uo":        (lambda df:    ultimate_oscillator(df["high"], df["low"], df["close"], 7,14,28)),
        "tsi":       (lambda df:    tsi(df["close"], 25,13)),
        "eom_14":    (lambda df:    eom(df["high"], df["low"], df["close"], df["volume"],14)),
        "trix_18":   (lambda df:    trix(df["close"], 18)),
        # Volume‐based
        "obv":       (lambda df:    obv(df["close"], df["volume"])),
        "mfi_14":    (lambda df:    mfi(df["high"], df["low"], df["close"], df["volume"],14)),
        "cmf_20":    (lambda df:    cmf(df["high"], df["low"], df["close"], df["volume"],20)),
        # Channels / bands
        **{f"bb_up":    (lambda df: bollinger_bands(df["close"],20,2)[0])},
        **{f"bb_mid":   (lambda df: bollinger_bands(df["close"],20,2)[1])},
        **{f"bb_lo":    (lambda df: bollinger_bands(df["close"],20,2)[2])},
        **{f"psar":     (lambda df: psar(df["high"], df["low"], df["close"], 0.02,0.2))},
        **{f"kc_up":    (lambda df: keltner(df["high"],df["low"],df["close"],df["volume"],14,1.5)[0])},
        **{f"kc_mid":   (lambda df: keltner(df["high"],df["low"],df["close"],df["volume"],14,1.5)[1])},
        **{f"kc_lo":    (lambda df: keltner(df["high"],df["low"],df["close"],df["volume"],14,1.5)[2])},
        **{f"vi_plus":  (lambda df: vortex_indicator(df["high"],df["low"],df["close"],14)[0])},
        **{f"vi_minus": (lambda df: vortex_indicator(df["high"],df["low"],df["close"],14)[1])},
        # FFT
        **{f"fft_{i}":  (lambda df,i=i: np.real(fft.fft(df["close"].ffill().bfill().values)[i]))
                    for i in range(1,6)},
    }

    # Registry cho composite features (kết hợp nhiều chỉ báo)
        COMPOSITE_REGISTRY = {
        # MACD histogram
        "macd_hist":          lambda df: df["macd"] - df["macd_signal"],
        # Độ rộng Bollinger Bands
        "bb_width":           lambda df: df["bb_upper"] - df["bb_lower"],
        # Hiệu giữa 2 EMA 20 và 89
        "ema_diff":           lambda df: df["ema_20"] - df["ema_89"],
        # Tỷ lệ ATR trên close
        "atr_ratio":          lambda df: df["atr"] / df["close"],
        # Cross‐terms core
        "rsi_x_adx":     lambda df: df["rsi"] * df["adx"],
        "rsi_x_macd":    lambda df: df["rsi"] * df["macd"],
        "rsi_x_atr":     lambda df: df["rsi"] * df["atr"],
        "macd_x_rsi":    lambda df: df["macd"] * df["rsi"],
        "bb_upper_x_bb_lower":lambda df: df["bb_upper"] * df["bb_lower"],
        }

class Training:
    """
    Training Class - AI model training with accuracy optimization
    
    Handles all model training operations including:
    - Multi-threaded training orchestration
    - Individual model trainers (XGBoost, LightGBM, etc.)
    - Hyperparameter optimization
    - Cross-validation and backtesting
    - Model evaluation and selection
    """
    
    def __init__(self):
        """Initialize training with global instances"""
        self.unified_trainer = _unified_trainer  # Reference to global instance
        self.enhanced_trainer = None  # Will be initialized when needed
        self.resource_optimizer = global_resource_optimizer
        self.gpu_manager = gpu_manager
        self.logger = get_logger('training')
    
    def train_all_models(self, symbol: str):
        """Train comprehensive AI models (multi-timeframe) for a given symbol."""
        logger.info(f"Bắt đầu huấn luyện mô hình AI cho {symbol}...")
        
        try:
            # Consolidated data loading and drift detection
            df_combined = self._handle_drift_detection_and_data_loading(symbol)
              # Sử dụng prepare_features để tạo X, y
            X, y = prepare_features(df_combined, symbol=symbol, training=True)
            logger.info(f"Đã tính toán xong đặc trưng. Số mẫu: {len(X)}, Số đặc trưng: {X.shape[1]}")
            
            # 3. Train models using UnifiedModelTrainer
            model_candidates = self._train_models_with_fallback(df_combined, X, y)
            
            # 4. Select best model and evaluate
            best_model, best_name, best_score = self._select_best_model(model_candidates, X, y)
            
            # 5. Save model and return results
            return self._save_model_and_return_results(symbol, best_model, best_name, best_score)
            
        except Exception as e:            logger.error(f"Error in train_all_models for {symbol}: {e}")
            raise
    
    def _handle_drift_detection_and_data_loading(self, symbol: str):
        """Consolidated drift detection and data loading logic"""
        try:
            # Load previous training data if exists
            df_old = pd.read_parquet(f"cache/{symbol}_last_train.parquet")
            data_frames = update_data(symbol)
            df_new = pd.concat(data_frames, ignore_index=True) if isinstance(data_frames, list) else data_frames
            
            # Perform drift detection
            dd = DriftDetector(alpha=0.05)
            drift_res = dd.compare(df_old, df_new)
            if drift_res.get("drift_detected"):
                logger.warning(f"[DriftDetector] Phát hiện drift: {drift_res}")
            
            # Save new data for next comparison
            df_new.to_parquet(f"cache/{symbol}_last_train.parquet", index=False)
            return df_new
            
        except FileNotFoundError:
            logger.info("[DriftDetector] Chưa có dữ liệu cũ, sẽ tạo cache lần này.")
            return self._load_and_cache_data(symbol)
        except Exception as e:
            logger.warning(f"[DriftDetector] Lỗi khi so sánh drift: {e}")
            return self._load_and_cache_data(symbol)
    
    def _load_and_cache_data(self, symbol: str):
        """Consolidated data loading and caching logic"""
        data_frames = update_data(symbol)
        df_combined = pd.concat(data_frames, ignore_index=True) if isinstance(data_frames, list) else data_frames
        df_combined.to_parquet(f"cache/{symbol}_last_train.parquet", index=False)
        return df_combined
    
    def _train_models_with_fallback(self, df_combined, X, y):
        """Consolidated model training with fallback logic"""
        trainer = UnifiedModelTrainer()
        model_candidates = []
        
        # Primary training attempt
        try:
            best_model, metrics = trainer.train_all_models(
                df=df_combined, 
                timeframe="multi_frame", 
                model_type=None, 
                progress_callback=None,
                X_override=X,
                y_override=y
            )
            
            if best_model and metrics:
                model_candidates.append((metrics.get('model_name', 'unknown'), best_model, metrics.get('accuracy', 0.0)))
                logger.info(f"Unified training completed. Best model: {metrics.get('model_name')}, accuracy: {metrics.get('accuracy', 0.0):.4f}")
            else:
                logger.error("Unified training failed to produce a model")
                
        except Exception as e:
            logger.error(f"Unified training failed: {e}")
            logger.info("Attempting individual model training as fallback...")
            
            # Fallback with predefined trainer functions
            fallback_trainers = [
                ("random_forest", trainer.train_random_forest),
                ("xgboost", trainer.train_xgboost),
                ("lightgbm", trainer.train_lightgbm),
                ("catboost", trainer.train_catboost),
                ("mlp", trainer.train_neural_network),
                ("logistic", trainer.train_logistic_regression)
            ]
            
            for model_name, train_func in fallback_trainers:
                try:
                    model, score = train_func(X, y)
                    if model and score > 0:
                        model_candidates.append((model_name, model, score))
                        logger.info(f"Fallback training - {model_name}: accuracy={score:.4f}")
                except Exception as train_error:
                    logger.error(f"Fallback training failed for {model_name}: {train_error}")
        
        return model_candidates
    
    def _select_best_model(self, model_candidates, X, y):
        """Consolidated best model selection and evaluation logic"""
        best_model, best_name, best_score = None, None, -1
        
        # Find best model by score
        for name, model, score in model_candidates:
            if model is not None and score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        if best_model is None:
            raise RuntimeError("Không có mô hình nào huấn luyện thành công.")
        
        logger.info(f"Mô hình tốt nhất: {best_name} (Accuracy = {best_score:.4f})")
        
        # Compute additional evaluation metrics
        try:
            preds = best_model.predict(X)
            acc = accuracy_score(y, preds)
            prec = precision_score(y, preds)
            rec = recall_score(y, preds)
            f1 = f1_score(y, preds)
            logger.info(f"Đánh giá trên tập huấn luyện - Accuracy={acc:.2%}, Precision={prec:.2%}, Recall={rec:.2%}, F1={f1:.2%}")
        except Exception as e:
            logger.warning(f"Không thể tính thêm metric: {e}")
        
        return best_model, best_name, best_score
    
    def _save_model_and_return_results(self, symbol: str, best_model, best_name: str, best_score: float):
        """Consolidated model saving and result preparation logic"""
        # Save best model
        os.makedirs("ai_models/trained_models", exist_ok=True)
        model_filename = f"ai_models/trained_models/{symbol.upper()}_{best_name}.pkl"
        joblib.dump(best_model, model_filename)
        logger.info(f"Đã lưu mô hình {best_name} vào file: {model_filename}")
        
        # Return result info
        return {
            "symbol": symbol,
            "best_model_name": best_name,
            "accuracy": best_score
        }
      def train_trading_agent(self, prices: list, timesteps: int = 1000000):
        """
        Train a reinforcement learning agent (PPO) on the given price data.
        - prices: historical price data for the environment.
        - timesteps: number of training timesteps for the RL algorithm.
        Returns the trained model (e.g., a Stable-Baselines3 PPO model).
        """
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            import torch
            
            # Create the trading environment instance
            env = TradingEnv(prices)
            # Wrap in a DummyVecEnv required by Stable-Baselines3 for vectorized environments
            env = DummyVecEnv([lambda: env])
            # Initialize the PPO agent with a simple MLP policy
            model = PPO("MlpPolicy", env, verbose=0,
                    device="cuda" if torch.cuda.is_available() else "cpu")
            # Train the agent for the specified number of timesteps
            model.learn(total_timesteps=timesteps)
            return model
        except ImportError:
            self.logger.error("Required packages for RL training not available")
            return None
        except Exception as e:
            self.logger.error(f"Error training RL agent: {e}")
            return None
      def train_reinforcement(self, prices, timesteps=10000):
        """Legacy reinforcement learning training method - redirects to unified implementation"""
                return self.train_trading_agent(prices, timesteps)
    
    @staticmethod
    def train_model(df, tf, model_type=None, n_epochs=20, progress_callback=None, X_override=None, y_override=None):
        """
        Unified train_model function - optimized implementation
        Uses UnifiedModelTrainer for consistent training across the project
        """
        trainer = Training.UnifiedModelTrainer()
        return trainer.train_all_models(df, tf, model_type, progress_callback, X_override, y_override)    # ================== TRAINING CONFIGURATION CLASSES ==================
    
    class TrainingResult:
        """Container for training results - consolidated from multiple implementations"""
        def __init__(self, 
                     model: Any = None, 
                     model_name: str = "", 
                     accuracy: float = 0.0, 
                     f1_score: float = 0.0, 
                     threshold: float = 0.5, 
                     backtest_results: pd.DataFrame = None, 
                     training_time: float = 0.0, 
                     error: Optional[str] = None):
            self.model = model
            self.model_name = model_name
            self.accuracy = accuracy
            self.f1_score = f1_score
            self.threshold = threshold
            self.backtest_results = backtest_results if backtest_results is not None else pd.DataFrame()
            self.training_time = training_time
            self.error = error

    class TrainingConfig:
        """Unified configuration for training - consolidates ModelTrainingConfig and removes duplication"""
        
        def __init__(self):
            # Basic training configuration
            self.max_workers = 4
            self.use_gpu = True
            self.enable_backtesting = True
            self.optimize_threshold = True
            self.backtest_window_size = 500
            self.backtest_step_size = 100
            self.threshold_metric = 'f1'  # 'accuracy', 'precision', 'recall', 'f1'
            self.models_to_train = None  # None = all models
            
            # Model training specific configuration (merged from ModelTrainingConfig)
            self.cv_folds = getattr(settings, 'CV_FOLDS', 3)
            self.optuna_trials = getattr(settings, 'OPTUNA_TRIALS', 30)
            self.min_samples_per_class = getattr(settings, 'MIN_SAMPLES_PER_CLASS', 5)
            self.min_total_samples = getattr(settings, 'MIN_TOTAL_SAMPLES', 20)
            self.early_stopping_threshold = 0.95
            self.max_retries = 3
      class UnifiedModelTrainer:
        """Unified trainer for all model types"""
        
        def __init__(self, config: 'TrainingConfig' = None):
            self.config = config or TrainingConfig()
            self.trained_models = {}
            self.best_model = None
            self.best_score = 0.0
            self.best_name = None
            
            # Pre-defined parameter grids to reduce duplication
            self._tree_param_grid = {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", 0.5],
                "criterion": ["gini", "entropy"]
            }
            
        def _setup_training_environment(self, model_name: str):
            """Consolidated setup for training environment"""
            logger.info(f"🎯 Training {model_name}...")
            resource_optimizer.throttle_ram()
            
            cv = self._get_cv_strategy(self.current_y)
            n_workers = resource_optimizer.suggest_workers(current_workers=os.cpu_count())
            
            return cv, n_workers
            
        def _execute_training_with_retry(self, search_cv, model_name: str) -> Tuple[Any, float]:
            """Consolidated retry logic for training"""
            best_model, best_score = None, 0.0
            
            for attempt in range(self.config.max_retries):
                try:
                    search_cv.fit(self.current_X, self.current_y)
                    if search_cv.best_score_ > best_score:
                        best_model = search_cv.best_estimator_
                        best_score = search_cv.best_score_
                        
                    if best_score >= self.config.early_stopping_threshold:
                        break
                        
                except Exception as e:
                    logger.warning(f"{model_name} training attempt {attempt + 1} failed: {e}")
                    continue
            
            gc.collect()
            return best_model, best_score
            """Validate training data"""
            if len(X) < self.config.min_total_samples:
                logger.warning(f"Insufficient samples: {len(X)} < {self.config.min_total_samples}")
                return False
                
            unique_labels = np.unique(y)
            if len(unique_labels) < 2:
                logger.warning(f"Insufficient classes: {len(unique_labels)} < 2")
                return False
                
            # Check minimum samples per class
            label_counts = Counter(y)
            min_count = min(label_counts.values())
            if min_count < self.config.min_samples_per_class:
                logger.warning(f"Insufficient samples per class: {min_count} < {self.config.min_samples_per_class}")
                return False
                
            return True
        
        def _apply_class_balancing(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
            """Apply SMOTE or undersampling for class balancing"""
            if not IMBALANCED_AVAILABLE:
                logger.info("⚠️ Imbalanced-learn not available, skipping class balancing")
                return X, y
                
            try:
                counts = Counter(y)
                majority_count = max(counts.values())
                minority_count = min(counts.values())
                
                # Only balance if significantly imbalanced
                if minority_count / majority_count < 0.8 and minority_count >= 2:
                    
                    # Check available memory
                    mem = psutil.virtual_memory()
                    
                    if mem.available / mem.total < 0.3:  # Less than 30% memory available
                        # Use undersampling
                        rus = RandomUnderSampler(random_state=42)
                        X_balanced, y_balanced = rus.fit_resample(X, y)
                        logger.info(f"📊 Applied undersampling: {Counter(y_balanced)}")
                    else:
                        # Use SMOTE
                        k_neighbors = min(3, minority_count - 1)
                        if k_neighbors > 0:
                            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
                            X_balanced, y_balanced = smote.fit_resample(X, y)
                            logger.info(f"📊 Applied SMOTE: {Counter(y_balanced)}")
                        else:
                            logger.warning("Not enough minority samples for SMOTE")
                            return X, y
                    
                    return X_balanced, y_balanced
                else:
                    logger.info("📊 Classes already balanced, skipping resampling")
                    return X, y
                    
            except Exception as e:
                logger.error(f"Class balancing failed: {e}")
                return X, y
        
        def _get_cv_strategy(self, y: Union[pd.Series, np.ndarray]) -> StratifiedKFold:
            """Get appropriate cross-validation strategy"""
            counts = Counter(y)
            min_count = min(counts.values())
            n_splits = min(self.config.cv_folds, min_count, len(y) // 10)
            n_splits = max(2, n_splits)  # Minimum 2 splits
            
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        def _train_tree_based_model(self, model_class, param_grid: dict, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
            """Train tree-based models (RF, ET)"""
            cv = self._get_cv_strategy(y)
            n_workers = resource_optimizer.suggest_workers(current_workers=os.cpu_count())
            
            model = model_class(
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
            
            search = RandomizedSearchCV(
                model, param_grid, n_iter=10, cv=cv,
                scoring="accuracy", n_jobs=n_workers, random_state=42,
                error_score='raise', verbose=0
            )
            
            best_model, best_score = None, 0.0
            
            for attempt in range(self.config.max_retries):
                try:
                    search.fit(X, y)
                    if search.best_score_ > best_score:
                        best_model = search.best_estimator_
                        best_score = search.best_score_
                        
                    if best_score >= self.config.early_stopping_threshold:
                        break
                        
                except Exception as e:
                    logger.warning(f"Training attempt {attempt + 1} failed: {e}")
                    continue
            
            gc.collect()
            return best_model, best_score
          def _train_gradient_boosting_model(self, model_class, param_grid: dict, X: pd.DataFrame, y: pd.Series, use_gpu: bool = False) -> Tuple[Any, float]:
            """Train gradient boosting models (XGB, LGBM, CatBoost) with GPU support"""
            
            # GPU configuration
            gpu_config = {}
            if use_gpu and gpu_manager.use_gpu:
                gpu_id = gpu_manager.select_best_gpu()
                if gpu_id is not None:
                    gpu_manager.set_gpu_device(gpu_id)
                    
                    if model_class == XGBClassifier:
                        gpu_config = gpu_manager.get_gpu_config_for_xgboost()
                    elif model_class == LGBMClassifier:
                        gpu_config = gpu_manager.get_gpu_config_for_lightgbm()
                    elif model_class == CatBoostClassifier:
                        gpu_config = gpu_manager.get_gpu_config_for_catboost()
                    
                    logger.info(f"🚀 Using GPU {gpu_id} for {model_class.__name__}")
                else:
                    logger.info(f"💻 Using CPU for {model_class.__name__}")
            
            # Common parameters
            common_params = {
                'random_state': 42
            }
            
            # Model-specific silent parameters
            if model_class == XGBClassifier:
                common_params.update({
                    'use_label_encoder': False,
                    'eval_metric': 'logloss',
                    'verbosity': 0
                })
            elif model_class == LGBMClassifier:
                common_params.update({
                    'verbose': -1,
                    'verbosity': -1
                })
            elif model_class == CatBoostClassifier:
                common_params.update({
                    'silent': True,
                    'verbose': False
                })
            
            # Combine GPU config with common params
            common_params.update(gpu_config)
            
            cv = self._get_cv_strategy(y)
            n_workers = resource_optimizer.suggest_workers(current_workers=os.cpu_count())
            
            model = model_class(**common_params)
            
            # Test GPU functionality if enabled
            if use_gpu and gpu_manager.use_gpu:
                test_size = min(50, len(X))
                if not gpu_manager.test_gpu_functionality(model_class, X.iloc[:test_size], y.iloc[:test_size]):
                    # Fallback to CPU
                    logger.warning(f"⚠️ GPU test failed for {model_class.__name__}, falling back to CPU")
                    if model_class == XGBClassifier:
                        common_params.update({'tree_method': 'hist', 'predictor': 'auto'})
                        common_params.pop('gpu_id', None)
                    elif model_class == LGBMClassifier:
                        common_params.update({'device': 'cpu'})
                        common_params.pop('gpu_device_id', None)
                    elif model_class == CatBoostClassifier:
                        common_params.update({'task_type': 'CPU'})
                        common_params.pop('devices', None)
                    
                    model = model_class(**common_params)
            
            search = RandomizedSearchCV(
                model, param_grid, n_iter=10, cv=cv,
                scoring="accuracy", n_jobs=n_workers, random_state=42,
                error_score='raise', verbose=0
            )
            
            best_model, best_score = self._execute_training_with_retry(search, model_class.__name__)
            
            gpu_manager.cleanup_gpu_memory()
            return best_model, best_score
          def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
            """Train Random Forest model"""
            logger.info("🌲 Training Random Forest...")
            return self._train_tree_based_model(RandomForestClassifier, self._tree_param_grid, X, y)
          def train_extra_trees(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
            """Train Extra Trees model"""
            logger.info("🌳 Training Extra Trees...")
            return self._train_tree_based_model(ExtraTreesClassifier, self._tree_param_grid, X, y)
          def train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
            """Train XGBoost model"""
            if not XGBOOST_AVAILABLE:
                logger.warning("XGBoost not available")
                return None, 0.0
                
            logger.info("🚀 Training XGBoost...")
            cv, n_workers = self._setup_training_environment("XGBoost")
            
            param_grid = {
                "n_estimators": [100, 200, 500],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_alpha": [0, 0.1, 1],
                "reg_lambda": [1, 5, 10]
            }
            
            return self._train_gradient_boosting_model(XGBClassifier, param_grid, X, y, use_gpu=True)
          def train_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
            """Train LightGBM model"""
            if not LIGHTGBM_AVAILABLE:
                logger.warning("LightGBM not available")
                return None, 0.0
                
            logger.info("💡 Training LightGBM...")
            cv, n_workers = self._setup_training_environment("LightGBM")
            
            param_grid = {
                "n_estimators": [200, 500, 1000],
                "num_leaves": [31, 63, 127],
                "learning_rate": [0.01, 0.05],
                "reg_alpha": [0.0, 0.1, 1],
                "reg_lambda": [0.0, 0.1, 1]
            }
            
            return self._train_gradient_boosting_model(LGBMClassifier, param_grid, X, y, use_gpu=True)
          def train_catboost(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
            """Train CatBoost model"""
            if not CATBOOST_AVAILABLE:
                logger.warning("CatBoost not available")
                return None, 0.0
                
            logger.info("🐱 Training CatBoost...")
            cv, n_workers = self._setup_training_environment("CatBoost")
            
            param_grid = {
                "iterations": [100, 200, 500],
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "l2_leaf_reg": [3, 5, 7],
                "border_count": [128, 254]
            }
            
            return self._train_gradient_boosting_model(CatBoostClassifier, param_grid, X, y, use_gpu=True)
          def train_neural_network(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
            """Train Neural Network (MLP)"""
            logger.info("🧠 Training Neural Network...")
            cv, n_workers = self._setup_training_environment("Neural Network")
            
            param_grid = {
                "hidden_layer_sizes": [(100,), (50, 50), (100, 50)],
                "alpha": [1e-4, 1e-3, 1e-2],
                "learning_rate_init": [1e-3, 1e-2],
                "solver": ["adam"],
                "activation": ["relu", "tanh"]
            }
            
            model = MLPClassifier(
                max_iter=1000,
                early_stopping=True,
                random_state=42
            )
            
            search = RandomizedSearchCV(
                model, param_grid, n_iter=5, cv=cv,
                scoring="accuracy", n_jobs=n_workers, random_state=42,
                error_score='raise', verbose=0
            )
            
            return self._execute_training_with_retry(search, "Neural Network")
          def train_logistic_regression(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
            """Train Logistic Regression"""
            logger.info("📈 Training Logistic Regression...")
            cv, n_workers = self._setup_training_environment("Logistic Regression")
            
            param_grid = {
                "C": [1e-3, 1e-2, 1e-1, 1, 10],
                "penalty": ["l2", "l1"],
                "solver": ["saga"],
                "class_weight": ["balanced", None]
            }
            
            model = LogisticRegression(
                max_iter=3000,
                solver='saga',
                n_jobs=-1,
                random_state=42
            )
            
            search = GridSearchCV(
                model, param_grid, cv=cv,
                scoring="accuracy", n_jobs=-1
            )
            
            return self._execute_training_with_retry(search, "Logistic Regression")
          def train_svm(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
            """Train Support Vector Machine"""
            logger.info("🎯 Training SVM...")
            cv, n_workers = self._setup_training_environment("SVM")
            
            param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"]
            }
            
            model = SVC(
                probability=True,
                class_weight='balanced',
                random_state=42
            )
            
            search = RandomizedSearchCV(
                model, param_grid, n_iter=5, cv=cv,
                scoring="accuracy", n_jobs=n_workers, random_state=42,
                error_score='raise', verbose=0
            )
            
            return self._execute_training_with_retry(search, "SVM")
        
        def create_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float, str]:
            """Create ensemble models from trained models"""
            if len(self.trained_models) < 2:
                logger.info("Not enough models for ensemble")
                return None, 0.0, ""
            
            # Get top 3 models
            sorted_models = sorted(self.trained_models.items(), key=lambda x: x[1][1], reverse=True)[:3]
            estimators = [(name, model[0]) for name, model in sorted_models]
            
            cv = self._get_cv_strategy(y)
            best_ensemble_model, best_ensemble_score, best_ensemble_name = None, 0.0, ""
            
            # Voting Classifier
            try:
                logger.info("🗳️ Creating Voting Classifier...")
                voting_classifier = VotingClassifier(estimators=estimators, voting='soft')
                voting_scores = cross_val_score(voting_classifier, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
                voting_score = voting_scores.mean()
                
                if voting_score > best_ensemble_score:
                    best_ensemble_model = voting_classifier
                    best_ensemble_score = voting_score
                    best_ensemble_name = "voting"
                    
            except Exception as e:
                logger.warning(f"Voting classifier failed: {e}")
            
            # Stacking Classifier
            try:
                logger.info("🏗️ Creating Stacking Classifier...")
                stacking_classifier = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(solver='saga', max_iter=1000, n_jobs=-1, random_state=42),
                    cv=cv,
                    n_jobs=-1
                )
                stacking_scores = cross_val_score(stacking_classifier, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
                stacking_score = stacking_scores.mean()
                
                if stacking_score > best_ensemble_score:
                    best_ensemble_model = stacking_classifier
                    best_ensemble_score = stacking_score
                    best_ensemble_name = "stacking"
                    
            except Exception as e:
                logger.warning(f"Stacking classifier failed: {e}")
            
            return best_ensemble_model, best_ensemble_score, best_ensemble_name
        
        def compute_metrics(self, model, X: pd.DataFrame, y: pd.Series) -> dict:
            """Compute comprehensive metrics for the model"""
            try:
                labels = np.unique(y)
                
                if len(labels) == 2 and hasattr(model, "predict_proba"):
                    # Binary classification with probability
                    y_proba = model.predict_proba(X)[:, 1]
                    precision, recall, thresholds = precision_recall_curve(y, y_proba)
                    f1_scores = [2 * p * r / (p + r) if (p + r) else 0 for p, r in zip(precision, recall)]
                    optimal_threshold = float(thresholds[np.argmax(f1_scores[:-1])] if len(thresholds) else 0.5)
                    y_pred = (y_proba >= optimal_threshold).astype(int)
                    
                    metrics = {
                        "accuracy": accuracy_score(y, y_pred),
                        "precision": precision_score(y, y_pred, average='weighted', zero_division=0),
                        "recall": recall_score(y, y_pred, average='weighted', zero_division=0),
                        "f1": f1_score(y, y_pred, average='weighted', zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                        "threshold": optimal_threshold
                    }
                else:
                    # Multiclass or binary without probability
                    y_pred = model.predict(X)
                    metrics = {
                        "accuracy": accuracy_score(y, y_pred),
                        "precision": precision_score(y, y_pred, average='weighted', zero_division=0),
                        "recall": recall_score(y, y_pred, average='weighted', zero_division=0),
                        "f1": f1_score(y, y_pred, average='weighted', zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(y, y_pred)
                    }
                    
                return metrics
                
            except Exception as e:
                logger.error(f"Metrics computation failed: {e}")
                return {"error": str(e)}
        
        def train_all_models(self, df: pd.DataFrame, timeframe: str, model_type: str = None, 
                            progress_callback=None, X_override=None, y_override=None) -> Tuple[Any, dict]:
            """
            Train all available models and return the best one
            
            Args:
                df: Input dataframe
                timeframe: Timeframe string
                model_type: Specific model type to train (None for all)
                progress_callback: Progress callback function
                X_override: Pre-prepared features
                y_override: Pre-prepared labels
                
            Returns:
                Tuple of (best_model, metrics)
            """
            
            def update_progress(step, total_steps, status=""):
                if progress_callback:
                    progress_callback(step / total_steps, status)
            
            logger.info(f"🚀 Starting unified model training for {timeframe}")
            start_time = time.time()
            
            # Step 1: Prepare features
            update_progress(1, 10, "Preparing features...")
            if X_override is None or y_override is None:
                X, y = prepare_features(df, training=True)
            else:
                X, y = X_override, y_override
                
            # Validate data
            if not self._validate_data(X, y):
                logger.error("Data validation failed")
                return None, None
            
            # Step 2: Scale features
            update_progress(2, 10, "Scaling features...")
            if X_override is None or y_override is None:
                scaler = StandardScaler()
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            else:
                scaler = StandardScaler()
                if len(X) > 0:
                    numeric_cols = X.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        scaler.fit(X[numeric_cols].iloc[:min(10, len(X))])
            
            # Step 3: Apply class balancing
            update_progress(3, 10, "Balancing classes...")
            X, y = self._apply_class_balancing(X, y)
            
            # Define training functions
            training_functions = {
                "random_forest": self.train_random_forest,
                "extra_trees": self.train_extra_trees,
                "xgboost": self.train_xgboost,
                "lightgbm": self.train_lightgbm,
                "catboost": self.train_catboost,
                "mlp": self.train_neural_network,
                "logistic": self.train_logistic_regression,
                "svm": self.train_svm
            }
            
            # Filter training functions
            if model_type:
                training_functions = {model_type: training_functions.get(model_type)}
                training_functions = {k: v for k, v in training_functions.items() if v is not None}
            
            # Step 4-8: Train individual models
            step = 4
            for name, train_func in training_functions.items():
                update_progress(step, 10, f"Training {name}...")
                try:
                    model, score = train_func(X, y)
                    if model is not None and score > 0:
                        self.trained_models[name] = (model, score)
                        if score > self.best_score:
                            self.best_model = model
                            self.best_score = score
                            self.best_name = name
                        logger.info(f"✅ {name} completed with score={score:.4f}")
                    else:
                        logger.warning(f"❌ {name} training failed")
                except Exception as e:
                    logger.error(f"❌ {name} training error: {e}")
                
                step += 1
                if step > 8:  # Limit to available steps
                    break
            
            # Step 9: Create ensemble if multiple models
            update_progress(9, 10, "Creating ensemble...")
            if len(self.trained_models) > 1:
                ensemble_model, ensemble_score, ensemble_name = self.create_ensemble_models(X, y)
                if ensemble_model and ensemble_score > self.best_score:
                    self.best_model = ensemble_model
                    self.best_score = ensemble_score
                    self.best_name = ensemble_name
            
            # Step 10: Compute final metrics
            update_progress(10, 10, "Computing metrics...")
            if self.best_model is None:
                logger.error("No models were trained successfully")
                return None, None
            
            metrics = self.compute_metrics(self.best_model, X, y)
            metrics["model_name"] = self.best_name
            
            # Attach scaler to model
            self.best_model.scaler = scaler
            
            # Cleanup
            resource_optimizer.aggressive_cleanup()
            
            logger.info(f"🏆 Best model: {self.best_name} (accuracy={self.best_score:.4f})")
            logger.info(f"⏱️ Training completed in {time.time() - start_time:.1f}s")
            
            return self.best_model, metrics
          def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, float]:
            """Train ensemble model using existing trained models or create new ones"""
            # If we don't have any trained models, train some first
            if len(self.trained_models) < 2:
                logger.info("🔄 Training base models for ensemble...")
                
                # Train a few key models
                models_to_train = [
                    ("random_forest", self.train_random_forest),
                    ("xgboost", self.train_xgboost),
                    ("lightgbm", self.train_lightgbm)
                ]
                
                for name, train_func in models_to_train:
                    try:
                        model, score = train_func(X, y)
                        if model is not None and score > 0:
                            self.trained_models[name] = (model, score)
                            logger.info(f"✅ {name} trained for ensemble with score={score:.4f}")
                    except Exception as e:
                        logger.warning(f"❌ {name} training failed for ensemble: {e}")
            
            # Create ensemble
            ensemble_model, ensemble_score, ensemble_name = self.create_ensemble_models(X, y)
            
            if ensemble_model is None:
                logger.warning("Ensemble creation failed, falling back to best individual model")
                if self.trained_models:
                    best_name, (best_model, best_score) = max(self.trained_models.items(), key=lambda x: x[1][1])
                    return best_model, best_score                else:
                    # Last resort: train a single random forest
                    return self.train_random_forest(X, y)
                return ensemble_model, ensemble_score
    
    class FeatureImportance:
        """
        Module theo dõi và đánh giá độ quan trọng của feature cho AI.
        - compute_importance: lấy importance từ model tree‑based hoặc linear
        - ranking: xuất DataFrame sắp xếp feature theo importance
        - select_top_k: chọn top k feature
        - select_by_threshold: chọn feature trên ngưỡng
        """

        def __init__(self, model):
            """
            model: estimator của sklearn (fitted hoặc unfitted).
            Nếu model chưa fit, bạn cần fit trước khi gọi compute_importance.
            """
            self.model = model

        def compute_importance(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]=None) -> pd.Series:
            """
            Trả về Series importance, index = X.columns, sorted giảm dần.
            Nếu model chưa fit và y != None, sẽ tự fit model.
            """
            # Fit model nếu cần
            if not hasattr(self.model, 'feature_importances_') and not hasattr(self.model, 'coef_'):
                if y is None:
                    raise ValueError("Model chưa fit và không có y để fit.")
                self.model = clone(self.model)
                self.model.fit(X, y)

            # Tree-based
            if hasattr(self.model, 'feature_importances_'):
                imp = self.model.feature_importances_
            # Linear model
            elif hasattr(self.model, 'coef_'):
                coef = self.model.coef_
                # với multiclass/regression, trung bình hóa các lớp
                if coef.ndim > 1:
                    imp = np.mean(np.abs(coef), axis=0)
                else:
                    imp = np.abs(coef)
            else:
                raise ValueError("Model không có attribute feature_importances_ hoặc coef_")

            return pd.Series(imp, index=X.columns).sort_values(ascending=False)

        def ranking(self,
                    X: pd.DataFrame,
                    y: Union[pd.Series, np.ndarray]=None,
                    top_n: int = None
                ) -> pd.DataFrame:
            """
            Trả DataFrame 2 cột ['feature','importance'] sắp giảm dần.
            Nếu top_n != None, chỉ lấy top_n dòng.
            """
            imp = self.compute_importance(X, y)
            df = imp.reset_index()
            df.columns = ['feature','importance']
            if top_n:
                return df.head(top_n)
            return df

        def select_top_k(self, X: pd.DataFrame, k: int, 
                        y: Union[pd.Series, np.ndarray]=None
                        ) -> pd.DataFrame:
            """
            Lấy DataFrame chỉ gồm k feature quan trọng nhất.
            """
            imp = self.compute_importance(X, y)
            top_feats = imp.nlargest(k).index.tolist()
            return X[top_feats]

        def select_by_threshold(self,
                                X: pd.DataFrame,
                                threshold: float,
                                y: Union[pd.Series, np.ndarray]=None
                            ) -> pd.DataFrame:
            """
            Lấy feature có importance >= threshold.
            """
            imp = self.compute_importance(X, y)
            feats = imp[imp >= threshold].index.tolist()
            return X[feats]

        def track_over_time(self,
                            X_list: List[pd.DataFrame],
                            y_list: List[Union[pd.Series, np.ndarray]]
                        ) -> pd.DataFrame:
            """
            Cho lista các tập X,y theo thời gian (ví dụ retrain mỗi tuần),
            tính importance cho mỗi lần và trả DataFrame:
            index = feature, cột = lần thứ i, giá trị = importance.
            """
            records = []        
            for X, y in zip(X_list, y_list):            imp = self.compute_importance(X, y)
                records.append(imp)
            df = pd.concat(records, axis=1)
            df.columns = [f'round_{i+1}' for i in range(len(records))]
            return df
    
    class SHAPExplainer:
        """
        Giải thích mô hình đã train bằng SHAP.
        Trích xuất top N feature quan trọng nhất theo mean(abs SHAP value).
        """

        def __init__(self, model, X_sample: pd.DataFrame, explainer_type: str = 'auto'):
            """
            model: mô hình đã fit (tree-based, linear...)
            X_sample: sample features để giải thích (DataFrame)
            explainer_type: 'auto' (default), 'tree', 'linear', 'kernel'
            """
            self.model = model
            self.X = X_sample
            self.explainer_type = explainer_type
            self.explainer = None
            self.shap_values = None

        def explain(self):
            """
            Tính shap_values tương ứng model + sample
            """
            if self.explainer_type == 'auto':
                try:
                    self.explainer = shap.Explainer(self.model, self.X)
                except Exception:
                    self.explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, self.X)
            elif self.explainer_type == 'kernel':
                self.explainer = shap.KernelExplainer(self.model.predict_proba, self.X)
            else:
                raise ValueError("Unsupported explainer type")

            self.shap_values = self.explainer(self.X)

        def top_features(self, top_n: int = 10) -> pd.Series:
            """
            Trả về Series feature_name → mean(abs(shap_value)) sắp xếp giảm dần.
            """
            if self.shap_values is None:
                self.explain()

            # Trường hợp binary-class: lấy lớp 1
            if isinstance(self.shap_values.values, list):
                shap_arr = self.shap_values.values[1]
            else:
                shap_arr = self.shap_values.values

            mean_abs = abs(shap_arr).mean(axis=0)
            return pd.Series(mean_abs, index=self.X.columns).sort_values(ascending=False).head(top_n)        
    # 5. shap module - Tính toán SHAP values
    def compute_shap_values(model, X):
        """
        Tính SHAP values cho mô hình và tập X trả về numpy array hoặc None.
        """
        try:
        except ImportError:
            os.system("pip install shap")
            try:
            except ImportError:
                print("Không thể cài đặt shap.")
                return None

        try:
            explainer = shap.Explainer(model, X)
            shap_res = explainer(X)
            return shap_res.values
        except Exception as e:
            print("Lỗi khi tính SHAP values:", e)
            return None# Note: train_all_models function moved to Training class
class Prediction:
    """
    Prediction Class - Forecasting with TP/SL entry using meta AI
    
    Handles all prediction operations including:
    - Model loading and prediction execution
    - Meta AI optimization and recommendations
    - Take Profit/Stop Loss calculations
    - Long-term forecasting
    - Performance analysis and backtesting
    """
    
    def __init__(self):
        """Initialize prediction with global instances"""
        self.model_predictor = None  # Will be initialized when needed
        self.meta_ai = None  # Will be initialized when needed
        self.logger = get_logger('prediction')
        self.model_dir = MODEL_DIR
          # ================== MAIN PREDICTION METHODS ==================
    
    def run_prediction(self, symbol: str, timeframe: str = "15m") -> Dict[str, Any]:
        """
        CONSOLIDATED prediction method - replaces run_prediction_thread
        Executes complete prediction pipeline with TP/SL calculations
        """
        try:
            logger.info(f"Starting prediction for {symbol}@{timeframe}")
            
            # Load latest data
            data_collection = DataCollection()
            df = data_collection.update_data(symbol, timeframe)
            
            if df is None or df.empty:
                return {"error": "No data available"}
            
            # Prepare features
            X, _ = prepare_features(df, symbol=symbol, training=False)
            
            # Load model
            model_path = f"ai_models/trained_models/{symbol.upper()}_best.pkl"
            if not os.path.exists(model_path):
                return {"error": f"No trained model found for {symbol}"}
            
            predictor = self.ModelPredictor(model_path)
            predictor.load_model()
            
            # Make prediction
            latest_features = X.iloc[[-1]]
            prob = predictor.predict_proba(latest_features)[0]
            prediction = predictor.predict(latest_features)[0]
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Calculate TP/SL using MetaAI
            meta_ai = self.MetaAI(symbol, df)
            tp_sl = meta_ai.suggest_tp_sl(current_price, prob[1], df)
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "prediction": int(prediction),
                "probability": float(prob[1]),
                "current_price": float(current_price),
                "tp_sl": tp_sl,
                "model_name": predictor.model_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}
    class MetaAI:
        """
        Module Meta AI:
        - Chọn khung thời gian và mô hình tốt nhất dựa trên metrics từ huấn luyện.
        - Resample giá về khung thời gian tốt nhất.
        - Tạo feature từ df_price đã resample, load mô hình & threshold, và dự đoán Long/Short.
        """
        def __init__(self, symbol: str, metrics: pd.DataFrame, project_dir: str = None):
            self.symbol = symbol; self.metrics = metrics.copy()
            self.project_dir = project_dir or os.getcwd()
        

        def recommend(self, metric: str = 'f1') -> Dict[str, str]:
            """
            Đề xuất top khung thời gian và mô hình tốt nhất theo metric (vd. 'accuracy'/'f1').
            """
            m = self.metrics
            if m.index.name == 'timeframe':
                m = m.reset_index()
            ao = AutoOptimizer(self.metrics)
            return ao.recommend(metric)

        @staticmethod
        def resample_price(df_price: pd.DataFrame, timeframe: str) -> pd.DataFrame:
            """
            Resample df_price về khung timeframe: 
            open = first, high = max, low = min, close = last, volume = sum.
            """
            df_price['timestamp'] = pd.to_datetime(df_price['timestamp'], utc=True)
            df = df_price.set_index('timestamp')
            agg = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            df_res = df.resample(timeframe).agg(agg).dropna().reset_index()
            return df_res

        def predict(self, df_price: pd.DataFrame, metric: str = 'f1') -> pd.Series:
            """
            Dự đoán Long/Short trên df_price:
            1) Chọn best_timeframe & best_model dựa trên metrics.
            2) Resample giá về khung best_timeframe.
            3) Tạo feature (sử dụng FeatureEngineer).
            4) Load model & threshold từ thư mục models.
            5) Trả về Series dự đoán (0/1) với index là timestamp khung best_timeframe.
            """
            # 1) Chọn best time frame & model
            info = self.recommend(metric)
            tf = info['best_timeframe']
            model_name = info['best_models'][tf]

            # 2) Resample giá
            df_tf = self.resample_price(df_price, tf)

            # 3) Tạo feature
            X = FeatureEngineer(df_tf)

            # 4) Load model và threshold
            model_path = os.path.join(
                self.project_dir, 'models',
                f"{self.symbol}_{tf}",
                f"{model_name}.pkl"
            )
            mp = ModelPredictor(model_path)
            mp.load_model()

            # 5) Dự đoán
            preds = mp.predict(X)
            return preds
    def backtest_strategy(signals, prices):
        """
        signals: list tín hiệu (1/0/-1), prices: list giá (len = len(signals)+1).
        Trả về dict: total_profit, total_trades, win_rate_percent, average_profit_per_trade, profit_history.
        """
        n = len(signals)
        if len(prices) < n + 1:
            print("Dữ liệu giá không đủ cho backtest.")
            return None

        total_profit = 0.0
        wins = losses = 0
        profit_history = []

        for i, sig in enumerate(signals):
            profit = sig * (prices[i+1] - prices[i])
            total_profit += profit
            profit_history.append(total_profit)
            if profit > 0:
                wins += 1
            elif profit < 0:
                losses += 1

        total_trades = n
        win_rate = (wins / total_trades * 100) if total_trades else 0.0
        avg_profit = total_profit / total_trades if total_trades else 0.0

        return {
            "total_profit": total_profit,
            "total_trades": total_trades,
            "win_rate_percent": win_rate,
            "average_profit_per_trade": avg_profit,
            "profit_history": profit_history
        }
    class RealtimeBacktester:
        """
        Backtest mô hình đã huấn luyện trên dữ liệu thực tế.
        - Load model + threshold từ file .pkl
        - Tạo feature từ DataFrame giá
        - Dự đoán Long/Short
        - Tính accuracy, precision, recall, f1
        """

        def __init__(self, model_path: str = None, df_price: pd.DataFrame = None, feature_func=None):
            # Nếu không truyền model_path, không làm gì (để test không lỗi)
            self.model_path = model_path
            self.df_price  = df_price.copy() if df_price is not None else pd.DataFrame()
            self.feature_func = feature_func
            self.model = None
            self.threshold = None
            self.df_feat = None
            self.results = {}

        def load_model(self):
            """Load model và threshold từ file .pkl"""
            with gzip.open(self.model_path, 'rb') as f:
                bundle = pickle.load(f)
                self.model = bundle['model']
                self.threshold = bundle.get('threshold', 0.5)

        def prepare_features(self):
            """Gọi hàm feature_func để tạo feature từ df_price"""
            self.df_feat = self.feature_func(self.df_price)
            return self.df_feat

        def simulate(self):
            """
            Thực thi backtest:
            - Dự đoán Long/Short với threshold
            - Tính label thực từ close shift(-1)
            - Tính các metric và trả về kết quả
            """
            if self.model is None or self.df_feat is None:
                raise RuntimeError("Call load_model() and prepare_features() first")

            # predict probabilities & apply threshold
            y_proba = self.model.predict_proba(self.df_feat)[:, 1]
            y_pred = (y_proba >= self.threshold).astype(int)

            df = self.df_price.iloc[-len(y_pred):].copy()
            df['prediction'] = y_pred
            df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
            df = df.dropna()

            y_true = df['label'].astype(int).values
            y_pred_final = df['prediction'].astype(int).values

            self.results = {
                'df': df,
                'accuracy': accuracy_score(y_true, y_pred_final),
                'precision': precision_score(y_true, y_pred_final, zero_division=0),
                'recall': recall_score(y_true, y_pred_final, zero_division=0),
                'f1': f1_score(y_true, y_pred_final)
            }
            return self.results
    # File: AI_Crypto_Project/backtesting/walk_forward.py
    class WalkForwardBacktester:
        """
        Walk-forward backtest:
        - Train trên window_size mẫu
        - Test trên step_size mẫu kế tiếp
        - Lưu kết quả accuracy, f1 cho mỗi bước
        """

        def __init__(self, window_size: int = 500, step_size: int = 100):
            """
            :param window_size: số mẫu dùng để huấn luyện mỗi bước
            :param step_size: số mẫu dùng để kiểm thử mỗi bước
            """
            self.window_size = window_size
            self.step_size = step_size

        def run(self, model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
            """
            Chạy walk-forward với error handling cải thiện:
            :param model: mô hình sklearn (chưa fit)
            :param X: DataFrame feature toàn bộ chuỗi
            :param y: Series nhãn (0/1) tương ứng
            :return: DataFrame với các cột
                    ['train_start','train_end','test_end','accuracy','f1']
            """
            
            logger = get_logger('walk_forward')
            
            scores = []
            n = len(X)
            
            # Kiểm tra dữ liệu đầu vào
            if n < self.window_size + self.step_size:
                logger.warning(f"Insufficient data for backtest: {n} < {self.window_size + self.step_size}")
                return pd.DataFrame(columns=['train_start','train_end','test_end','accuracy','f1'])
            
            # Kiểm tra unique classes
            unique_classes = y.nunique()
            if unique_classes < 2:
                logger.warning(f"Insufficient classes for backtest: {unique_classes}")
                return pd.DataFrame(columns=['train_start','train_end','test_end','accuracy','f1'])
            
            # lặp qua các cửa sổ
            for start in range(0, n - self.window_size - self.step_size + 1, self.step_size):
                try:
                    end_train = start + self.window_size
                    end_test = end_train + self.step_size

                    X_train = X.iloc[start:end_train]
                    y_train = y.iloc[start:end_train]
                    X_test  = X.iloc[end_train:end_test]
                    y_test  = y.iloc[end_train:end_test]
                    
                    # Kiểm tra classes trong train set
                    if y_train.nunique() < 2:
                        logger.warning(f"Skipping window {start}: insufficient classes in train set")
                        continue
                    
                    # Kiểm tra classes trong test set
                    if y_test.nunique() < 1:
                        logger.warning(f"Skipping window {start}: no data in test set")
                        continue

                    # Clone model để tránh conflict
                    model_clone = clone(model)
                    model_clone.fit(X_train, y_train)
                    y_pred = model_clone.predict(X_test)

                    # Tính metrics với zero_division handling
                    acc = accuracy_score(y_test, y_pred)
                    
                    # F1 score với handling cho edge cases
                    try:
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    except Exception as e:
                        logger.warning(f"F1 calculation failed: {e}, using 0")
                        f1 = 0.0

                    scores.append({
                        'train_start': start,
                        'train_end': end_train,
                        'test_end': end_test,
                        'accuracy': acc,
                        'f1': f1
                    })
                    
                except Exception as e:
                    logger.warning(f"Backtest window {start} failed: {e}")
                    continue

            if not scores:
                logger.warning("No successful backtest windows")
                return pd.DataFrame(columns=['train_start','train_end','test_end','accuracy','f1'])
                
            return pd.DataFrame(scores)
    def generate_trade_signal(prob_up: float, current_price: float, future_max: float, future_min: float) -> dict:
        """
        Decide trade signal (entry, TP, SL, confidence) based on probability of upward movement and predicted range.
        Returns a dictionary with keys: direction, entry, take_profit, stop_loss, confidence.
        """
        if prob_up >= 0.5:
            direction = "BUY"
            confidence = prob_up
            entry_price = current_price
            take_profit = float(future_max)
            stop_loss = float(future_min)
        else:
            direction = "SELL"
            confidence = 1 - prob_up
            entry_price = current_price
            # For SELL signal, take_profit is a lower price (future_min), stop_loss is a higher price (future_max)
            take_profit = float(future_min)
            stop_loss = float(future_max)
        conf_percent = round(confidence * 100, 2)
        return {
            "direction": direction,
            "entry": float(entry_price),
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "confidence": conf_percent
        }
    def Threshold:
        """
        Module tối ưu ngưỡng (decision threshold) cho mô hình phân loại nhị phân.
        Cho phép tìm threshold tối ưu trên tập validation để cân bằng
        accuracy, precision, recall, F1 hoặc AUC.
        """

        def __init__(self, metric: str = 'f1'):
            """
            metric: one of 'accuracy', 'precision', 'recall', 'f1', 'auc'
            """
            allowed = {'accuracy', 'precision', 'recall', 'f1', 'auc'}
            if metric not in allowed:
                raise ValueError(f"Unknown metric '{metric}', choose one of {allowed}")
            self.metric = metric
            # mapping metric name to function
            self._metric_fn = {
                'accuracy': accuracy_score,
                'precision': precision_score,
                'recall': recall_score,
                'f1': f1_score
            }

        def optimize(self,
                    y_true: np.ndarray,
                    y_proba: np.ndarray,
                    thresholds: np.ndarray = None
                    ) -> dict:
            """
            Tìm threshold tốt nhất.
            - y_true: array of true binary labels (0/1)
            - y_proba: array of predicted positive-class probabilities
            - thresholds: array of thresholds để thử (nếu None, dùng 101 giá trị linspace 0→1)
            Trả về dict:
            {
                'best_threshold': float,
                'best_score': float,
                'scores': DataFrame with columns ['threshold','score']
            }
            """
            if thresholds is None:
                thresholds = np.linspace(0.0, 1.0, 101)

            records = []
            for t in thresholds:
                y_pred = (y_proba >= t).astype(int)
                if self.metric == 'auc':
                    # AUC không phụ thuộc threshold, tính 1 lần
                    score = roc_auc_score(y_true, y_proba)
                else:
                    # với precision/recall/f1, bỏ trường hợp không có positive predictions
                    if self.metric in {'precision', 'recall'} and y_pred.sum() == 0:
                        score = 0.0
                    else:
                        score = self._metric_fn[self.metric](y_true, y_pred)
                records.append((t, score))

            df_scores = pd.DataFrame(records, columns=['threshold','score'])
            # chọn the first threshold với max score
            best_idx = df_scores['score'].idxmax()
            best_threshold = df_scores.loc[best_idx, 'threshold']
            best_score = df_scores.loc[best_idx, 'score']

            return {
                'best_threshold': float(best_threshold),
                'best_score': float(best_score),
                'scores': df_scores
            }

        def apply_threshold(self,
                            y_proba: np.ndarray,
                            threshold: float
                        ) -> np.ndarray:
            """
            Áp ngưỡng mới để chuyển xác suất thành nhãn 0/1.
            """
            return (y_proba >= threshold).astype(int)
        
        def compute_threshold(self, model, X, y):
            """Tìm threshold tốt nhất dựa trên metric đã chọn."""
            # Kiểm tra số lượng class
            n_classes = len(np.unique(y))
            
            if n_classes == 2:
                # Binary classification: tìm threshold tối ưu
                y_proba = model.predict_proba(X)[:, 1]
                best_thr = 0.5
                best_score = 0.0
                # Duyệt ngưỡng từ 0.00 đến 1.00
                for thr in np.linspace(0, 1, 101):
                    y_pred = (y_proba >= thr).astype(int)
                    # Sử dụng average='binary' cho binary classification
                    if self.metric in ['precision', 'recall', 'f1']:
                        score = self._metric_fn[self.metric](y, y_pred, average='binary', zero_division=0)
                    else:
                        score = self._metric_fn[self.metric](y, y_pred)
                    if score > best_score:
                        best_score = score
                        best_thr = thr
                return best_thr
            else:
                # Multiclass: không có threshold, trả về 0.5 mặc định
                return 0.5
    # File: AI_Crypto_Project/ai_optimizer/hyperparameter_optimizer.py


    def Hyperparameter:
        """
        Dùng RandomizedSearchCV với TimeSeriesSplit để tìm tham số tốt nhất
        cho từng model, tối ưu theo một metric nhất định.
        """

        def __init__(self,
                    model,
                    param_distributions: dict,
                    n_iter: int = None,
                    cv_splits: int = 5,
                    metric: str = 'f1',
                    random_state: int = 42,
                    n_jobs: int = -1,
                    verbose: int = 1):
            """
            model: estimator sklearn
            param_distributions: dict cho RandomizedSearchCV
            n_iter: số lần thử ngẫu nhiên
            cv_splits: số splits cho TimeSeriesSplit
            metric: 'accuracy','precision','recall','f1'
            """
            allowed = {'accuracy','precision','recall','f1'}
            if metric not in allowed:
                raise ValueError(f"Unknown metric '{metric}', choose one of {allowed}")

            self.model = model
            self.param_distributions = param_distributions
            self.n_iter = n_iter
                    # --- nếu n_iter lớn hơn tổng số tổ hợp thì hạ xuống ---
            total_combinations = 1
            for vals in self.param_distributions.values():
                total_combinations *= len(vals)
            if self.n_iter > total_combinations:
                self.n_iter = total_combinations
            self.cv = TimeSeriesSplit(n_splits=cv_splits)
            self.metric = metric
            self.random_state = random_state
            self.n_jobs = n_jobs
            self.verbose = verbose

            # map metric to scorer
            self.scorer = make_scorer({
                'accuracy': accuracy_score,
                'precision': precision_score,
                'recall': recall_score,
                'f1': f1_score
            }[metric], zero_division=0)

            self.search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_distributions,
                n_iter=self.n_iter,
                scoring=self.scorer,
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )

        def optimize(self, X, y):
            """
            Thực thi RandomizedSearchCV trên X, y.
            Trả về tuple: (best_estimator, best_params, best_score).
            """
            self.search.fit(X, y)
            return (
                self.search.best_estimator_,
                self.search.best_params_,
                self.search.best_score_
            )
    # File: AI_Crypto_Project/ai_optimizer/drift_detector.py


    def DriftDetector:
        """
        Phát hiện sự thay đổi phân phối (drift) giữa hai tập dữ liệu.
        Sử dụng Kolmogorov–Smirnov Test (KS Test) cho các feature numeric.
        """

        def __init__(self, alpha: float = 0.05):
            """
            alpha: mức ý nghĩa để kết luận drift (mặc định 0.05)
            """
            self.alpha = alpha
            self.drifted_features = []

        def compare(self, df_old: pd.DataFrame, df_new: pd.DataFrame) -> dict:
            """
            So sánh hai DataFrame cùng schema.
            Trả về:
            {
                'drifted_features': list[str],
                'total_checked': int,
                'drift_ratio': float,
                'drift_detected': bool
            }
            """
            common_cols = df_old.columns.intersection(df_new.columns)
            drifted = []

            for col in common_cols:
                if not np.issubdtype(df_old[col].dtype, np.number):
                    continue
                x1 = df_old[col].dropna()
                x2 = df_new[col].dropna()
                if len(x1) < 30 or len(x2) < 30:
                    continue
                stat, p_value = ks_2samp(x1, x2)
                if p_value < self.alpha:
                    drifted.append(col)

            drift_ratio = len(drifted) / max(len(common_cols), 1)

            return {
                'drifted_features': drifted,
                'total_checked': len(common_cols),
                'drift_ratio': drift_ratio,
                'drift_detected': len(drifted) > 0
            }
    # File: AI_Crypto_Project/ai_analysis_modules/cross_asset_correlation.py
    # Nhiệm vụ: Tính tương quan giữa bitcoin với nhiều tài sản khác


    def calculate_correlation(main_df: pd.DataFrame, other_assets: dict) -> dict:
        """
        - main_df: DataFrame của asset chính, có 'timestamp','close'
        - other_assets: dict[name]=DataFrame tương tự
        - Trả dict[name]=correlation(close_pct_change)
        """
        if main_df is None or main_df.empty:
            return {}
        main_df['timestamp'] = pd.to_datetime(main_df['timestamp'], utc=True)
        main_series = main_df.set_index('timestamp')['close'].pct_change().dropna()
        corr = {}
        for name, df in other_assets.items():
            if df is None or df.empty:
                continue
            other = df.set_index('timestamp')['close'].pct_change().dropna()
            corr[name] = main_series.corr(other)
            # ================== MODEL PREDICTOR CLASS ==================
    
    class ModelPredictor:
        """Model predictor for loading and using trained models - consolidated implementation"""
        
        def __init__(self, model_path: str):
            self.model_path = model_path
            self.model = None
            self.threshold = 0.5
            self.model_name = "unknown"
            self.scaler = None
            
        def load_model(self):
            """Load model bundle from file"""
            try:
                with gzip.open(self.model_path, 'rb') as f:
                    bundle = pickle.load(f)
                    self.model = bundle['model']
                    self.threshold = bundle.get('threshold', 0.5)
                    self.model_name = bundle.get('model_name', 'unknown')
                    self.scaler = getattr(bundle['model'], 'scaler', None)
                    logger.info(f"✅ Loaded model: {self.model_name} with threshold: {self.threshold:.3f}")
            except Exception as e:
                logger.error(f"❌ Failed to load model from {self.model_path}: {e}")
                raise
                
        def predict(self, X):
            """Make predictions on input data"""
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")
            
            # Apply scaling if available
            if self.scaler is not None:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X_scaled = X.copy()
                X_scaled[numeric_cols] = self.scaler.transform(X[numeric_cols])
                return self.model.predict(X_scaled)
            
            return self.model.predict(X)
        
        def predict_proba(self, X):
            """Make probability predictions on input data"""
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")
            
            # Apply scaling if available
            if self.scaler is not None:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X_scaled = X.copy()
                X_scaled[numeric_cols] = self.scaler.transform(X[numeric_cols])
                return self.model.predict_proba(X_scaled)
            
            return self.model.predict_proba(X)
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")
                
            if not hasattr(self.model, 'predict_proba'):
                # For models without predict_proba, return binary predictions as probabilities
                predictions = self.predict(X)
                proba_0 = 1 - predictions
                proba_1 = predictions
                return pd.DataFrame({0: proba_0, 1: proba_1})
            
            # Apply scaling if available
            if self.scaler is not None:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X_scaled = X.copy()
                X_scaled[numeric_cols] = self.scaler.transform(X[numeric_cols])
                proba = self.model.predict_proba(X_scaled)
            else:
                proba = self.model.predict_proba(X)
            
            return pd.DataFrame({0: proba[:, 0], 1: proba[:, 1]})
    def suggest_tp_sl(df, signal: str) -> dict:
        """
        Gợi ý mức Take Profit (TP) và Stop Loss (SL) động dựa trên ATR và biến động hiện tại:
        - Tính ATR chu kỳ 14 cho dữ liệu giá (đo lường biến động gần đây).
        - So sánh ATR hiện tại với ATR trung bình (biến động lịch sử) để điều chỉnh TP/SL:
        + Nếu thị trường biến động mạnh (ATR cao hơn trung bình) thì rút ngắn TP, nới SL hơn một chút.
        + Nếu thị trường đang bình lặng (ATR thấp hơn trung bình) thì tăng TP, giảm SL tương ứng.
        - Nếu `signal` = "Long": TP = giá hiện tại + hệ số_tp * ATR, SL = giá hiện tại - hệ số_sl * ATR.
        Nếu `signal` = "Short": TP = giá hiện tại - hệ số_tp * ATR, SL = giá hiện tại + hệ số_sl * ATR.
        - Trả về dictionary {'TP': ..., 'SL': ...} chứa mức TP/SL đề xuất.
        """
        if df is None or df.empty or signal not in ('Long', 'Short'):
            return {}
        # Tính ATR14
        atr_series = FeatureEngineer(df['high'], df['low'], df['close'], period=14)
        if len(atr_series) == 0:
            return {}
        atr_current = float(atr_series.iloc[-1])
        price = float(df['close'].iloc[-1])
        if atr_current <= 0 or price <= 0:
            return {}

        # Mặc định hệ số TP/SL
        tp_factor = 1.5
        sl_factor = 1.0

        # Xét biến động lịch sử (ATR trung bình)
        atr_avg = float(np.nanmean(atr_series))
        if atr_avg > 0:
            vol_ratio = atr_current / atr_avg
        else:
            vol_ratio = 1.0

        # Điều chỉnh hệ số dựa trên mức độ biến động so với trung bình
        if vol_ratio > 1.2:
            # Biến động hiện tại cao hơn ~20% so với trung bình -> thị trường biến động mạnh
            tp_factor *= 0.9   # giảm nhẹ mục tiêu TP để chốt lời sớm hơn trong thị trường nhiều biến động
            sl_factor *= 1.1   # nới SL thêm chút để tránh nhiễu do biến động lớn
        elif vol_ratio < 0.8:
            # Biến động hiện tại thấp hơn ~20% so với trung bình -> thị trường ít biến động
            tp_factor *= 1.1   # tăng nhẹ mục tiêu TP vì thị trường di chuyển chậm, cần mục tiêu xa hơn một chút
            sl_factor *= 0.9   # giảm SL để hạn chế rủi ro trong thị trường đi ngang

        # Tính toán TP và SL dựa trên tín hiệu Long/Short
        if signal == 'Long':
            tp = price + tp_factor * atr_current
            sl = price - sl_factor * atr_current
        else:  # Short
            tp = price - tp_factor * atr_current
            sl = price + sl_factor * atr_current

        return {'TP': tp, 'SL': sl}
    
class SystemControl:
    """
    System Control Class - CPU, GPU, RAM monitoring
    
    Handles all system monitoring and control operations including:
    - Resource monitoring and optimization
    - GPU management and allocation
    - Memory and CPU optimization
    - Background monitoring systems
    - Performance alerts and management
    """
    
    def __init__(self):
        """Initialize system control with global instances"""
        self.resource_optimizer = global_resource_optimizer  # Reference to global instance
        self.gpu_manager = gpu_manager  # Reference to global instance
        self.logger = get_logger('system_control')
        
    # Resource Management Methods (will be moved here)
    # - ResourceOptimizer class methods
    # - get_comprehensive_system_stats()
    # - predict_resource_usage()
    
    # GPU Management Methods (will be moved here)
    # - GPUManager class methods
    # - monitor_gpu_health()
    # - adaptive_resource_management()
    
    # Monitoring Methods (will be moved here)
    # - Background monitoring and alerting systems
    # - Memory and CPU optimization functions

    def SystemResourceMetrics:
        """Class để lưu trữ metrics tài nguyên hệ thống"""
        timestamp: datetime
        cpu_percent: float
        memory_percent: float
        memory_available_gb: float
        gpu_metrics: List[Dict[str, Any]]
        disk_usage_percent: float
        network_io: Dict[str, int]
        temperature: Optional[float] = None

    # Try to import GPUtil for GPU monitoring
    try:
        GPU_AVAILABLE = True
    except ImportError:
        GPUtil = None
        GPU_AVAILABLE = False

    # Try to import pynvml for NVIDIA GPU monitoring
    try:
        pynvml.nvmlInit()
        NVML_AVAILABLE = True
    except (ImportError, Exception):
        pynvml = None
        NVML_AVAILABLE = False

    def ResourceOptimizer:
        """
        Giám sát và tối ưu tài nguyên:
        - CPU/RAM monitoring & worker adjustment
        - GPU monitoring & selection/throttling (nếu có GPU)
        - System-wide resource management
        """
        # Singleton instance
        _instance = None
        _initialized = False
        _monitor_thread = None

        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = super(ResourceOptimizer, cls).__new__(cls)
            return cls._instance

        def __init__(self,
                    max_workers: int = MAX_WORKERS,
                    target_cpu: int = MAX_CPU_PERCENT,
                    memory_threshold: float = MAX_RAM_UTIL,
                    max_gpu_load: float = MAX_GPU_LOAD,
                    min_free_gpu_mem: float = MIN_FREE_GPU_MEM):
            
            # Không khởi tạo lại nếu đã được khởi tạo
            if ResourceOptimizer._initialized:
                return
                
            # CPU/RAM params
            self.max_workers = max_workers
            self.target_cpu = target_cpu
            self.memory_threshold = memory_threshold * 100  # Convert to percentage

            # GPU params
            self.max_gpu_load = max_gpu_load
            self.min_free_gpu_mem = min_free_gpu_mem
            
            # System monitoring
            self._resource_history = []
            self._max_history_size = 100
            self._monitoring_lock = threading.Lock()
            
            # Performance metrics
            self._performance_metrics = {
                'cpu_utilization': [],
                'memory_utilization': [],
                'gpu_utilization': [],
                'throttle_events': 0,
                'optimization_events': 0
            }
            
            # Kiểm tra và khởi tạo GPU
            self.has_gpu = self._check_gpu_available()
            self.gpu_count = 0
            self.gpu_names = []
            
            if self.has_gpu:
                self._init_gpu_settings()
                self._detect_gpu_info()
                
            ResourceOptimizer._initialized = True
            
            # Use unified logging instead of duplicate basicConfig
            
            self.logger = get_logger('resource_optimizer')
            
            # Thread pool cho background tasks
            self.thread_pool = ThreadPoolExecutor(max_workers=2)
            
            self.logger.info(f"ResourceOptimizer initialized - GPU: {self.has_gpu}, Count: {self.gpu_count}")

        def _check_gpu_available(self) -> bool:
            """Kiểm tra GPU có khả dụng không"""
            try:
                return torch.cuda.is_available()
            except ImportError:
                return False

        def _detect_gpu_info(self):
            """Phát hiện thông tin GPU trong hệ thống"""
            try:
                if torch.cuda.is_available():
                    self.gpu_count = torch.cuda.device_count()
                    self.gpu_names = [torch.cuda.get_device_name(i) for i in range(self.gpu_count)]
                    self.logger.info(f"Detected {self.gpu_count} GPU(s): {self.gpu_names}")
                else:
                    self.gpu_count = 0
                    self.gpu_names = []
            except Exception as e:
                self.logger.warning(f"Error detecting GPU info: {e}")
                self.gpu_count = 0
                self.gpu_names = []

        def _init_gpu_settings(self):
            """Khởi tạo cài đặt GPU nếu có"""
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.cuda.empty_cache()
                
                # Set memory fraction cho mỗi GPU
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.set_per_process_memory_fraction(0.8)
                            
            except Exception as e:
                self.logger.warning(f"Error initializing GPU settings: {e}")
        # --- System Resource Management ---
        
        def get_comprehensive_system_stats(self) -> SystemResourceMetrics:
            """Thu thập metrics toàn diện về tài nguyên hệ thống"""
            try:
                # CPU và Memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_available_gb = memory.available / (1024**3)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_usage_percent = disk.percent
                
                # Network I/O
                net_io = psutil.net_io_counters()
                network_io = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
                
                # GPU metrics
                gpu_metrics = []
                if self.has_gpu:
                    gpu_stats = self.monitor_gpu()
                    if gpu_stats:
                        gpu_metrics = gpu_stats
                        
                    # Thêm thông tin từ PyTorch CUDA
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            mem_info = torch.cuda.memory_stats(i)
                            gpu_metrics.append({
                                'device_id': i,
                                'name': torch.cuda.get_device_name(i),
                                'memory_allocated': mem_info.get('allocated_bytes.all.current', 0) / (1024**3),
                                'memory_cached': mem_info.get('reserved_bytes.all.current', 0) / (1024**3),
                                'memory_total': torch.cuda.get_device_properties(i).total_memory / (1024**3)
                            })
                
                # Temperature (nếu có)
                temperature = None
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for name, entries in temps.items():
                            if entries:
                                temperature = entries[0].current
                                break
                except (AttributeError, OSError):
                    pass
                
                metrics = SystemResourceMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_available_gb=memory_available_gb,
                    gpu_metrics=gpu_metrics,
                    disk_usage_percent=disk_usage_percent,
                    network_io=network_io,
                    temperature=temperature
                )
                
                # Lưu vào history
                with self._monitoring_lock:
                    self._resource_history.append(metrics)
                    if len(self._resource_history) > self._max_history_size:
                        self._resource_history.pop(0)
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"Error collecting system stats: {e}")
                return None
        
        def predict_resource_usage(self) -> Dict[str, float]:
            """Dự đoán xu hướng sử dụng tài nguyên dựa trên lịch sử"""
            with self._monitoring_lock:
                if len(self._resource_history) < 5:
                    return {}
                
                # Lấy 10 điểm dữ liệu gần nhất
                recent_history = self._resource_history[-10:]
                
                cpu_trend = np.polyfit(range(len(recent_history)), 
                                    [h.cpu_percent for h in recent_history], 1)[0]
                memory_trend = np.polyfit(range(len(recent_history)), 
                                        [h.memory_percent for h in recent_history], 1)[0]
                
                return {
                    'cpu_trend': cpu_trend,
                    'memory_trend': memory_trend,
                    'predicted_cpu_5min': recent_history[-1].cpu_percent + (cpu_trend * 5),
                    'predicted_memory_5min': recent_history[-1].memory_percent + (memory_trend * 5)
                }
        
        def adaptive_resource_management(self):
            """Quản lý tài nguyên thích ứng dựa trên dự đoán"""
            try:
                current_stats = self.get_comprehensive_system_stats()
                if not current_stats:
                    return
                    
                predictions = self.predict_resource_usage()
                
                # Kiểm tra ngưỡng nguy hiểm
                critical_cpu = current_stats.cpu_percent > self.target_cpu * 0.9
                critical_memory = current_stats.memory_percent > 80.0  # Đặt cứng ngưỡng 80% cho RAM
                
                # Dự đoán sẽ quá tải trong 5 phút tới
                predicted_overload = (
                    predictions.get('predicted_cpu_5min', 0) > self.target_cpu or
                    predictions.get('predicted_memory_5min', 0) > 80.0  # Cũng áp dụng ngưỡng 80% cho dự đoán
                )
                
                if critical_cpu or critical_memory or predicted_overload:
                    self._performance_metrics['optimization_events'] += 1
                    self.logger.warning(f"Adaptive optimization triggered - "
                                    f"CPU: {current_stats.cpu_percent:.1f}%, "
                                    f"Memory: {current_stats.memory_percent:.1f}%")
                    
                    # Giải phóng tài nguyên
                    self.aggressive_cleanup()
                    
                    # Giảm số workers
                    if hasattr(self, 'thread_pool'):
                        self.thread_pool._max_workers = max(1, self.thread_pool._max_workers - 1)
                    
                    # GPU cleanup nếu có
                    if self.has_gpu:
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                self.logger.error(f"Error in adaptive resource management: {e}")
        
        def aggressive_cleanup(self):
            """Dọn dẹp tài nguyên một cách tích cực"""
            try:
                # Force garbage collection với thông số tối ưu
                gc.collect(generation=2)  # Full collection of all generations
                
                # GPU memory cleanup
                if self.has_gpu and torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            # Reset peak stats first
                            torch.cuda.reset_peak_memory_stats(i)
                            # Empty cache
                            torch.cuda.empty_cache()
                            # Force synchronize
                            torch.cuda.synchronize(i)
                    
                # Python memory optimization
                sys.modules.clear()  # Clear module cache
                
                # Giải phóng các tham chiếu không sử dụng
                locals_copy = list(locals().items())
                for name, obj in locals_copy:
                    if not name.startswith('_') and name != 'self':
                        del locals()[name]
                        
                # Free NumPy caches if available
                try:
                    np.random.seed(0)  # Reset random state
                    if hasattr(np, 'clear_cache'):
                        np.clear_cache()
                except:
                    pass
                    
                # Suggestion for OS to release memory
                if hasattr(psutil, 'Process'):
                    try:
                        p = psutil.Process()
                        if hasattr(p, 'memory_full_info'):
                            p.memory_full_info()  # Trigger memory info update
                    except:
                        pass
                        
                # Compact Python memory if on Windows
                if sys.platform == 'win32':
                    try:
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
                    except:
                        pass
                    
                self.logger.info("Aggressive cleanup completed")
                
            except Exception as e:
                self.logger.error(f"Error in aggressive cleanup: {e}")
        
        def get_optimal_device(self) -> str:
            """Chọn thiết bị tối ưu (GPU hoặc CPU) dựa trên tình trạng tài nguyên"""
            if not self.has_gpu:
                return "cpu"
                
            try:
                # Kiểm tra RAM trước - nếu RAM > 75%, ưu tiên dùng GPU nếu có thể
                memory = psutil.virtual_memory()
                high_ram_usage = memory.percent > 75.0
                
                gpu_stats = self.monitor_gpu()
                if not gpu_stats:
                    return "cpu"
                
                # Tìm GPU tốt nhất
                best_gpu = None
                best_score = float('inf')
                
                for gpu in gpu_stats:
                    # Tính điểm dựa trên load và memory usage
                    score = gpu['load'] * 0.6 + gpu['memoryUtil'] * 0.4
                    if score < best_score and gpu['load'] < self.max_gpu_load:
                        best_score = score
                        best_gpu = gpu
                
                # Nếu RAM cao và có GPU tốt, ưu tiên dùng GPU
                if high_ram_usage and best_gpu and (1 - best_gpu['memoryUtil']) >= self.min_free_gpu_mem:
                    self.logger.info(f"High RAM usage ({memory.percent:.1f}%), using GPU:{best_gpu['id']}")
                    return f"cuda:{best_gpu['id']}"
                    
                # Nếu không, sử dụng logic bình thường
                if best_gpu and (1 - best_gpu['memoryUtil']) >= self.min_free_gpu_mem:
                    return f"cuda:{best_gpu['id']}"
                else:
                    # Kiểm tra CPU load
                    cpu_stats = psutil.cpu_percent(interval=0.1)
                    if cpu_stats < self.target_cpu * 0.8 and not high_ram_usage:
                        return "cpu"
                    else:
                        # Chọn GPU ít tải nhất dù không lý tưởng nếu RAM cao
                        if gpu_stats and high_ram_usage:
                            least_loaded = min(gpu_stats, key=lambda x: x['load'])
                            return f"cuda:{least_loaded['id']}"
                        return "cpu"
                        
            except Exception as e:
                self.logger.error(f"Error selecting optimal device: {e}")
                return "cpu"
        
        def monitor_and_alert(self):
            """Giám sát liên tục và cảnh báo khi có vấn đề"""
            try:
                stats = self.get_comprehensive_system_stats()
                if not stats:
                    return
                    
                alerts = []
                
                # CPU alerts
                if stats.cpu_percent > self.target_cpu * 0.95:
                    alerts.append(f"CPU usage critical: {stats.cpu_percent:.1f}%")
                    
                # Memory alerts
                if stats.memory_percent > self.memory_threshold * 0.95:
                    alerts.append(f"Memory usage critical: {stats.memory_percent:.1f}%")
                    
                # GPU alerts
                for gpu in stats.gpu_metrics:
                    if 'load' in gpu and gpu['load'] > self.max_gpu_load * 0.95:
                        alerts.append(f"GPU {gpu.get('id', 'N/A')} load critical: {gpu['load']:.1f}")
                    if 'memoryUtil' in gpu and gpu['memoryUtil'] > (1 - self.min_free_gpu_mem) * 0.95:
                        alerts.append(f"GPU {gpu.get('id', 'N/A')} memory critical: {gpu['memoryUtil']:.1f}")
                        
                # Temperature alerts
                if stats.temperature and stats.temperature > 80:
                    alerts.append(f"High temperature: {stats.temperature:.1f}°C")
                    
                # Disk space alerts
                if stats.disk_usage_percent > 90:
                    alerts.append(f"Disk usage critical: {stats.disk_usage_percent:.1f}%")
                    
                if alerts:
                    self.logger.warning("System alerts: " + "; ".join(alerts))
                    
            except Exception as e:
                self.logger.error(f"Error in monitoring and alerts: {e}")

        # --- CPU / RAM methods ---

        def monitor_cpu(self) -> dict:
            """Trả về thông tin tài nguyên CPU, RAM và GPU"""
            stats = {
                'cpu': psutil.cpu_percent(interval=1),
                'memory': psutil.virtual_memory().percent
            }

            # Thêm thông tin GPU nếu có
            if self.has_gpu:
                gpu_stats = self.monitor_gpu()
                if gpu_stats:
                    stats['gpu'] = gpu_stats

            self.logger.debug(f"System stats: {stats}")
            return stats

        def suggest_workers(self, current_workers: int) -> int:
            """
            Đề xuất tăng/giảm số worker dựa trên CPU/RAM và GPU.
            """
            stats = self.monitor_cpu()
            cpu, mem = stats['cpu'], stats['memory']

            # Xét thêm GPU load nếu có
            gpu_overloaded = False
            if self.has_gpu and 'gpu' in stats:
                gpu_loads = [g['load'] for g in stats['gpu']]
                gpu_overloaded = any(load > self.max_gpu_load for load in gpu_loads)
                
            # Giảm worker nếu quá tải
            if cpu > self.target_cpu or mem > self.memory_threshold or gpu_overloaded:
                new = max(1, current_workers - 1)
                self.logger.info(f"High load (CPU={cpu}%, RAM={mem}%, GPU overloaded={gpu_overloaded}). "
                                f"Reducing workers: {current_workers} -> {new}")
            # Tăng worker nếu dư tải
            elif cpu < (self.target_cpu * 0.5) and mem < (self.memory_threshold * 0.5) and current_workers < self.max_workers:
                new = current_workers + 1
                self.logger.info(f"Under-utilized (CPU={cpu}%, RAM={mem}%). "
                                f"Increasing workers: {current_workers} -> {new}")
            else:
                new = current_workers
                self.logger.debug(f"Workers unchanged at {current_workers} "
                                f"(CPU={cpu}%, RAM={mem}%)")
            return new

        def throttle_cpu(self, pause_sec: float = 30.0):
            """
            Throttle (dừng) nếu CPU quá tải.
            """
            stats = self.monitor_cpu()
            should_throttle = stats['cpu'] > self.target_cpu
            
            # Thêm kiểm tra GPU
            if self.has_gpu and 'gpu' in stats:
                gpu_loads = [g['load'] for g in stats['gpu']]
                should_throttle = should_throttle or any(load > self.max_gpu_load for load in gpu_loads)

            if should_throttle:
                self.logger.warning(f"Throttling system for {pause_sec}s")
                if self.has_gpu:
                    torch.cuda.empty_cache()
                time.sleep(pause_sec)

        def throttle_ram(self, pause_sec: float = 30.0):
            # Lấy % tiêu thụ RAM (0–100)
            # Giữ nguyên logic hiện có
            mem_percent = self.monitor_cpu()['memory']
            threshold = self.memory_threshold * 100 if self.memory_threshold <= 1 else self.memory_threshold
            
            # Thêm kiểm tra GPU memory
            gpu_mem_full = False
            if self.has_gpu:
                gpu_stats = self.monitor_gpu()
                if gpu_stats:
                    gpu_mem_utils = [g['memoryUtil'] for g in gpu_stats]
                    gpu_mem_full = any(util > (1 - self.min_free_gpu_mem) for util in gpu_mem_utils)

            if mem_percent > threshold or gpu_mem_full:
                self.logger.warning(
                    f"Memory high - RAM: {mem_percent:.1f}% > {threshold}%, GPU full: {gpu_mem_full}"
                    f" → sleep {pause_sec}s"
                )
                if self.has_gpu:
                    torch.cuda.empty_cache()
                time.sleep(pause_sec)


        # --- GPU methods (nếu GPUtil cài đặt) ---

        def monitor_gpu(self) -> list:
            """
            Trả về list các dict GPU: 
            [{'id', 'load'(0-1), 'memoryFree'(MB), 'memoryUtil'(0-1)}, ...]
            """
            stats = []
            
            # Sử dụng GPUtil nếu có
            if GPUtil:
                try:
                    for gpu in GPUtil.getGPUs():
                        stats.append({
                            'id': gpu.id,
                            'load': gpu.load,
                            'memoryFree': gpu.memoryFree,
                            'memoryUtil': gpu.memoryUtil
                        })
                        self.logger.debug(f"GPU {gpu.id}: load={gpu.load:.2f}, "
                                        f"free_mem={gpu.memoryFree}MB, util={gpu.memoryUtil:.2f}")
                except Exception as e:
                    self.logger.warning(f"Error using GPUtil: {e}")
            
            # Fallback: sử dụng PyTorch CUDA nếu GPUtil không có
            elif self.has_gpu and torch.cuda.is_available():
                try:
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            props = torch.cuda.get_device_properties(i)
                            mem_reserved = torch.cuda.memory_reserved(i)
                            mem_allocated = torch.cuda.memory_allocated(i)
                            total_memory = props.total_memory
                            
                            stats.append({
                                'id': i,
                                'load': 0.0,  # PyTorch không cung cấp GPU load
                                'memoryFree': (total_memory - mem_allocated) / (1024**2),  # MB
                                'memoryUtil': mem_allocated / total_memory
                            })
                            
                            self.logger.debug(f"GPU {i}: mem_allocated={mem_allocated/1024**3:.2f}GB, "
                                            f"mem_reserved={mem_reserved/1024**3:.2f}GB")
                except Exception as e:
                    self.logger.warning(f"Error monitoring GPU with PyTorch: {e}")
            
            return stats

        def select_best_gpu(self):
            """
            Chọn GPU phù hợp theo load <= max_gpu_load 
            và free mem >= min_free_gpu_mem.
            Trả về GPU id hoặc None.
            """
            gpus = self.monitor_gpu()
            candidates = [
                g for g in gpus
                if g['load'] <= self.max_gpu_load
                and (1 - g['memoryUtil']) >= self.min_free_gpu_mem
            ]
            if not candidates:
                self.logger.warning("No suitable GPU found")
                return None
            # Chọn GPU có load thấp nhất, nếu bằng thì free mem cao nhất
            best = sorted(candidates, key=lambda g: (g['load'], -g['memoryFree']))[0]
            self.logger.info(f"Selected GPU {best['id']} "
                            f"(load={best['load']:.2f}, free={best['memoryFree']}MB)")
            return best['id']

        def throttle_gpu(self, pause_sec: float = 60.0):
            """
            Nếu tất cả GPU đều quá tải, đợi pause_sec rồi thử lại.
            """
            gpus = self.monitor_gpu()
            if gpus and all(
                (g['load'] > self.max_gpu_load or 
                (1 - g['memoryUtil']) < self.min_free_gpu_mem)
                for g in gpus
            ):
                self.logger.warning(f"All GPUs overloaded, throttling for {pause_sec}s")
                time.sleep(pause_sec)

        def cleanup_resources(self):
            """Dọn dẹp tài nguyên sau khi xử lý"""
            try:
                # Giải phóng GPU memory
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Giải phóng CPU memory
                gc.collect()
                
                # Đóng các file handlers
                handlers = logging.getLogger().handlers[:]
                for handler in handlers:
                    handler.close()
                    logging.getLogger().removeHandler(handler)
                    
                # Reset các biến training_status
                training_status.update({
                    "train_progress": 0.0,
                    "train_status": "",
                    "train_error": None,
                    "predicting": False,
                    "predict_done": 0,
                    "predict_total_steps": 0
                })
                
                # Đóng các thread pools nếu có
                if hasattr(self, 'thread_pool'):
                    self.thread_pool.shutdown(wait=True)
                    
            except Exception as e:
                logging.error(f"Lỗi khi cleanup: {str(e)}")
        
        # --- Advanced monitoring methods ---
        
        def get_system_health_score(self) -> float:
            """Tính điểm sức khỏe hệ thống từ 0-100"""
            try:
                stats = self.get_comprehensive_system_stats()
                if not stats:
                    return 0.0
                    
                # Tính điểm từng thành phần
                cpu_score = max(0, 100 - stats.cpu_percent)
                memory_score = max(0, 100 - stats.memory_percent)
                disk_score = max(0, 100 - stats.disk_usage_percent)
                
                # GPU score
                gpu_score = 100
                if stats.gpu_metrics:
                    gpu_loads = [gpu.get('load', 0) * 100 for gpu in stats.gpu_metrics]
                    gpu_memory_utils = [gpu.get('memoryUtil', 0) * 100 for gpu in stats.gpu_metrics]
                    gpu_score = max(0, 100 - max(gpu_loads + gpu_memory_utils))
                
                # Temperature score
                temp_score = 100
                if stats.temperature:
                    temp_score = max(0, 100 - max(0, stats.temperature - 30) * 2)
                
                # Weighted average
                total_score = (
                    cpu_score * 0.25 +
                    memory_score * 0.25 +
                    disk_score * 0.15 +
                    gpu_score * 0.20 +
                    temp_score * 0.15
                )
                
                return min(100, max(0, total_score))
                
            except Exception as e:
                self.logger.error(f"Error calculating health score: {e}")
                return 0.0
        
        def get_resource_recommendations(self) -> List[str]:
            """Đưa ra các khuyến nghị tối ưu hóa tài nguyên"""
            try:
                stats = self.get_comprehensive_system_stats()
                if not stats:
                    return ["Unable to collect system stats"]
                    
                recommendations = []
                
                # CPU recommendations
                if stats.cpu_percent > 90:
                    recommendations.append("CPU usage is critical. Consider reducing workload or upgrading CPU.")
                elif stats.cpu_percent > 70:
                    recommendations.append("High CPU usage detected. Monitor for potential bottlenecks.")
                    
                # Memory recommendations
                if stats.memory_percent > 90:
                    recommendations.append("Memory usage is critical. Close unnecessary applications or add more RAM.")
                elif stats.memory_percent > 70:
                    recommendations.append("High memory usage. Consider memory optimization.")
                    
                # GPU recommendations
                for gpu in stats.gpu_metrics:
                    gpu_id = gpu.get('id', 'N/A')
                    if gpu.get('memoryUtil', 0) > 0.9:
                        recommendations.append(f"GPU {gpu_id} memory usage is critical.")
                    elif gpu.get('load', 0) > 0.9:
                        recommendations.append(f"GPU {gpu_id} load is very high.")
                        
                # Disk recommendations
                if stats.disk_usage_percent > 90:
                    recommendations.append("Disk space is critical. Clean up unnecessary files.")
                elif stats.disk_usage_percent > 80:
                    recommendations.append("Disk space is getting low. Consider cleanup.")
                    
                # Temperature recommendations
                if stats.temperature and stats.temperature > 85:
                    recommendations.append("System temperature is high. Check cooling system.")
                    
                if not recommendations:
                    recommendations.append("System is running optimally.")
                    
                return recommendations
                
            except Exception as e:
                self.logger.error(f"Error generating recommendations: {e}")
                return [f"Error generating recommendations: {e}"]
        
        def start_background_monitoring(self, interval: int = 60):
            """Bắt đầu giám sát tài nguyên trong background"""
            try:
                # Nếu đã có thread giám sát đang chạy, không tạo thread mới
                if ResourceOptimizer._monitor_thread is not None and ResourceOptimizer._monitor_thread.is_alive():
                    self.logger.debug(f"Background monitoring đã chạy với interval {interval}s")
                    return
                    
                def monitor_loop():
                    while True:
                        try:
                            # Kiểm tra mức sử dụng RAM hiện tại
                            memory = psutil.virtual_memory()
                            if memory.percent > 80:
                                self.logger.warning(f"RAM usage critical: {memory.percent:.1f}% > 80%. Performing emergency cleanup.")
                                self.aggressive_cleanup()
                                time.sleep(5)  # Nghỉ ngắn để cho cleanup có hiệu quả
                                continue
                                
                            # Thu thập thống kê
                            stats = self.get_comprehensive_system_stats()
                            if stats:
                                # Lưu vào performance metrics
                                self._performance_metrics['cpu_utilization'].append(stats.cpu_percent)
                                self._performance_metrics['memory_utilization'].append(stats.memory_percent)
                                
                                # Giữ chỉ 100 điểm dữ liệu gần nhất
                                for key in ['cpu_utilization', 'memory_utilization', 'gpu_utilization']:
                                    if len(self._performance_metrics[key]) > 100:
                                        self._performance_metrics[key] = self._performance_metrics[key][-100:]
                            
                            # Kiểm tra cảnh báo
                            self.monitor_and_alert()
                            
                            # Quản lý tài nguyên thích ứng
                            self.adaptive_resource_management()
                            
                            time.sleep(interval)
                            
                        except Exception as e:
                            self.logger.error(f"Error in monitoring loop: {e}")
                            time.sleep(interval)
                
                # Chạy trong background thread
                ResourceOptimizer._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
                ResourceOptimizer._monitor_thread.start()
                self.logger.info(f"Background monitoring started with {interval}s interval")
                
            except Exception as e:
                self.logger.error(f"Error starting background monitoring: {e}")
        
        def get_performance_report(self) -> Dict[str, Any]:
            """Tạo báo cáo hiệu suất chi tiết"""
            try:
                current_stats = self.get_comprehensive_system_stats()
                health_score = self.get_system_health_score()
                recommendations = self.get_resource_recommendations()
                
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'health_score': health_score,
                    'current_stats': {
                        'cpu_percent': current_stats.cpu_percent if current_stats else 0,
                        'memory_percent': current_stats.memory_percent if current_stats else 0,
                        'memory_available_gb': current_stats.memory_available_gb if current_stats else 0,
                        'disk_usage_percent': current_stats.disk_usage_percent if current_stats else 0,
                        'temperature': current_stats.temperature if current_stats else None,
                        'gpu_count': len(current_stats.gpu_metrics) if current_stats else 0
                    },
                    'performance_metrics': {
                        'avg_cpu': np.mean(self._performance_metrics['cpu_utilization']) if self._performance_metrics['cpu_utilization'] else 0,
                        'avg_memory': np.mean(self._performance_metrics['memory_utilization']) if self._performance_metrics['memory_utilization'] else 0,
                        'throttle_events': self._performance_metrics['throttle_events'],
                        'optimization_events': self._performance_metrics['optimization_events']
                    },
                    'recommendations': recommendations,
                    'gpu_info': {
                        'has_gpu': self.has_gpu,
                        'gpu_count': self.gpu_count,
                        'gpu_names': self.gpu_names
                    }
                }
                
                return report
                
            except Exception as e:
                self.logger.error(f"Error generating performance report: {e}")
                return {'error': str(e)}
        
        def get_memory_usage(self) -> float:
            """Trả về tỉ lệ sử dụng bộ nhớ (0.0-1.0)"""
            try:
                memory = psutil.virtual_memory()
                return memory.percent / 100.0
            except Exception as e:
                # Nếu có logger
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error getting memory usage: {e}")
                else:
                    print(f"Error getting memory usage: {e}")
                return 0.0

    # Global instance để sử dụng trong toàn bộ ứng dụng
    global_resource_optimizer = ResourceOptimizer()
    # ai_optimizer/gpu_manager.py
    """
    GPU Detection và Management tập trung
    Loại bỏ tất cả các GPU detection code trùng lặp và hợp nhất các chức năng từ gpu_optimizer.py
    """



    logger = get_logger('gpu_manager')

    # GPU imports with fallbacks
    try:
        TORCH_AVAILABLE = True
    except ImportError:
        torch = None
        TORCH_AVAILABLE = False

    try:
        GPUTIL_AVAILABLE = True
    except ImportError:
        GPUtil = None
        GPUTIL_AVAILABLE = False

    try:
        pynvml.nvmlInit()
        PYNVML_AVAILABLE = True
    except ImportError:
        pynvml = None
        PYNVML_AVAILABLE = False


    
    def GPUInfo:
        """GPU information container"""
        id: int
        name: str
        memory_total: float  # GB
        memory_free: float   # GB
        memory_used: float   # GB
        memory_percent: float
        temperature: Optional[float] = None
        power_usage: Optional[float] = None
        utilization: float = 0.0
        compute_capability: Optional[Tuple[int, int]] = None
        is_available: bool = True


    
    class GPUAllocation:
        """GPU allocation tracking"""
        gpu_id: int
        allocated_memory: float  # GB
        allocated_tasks: List[str]
        max_memory: float  # GB
        reservation_time: float


    class GPUManager:
        """Quản lý GPU tập trung cho toàn bộ ứng dụng"""
        
        def __init__(self, 
                    max_gpu_memory_fraction: float = 0.8,
                    temperature_threshold: float = 85.0,
                    memory_reserve_gb: float = 1.0):
            
            # Cấu hình cơ bản
            self.use_gpu = False
            self.gpu_count = 0
            self.gpu_info = []
            self.selected_gpu_id = None
            
            # Cấu hình nâng cao
            self.max_memory_fraction = max_gpu_memory_fraction
            self.temperature_threshold = temperature_threshold
            self.memory_reserve_gb = memory_reserve_gb
            
            # Tracking và caching
            self._allocation_lock = RLock()
            self._gpu_allocations: Dict[int, GPUAllocation] = {}
            self._gpu_info_cache: Dict[int, GPUInfo] = {}
            self._last_gpu_scan = 0.0
            self._scan_interval = 5.0  # seconds
            
            # GPU tracking nâng cao
            self.available_gpus: List[GPUInfo] = []
            self.has_gpu = False
            
            # Tiến hành phát hiện GPU
            self._detect_gpu()
            self._setup_gpu_environment()
        
        def _detect_gpu(self):
            """Phát hiện GPU availability kết hợp các phương pháp từ cả hai module"""
            detected_gpus = []
            
            # Method 1: PyTorch CUDA detection
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        
                        # Get memory info
                        with torch.cuda.device(i):
                            memory_total = props.total_memory / (1024**3)  # Convert to GB
                            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                            memory_free = memory_total - memory_allocated
                            memory_percent = (memory_allocated / memory_total) * 100
                        
                        gpu_info = GPUInfo(
                            id=i,
                            name=props.name,
                            memory_total=memory_total,
                            memory_free=memory_free,
                            memory_used=memory_allocated,
                            memory_percent=memory_percent,
                            compute_capability=(props.major, props.minor),
                            is_available=memory_free > self.memory_reserve_gb
                        )
                        
                        detected_gpus.append(gpu_info)
                        logger.info(f"PyTorch detected GPU {i}: {props.name} "
                                f"({memory_total:.1f}GB total, {memory_free:.1f}GB free)")
                        
                        # Thêm thông tin cơ bản vào gpu_info
                        self.gpu_info.append({
                            'id': i,
                            'name': props.name,
                            'memory_total': memory_total * 1024,  # Convert to MB for compatibility
                            'memory_free': memory_free * 1024,
                            'memory_used': memory_allocated * 1024,
                            'load': 0.0,  # Mặc định, sẽ cập nhật sau nếu có GPUtil
                            'temperature': None  # Mặc định, sẽ cập nhật sau nếu có GPUtil
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to get PyTorch GPU {i} info: {e}")
            
            # Method 2: GPUtil detection (bổ sung thông tin nếu PyTorch thành công hoặc thay thế nếu PyTorch thất bại)
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    
                    if not detected_gpus:  # Nếu PyTorch detection thất bại
                        self.gpu_info = []  # Reset nếu không có GPU nào được phát hiện qua PyTorch
                        
                        for gpu in gpus:
                            memory_total = gpu.memoryTotal / 1024  # MB to GB
                            memory_free = gpu.memoryFree / 1024
                            memory_used = gpu.memoryUsed / 1024
                            
                            gpu_info = GPUInfo(
                                id=gpu.id,
                                name=gpu.name,
                                memory_total=memory_total,
                                memory_free=memory_free,
                                memory_used=memory_used,
                                memory_percent=gpu.memoryUtil * 100,
                                temperature=gpu.temperature,
                                utilization=gpu.load * 100,
                                is_available=memory_free > self.memory_reserve_gb
                            )
                            
                            detected_gpus.append(gpu_info)
                            
                            # Thêm thông tin cơ bản vào gpu_info
                            self.gpu_info.append({
                                'id': gpu.id,
                                'name': gpu.name,
                                'memory_total': gpu.memoryTotal,
                                'memory_free': gpu.memoryFree,
                                'memory_used': gpu.memoryUsed,
                                'load': gpu.load,
                                'temperature': gpu.temperature
                            })
                            
                            logger.info(f"GPUtil detected GPU {gpu.id}: {gpu.name} "
                                    f"({memory_total:.1f}GB total, {memory_free:.1f}GB free)")
                    
                    else:  # Nếu PyTorch đã thành công, bổ sung thêm thông tin
                        for i, gpu_info_dict in enumerate(self.gpu_info):
                            for gpu in gpus:
                                if gpu.id == gpu_info_dict['id']:
                                    # Cập nhật thông tin từ GPUtil
                                    gpu_info_dict['load'] = gpu.load
                                    gpu_info_dict['temperature'] = gpu.temperature
                                    
                                    # Cập nhật thông tin trong detected_gpus
                                    for detected_gpu in detected_gpus:
                                        if detected_gpu.id == gpu.id:
                                            detected_gpu.temperature = gpu.temperature
                                            detected_gpu.utilization = gpu.load * 100
                    
                except Exception as e:
                    logger.warning(f"GPUtil detection failed: {e}")
            
            # Method 3: NVIDIA ML detection (bổ sung thêm thông tin chuyên sâu)
            if PYNVML_AVAILABLE:
                try:
                    device_count = pynvml.nvmlDeviceGetCount()
                    
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                        
                        # Memory info
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        memory_total = mem_info.total / (1024**3)
                        memory_free = mem_info.free / (1024**3)
                        memory_used = mem_info.used / (1024**3)
                        
                        # Temperature
                        try:
                            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        except:
                            temperature = None
                        
                        # Power usage
                        try:
                            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                        except:
                            power_usage = None
                        
                        # Utilization
                        try:
                            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            utilization = util_info.gpu
                        except:
                            utilization = 0.0
                        
                        # Check if this GPU is already detected
                        existing_gpu = next((gpu for gpu in detected_gpus if gpu.name == name), None)
                        
                        if existing_gpu:
                            # Update existing GPU info with NVML data
                            existing_gpu.temperature = temperature
                            existing_gpu.power_usage = power_usage
                            existing_gpu.utilization = utilization
                        else:
                            # Add new GPU info
                            gpu_info = GPUInfo(
                                id=i,
                                name=name,
                                memory_total=memory_total,
                                memory_free=memory_free,
                                memory_used=memory_used,
                                memory_percent=(memory_used / memory_total) * 100,
                                temperature=temperature,
                                power_usage=power_usage,
                                utilization=utilization,
                                is_available=memory_free > self.memory_reserve_gb
                            )
                            detected_gpus.append(gpu_info)
                            
                            # Thêm thông tin cơ bản vào gpu_info nếu chưa tồn tại
                            existing = False
                            for info in self.gpu_info:
                                if info['id'] == i:
                                    existing = True
                                    # Cập nhật thông tin nâng cao
                                    info['temperature'] = temperature
                                    break
                                    
                            if not existing:
                                self.gpu_info.append({
                                    'id': i,
                                    'name': name,
                                    'memory_total': memory_total * 1024,  # Convert to MB for compatibility
                                    'memory_free': memory_free * 1024,
                                    'memory_used': memory_used * 1024,
                                    'load': utilization / 100.0 if utilization is not None else 0.0,
                                    'temperature': temperature
                                })
                        
                except Exception as e:
                    logger.warning(f"NVIDIA-ML detection failed: {e}")
            
            # Kết luận và hoàn thiện thông tin GPU
            self.available_gpus = [gpu for gpu in detected_gpus if gpu.is_available]
            self.has_gpu = len(self.available_gpus) > 0
            self.gpu_count = len(self.available_gpus)
            self.use_gpu = self.has_gpu
            
            # Lưu thông tin vào cache
            self._gpu_info_cache = {gpu.id: gpu for gpu in detected_gpus}
            
            # Cập nhật last scan time
            self._last_gpu_scan = time.time()
            
            # Log thông tin
            if self.has_gpu:
                logger.info(f"✅ GPU detection successful: {self.gpu_count} GPU(s) found")
                if self.gpu_info:
                    logger.info(f"Primary GPU: {self.gpu_info[0]['name']}, VRAM={self.gpu_info[0]['memory_total']}MB")
                
                for gpu in self.available_gpus:
                    logger.info(f"  GPU {gpu.id}: {gpu.name} - "
                            f"{gpu.memory_free:.1f}GB free / {gpu.memory_total:.1f}GB total")
            else:
                logger.info("❌ No suitable GPUs found, using CPU")
                
            # Test GPU functionality nếu có
            if self.has_gpu and TORCH_AVAILABLE:
                try:
                    test_tensor = torch.tensor([1.0]).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"⚠️ GPU test failed: {e}, fallback to CPU")
                    self.use_gpu = False
        
        def _setup_gpu_environment(self) -> None:
            """Setup optimal GPU environment"""
            
            if not self.has_gpu or not TORCH_AVAILABLE:
                return
            
            try:
                # Enable optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Set memory fractions for each GPU
                for gpu in self.available_gpus:
                    with torch.cuda.device(gpu.id):
                        torch.cuda.set_per_process_memory_fraction(self.max_memory_fraction)
                        torch.cuda.empty_cache()
                
                logger.info("GPU environment configured successfully")
                
            except Exception as e:
                logger.error(f"Failed to setup GPU environment: {e}")
        
        def get_gpu_status(self, force_refresh: bool = False) -> List[GPUInfo]:
            """Get current GPU status with caching"""
            
            current_time = time.time()
            
            if not force_refresh and (current_time - self._last_gpu_scan) < self._scan_interval:
                return list(self._gpu_info_cache.values())
            
            updated_gpus = []
            
            for gpu in self.available_gpus:
                try:
                    # Update GPU info
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        with torch.cuda.device(gpu.id):
                            memory_total = gpu.memory_total
                            memory_allocated = torch.cuda.memory_allocated(gpu.id) / (1024**3)
                            memory_reserved = torch.cuda.memory_reserved(gpu.id) / (1024**3)
                            memory_free = memory_total - memory_allocated
                            memory_percent = (memory_allocated / memory_total) * 100
                            
                            updated_gpu = GPUInfo(
                                id=gpu.id,
                                name=gpu.name,
                                memory_total=memory_total,
                                memory_free=memory_free,
                                memory_used=memory_allocated,
                                memory_percent=memory_percent,
                                temperature=gpu.temperature,
                                power_usage=gpu.power_usage,
                                utilization=gpu.utilization,
                                compute_capability=gpu.compute_capability,
                                is_available=memory_free > self.memory_reserve_gb
                            )
                            
                            updated_gpus.append(updated_gpu)
                    else:
                        updated_gpus.append(gpu)
                        
                except Exception as e:
                    logger.warning(f"Failed to update GPU {gpu.id} status: {e}")
                    updated_gpus.append(gpu)
            
            # Update cache
            self._gpu_info_cache = {gpu.id: gpu for gpu in updated_gpus}
            self._last_gpu_scan = current_time
            
            return updated_gpus
        
        def select_best_gpu(self, min_free_memory_gb: float = 1.0, max_load: float = 0.8, prefer_low_utilization: bool = True) -> Optional[int]:
            """
            Chọn GPU tối ưu dựa trên memory, load và các tiêu chí phụ
            
            Args:
                min_free_memory_gb: Minimum free memory in GB
                max_load: Maximum GPU load (0.0 to 1.0)
                prefer_low_utilization: Prefer GPUs with lower utilization
                
            Returns:
                GPU ID hoặc None nếu không có GPU phù hợp
            """
            if not self.use_gpu:
                return None
                
            try:
                # Lấy thông tin GPU từ hai nguồn
                current_gpus = self.get_gpu_status(force_refresh=True)
                
                if not current_gpus:
                    return None
                    
                # Basic method (for legacy support)
                if GPUTIL_AVAILABLE:
                    gpus = GPUtil.getGPUs()
                    
                    suitable_gpus = []
                    for gpu in gpus:
                        free_memory_gb = gpu.memoryFree / 1024  # Convert MB to GB
                        if free_memory_gb >= min_free_memory_gb and gpu.load <= max_load:
                            suitable_gpus.append((gpu.id, gpu.load, free_memory_gb))
                    
                    if suitable_gpus:
                        # Sort by load (ascending) then by free memory (descending)
                        suitable_gpus.sort(key=lambda x: (x[1], -x[2]))
                        best_gpu_id = suitable_gpus[0][0]
                        self.selected_gpu_id = best_gpu_id
                        logger.info(f"🚀 Selected GPU {best_gpu_id} (Load: {suitable_gpus[0][1]:.2f}, Free: {suitable_gpus[0][2]:.1f}GB)")
                        return best_gpu_id
                    else:
                        logger.warning("⚠️ No suitable GPU found based on criteria")
                
                # Advanced method
                available_gpus = [gpu for gpu in current_gpus if gpu.is_available]
                
                if not available_gpus:
                    logger.warning("No GPUs available for allocation")
                    return None
                
                # Filter by memory requirement
                suitable_gpus = [gpu for gpu in available_gpus 
                                if gpu.memory_free >= min_free_memory_gb]
                
                if not suitable_gpus:
                    logger.warning(f"No GPUs with {min_free_memory_gb:.1f}GB free memory available")
                    return None
                
                # Score GPUs based on multiple criteria
                def score_gpu(gpu: GPUInfo) -> float:
                    score = 0.0
                    
                    # Memory availability (40% weight)
                    memory_score = gpu.memory_free / gpu.memory_total
                    score += memory_score * 0.4
                    
                    # Low utilization (30% weight)
                    if prefer_low_utilization:
                        util_score = 1.0 - (gpu.utilization / 100.0)
                        score += util_score * 0.3
                    
                    # Temperature (20% weight)
                    if gpu.temperature is not None:
                        temp_score = max(0, 1.0 - (gpu.temperature / self.temperature_threshold))
                        score += temp_score * 0.2
                    else:
                        score += 0.2  # Default if no temperature data
                    
                    # Existing allocations (10% weight)
                    with self._allocation_lock:
                        allocation = self._gpu_allocations.get(gpu.id)
                        if allocation:
                            alloc_score = 1.0 - (len(allocation.allocated_tasks) / 10.0)  # Penalize many tasks
                            score += max(0, alloc_score) * 0.1
                        else:
                            score += 0.1
                    
                    return score
                
                # Select best GPU
                best_gpu = max(suitable_gpus, key=score_gpu)
                self.selected_gpu_id = best_gpu.id
                
                logger.info(f"Selected GPU {best_gpu.id} ({best_gpu.name}) - "
                        f"{best_gpu.memory_free:.1f}GB free, "
                        f"{best_gpu.utilization:.1f}% util")
                
                return best_gpu.id
                    
            except Exception as e:
                logger.error(f"❌ GPU selection failed: {e}")
                return None
        
        def set_gpu_device(self, gpu_id: Optional[int] = None):
            """Set CUDA_VISIBLE_DEVICES environment variable"""
            if gpu_id is None:
                gpu_id = self.select_best_gpu()
                
            if gpu_id is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                self.selected_gpu_id = gpu_id
                logger.info(f"🎯 Set CUDA_VISIBLE_DEVICES to GPU {gpu_id}")
            else:
                # Disable GPU
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                logger.info("💻 GPU disabled, using CPU")
        
        def get_gpu_config_for_xgboost(self) -> dict:
            """Get XGBoost GPU configuration"""
            if self.use_gpu and self.selected_gpu_id is not None:
                return {
                    'tree_method': 'gpu_hist',
                    'predictor': 'gpu_predictor',
                    'gpu_id': self.selected_gpu_id
                }
            else:
                return {
                    'tree_method': 'hist',
                    'predictor': 'auto'
                }
        
        def get_gpu_config_for_lightgbm(self) -> dict:
            """Get LightGBM GPU configuration"""
            if self.use_gpu and self.selected_gpu_id is not None:
                return {
                    'device': 'gpu',
                    'gpu_device_id': self.selected_gpu_id
                }
            else:
                return {
                    'device': 'cpu'
                }
        
        def get_gpu_config_for_catboost(self) -> dict:
            """Get CatBoost GPU configuration"""
            if self.use_gpu and self.selected_gpu_id is not None:
                return {
                    'task_type': 'GPU',
                    'devices': str(self.selected_gpu_id)
                }
            else:
                return {
                    'task_type': 'CPU'
                }
        
        def allocate_gpu_memory(self, 
                            gpu_id: int, 
                            memory_gb: float, 
                            task_name: str) -> bool:
            """Allocate GPU memory for a task"""
            
            with self._allocation_lock:
                current_gpus = self.get_gpu_status(force_refresh=True)
                target_gpu = next((gpu for gpu in current_gpus if gpu.id == gpu_id), None)
                
                if target_gpu is None:
                    logger.error(f"GPU {gpu_id} not found")
                    return False
                
                if target_gpu.memory_free < memory_gb:
                    logger.warning(f"GPU {gpu_id} insufficient memory: "
                                f"{target_gpu.memory_free:.1f}GB available, "
                                f"{memory_gb:.1f}GB required")
                    return False
                
                # Create or update allocation
                if gpu_id not in self._gpu_allocations:
                    self._gpu_allocations[gpu_id] = GPUAllocation(
                        gpu_id=gpu_id,
                        allocated_memory=0.0,
                        allocated_tasks=[],
                        max_memory=target_gpu.memory_total * self.max_memory_fraction,
                        reservation_time=time.time()
                    )
                
                allocation = self._gpu_allocations[gpu_id]
                
                if allocation.allocated_memory + memory_gb > allocation.max_memory:
                    logger.warning(f"GPU {gpu_id} memory allocation would exceed limit")
                    return False
                
                # Update allocation
                allocation.allocated_memory += memory_gb
                allocation.allocated_tasks.append(task_name)
                allocation.reservation_time = time.time()
                
                logger.info(f"Allocated {memory_gb:.1f}GB on GPU {gpu_id} for {task_name}")
                return True
        
        def release_gpu_memory(self, gpu_id: int, memory_gb: float, task_name: str) -> None:
            """Release GPU memory from a task"""
            
            with self._allocation_lock:
                if gpu_id not in self._gpu_allocations:
                    logger.warning(f"No allocation found for GPU {gpu_id}")
                    return
                
                allocation = self._gpu_allocations[gpu_id]
                
                # Remove task and memory
                if task_name in allocation.allocated_tasks:
                    allocation.allocated_tasks.remove(task_name)
                
                allocation.allocated_memory = max(0, allocation.allocated_memory - memory_gb)
                
                # Clean up empty allocations
                if allocation.allocated_memory == 0 and not allocation.allocated_tasks:
                    del self._gpu_allocations[gpu_id]
                
                logger.info(f"Released {memory_gb:.1f}GB on GPU {gpu_id} from {task_name}")
        
        def test_gpu_functionality(self, model_class, X_sample, y_sample):
            """
            Test GPU functionality với model cụ thể
            
            Args:
                model_class: Class của model (XGBClassifier, LGBMClassifier, etc.)
                X_sample: Sample data X
                y_sample: Sample data y
                
            Returns:
                bool: True nếu GPU hoạt động tốt, False nếu cần fallback CPU
            """
            if not self.use_gpu:
                return False
                
            try:
                # Get appropriate config based on model type
                if 'XGB' in str(model_class):
                    config = self.get_gpu_config_for_xgboost()
                elif 'LGBM' in str(model_class):
                    config = self.get_gpu_config_for_lightgbm()
                elif 'CatBoost' in str(model_class):
                    config = self.get_gpu_config_for_catboost()
                else:
                    return False
                
                # Test model creation and fitting
                model = model_class(
                    **config,
                    verbosity=0 if 'verbose' in config else None,
                    verbose=-1 if 'verbose' not in config else None,
                    silent=True if 'CatBoost' in str(model_class) else None,
                    random_state=42
                )
                
                # Filter None values
                model_params = {k: v for k, v in model.__dict__.items() if v is not None}
                
                # Try fitting on small sample
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        model.fit(X_sample, y_sample)
                
                logger.info(f"✅ GPU test successful for {model_class.__name__}")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ GPU test failed for {model_class.__name__}: {e}")
                return False

        def get_optimal_batch_size(self, 
                                gpu_id: int, 
                                model_size_mb: float,
                                base_batch_size: int = 32) -> int:
            """Calculate optimal batch size for GPU"""
            
            current_gpus = self.get_gpu_status()
            target_gpu = next((gpu for gpu in current_gpus if gpu.id == gpu_id), None)
            
            if target_gpu is None:
                return base_batch_size
            
            # Estimate memory usage
            available_memory_gb = target_gpu.memory_free * 0.8  # Leave 20% buffer
            model_memory_gb = model_size_mb / 1024
            
            # Rough estimation: each batch item uses ~2x model memory
            memory_per_batch_item = model_memory_gb * 2
            
            if memory_per_batch_item > 0:
                max_batch_size = int(available_memory_gb / memory_per_batch_item)
                optimal_batch_size = min(max_batch_size, base_batch_size * 4)  # Cap at 4x base
                optimal_batch_size = max(optimal_batch_size, 1)  # Minimum 1
            else:
                optimal_batch_size = base_batch_size
            
            logger.info(f"Optimal batch size for GPU {gpu_id}: {optimal_batch_size} "
                    f"(available memory: {available_memory_gb:.1f}GB)")
            
            return optimal_batch_size
        
        def cleanup_gpu_memory(self):
            """Clean up GPU memory"""
            if self.use_gpu:
                try:
                    if TORCH_AVAILABLE:
                        # Clean all GPUs
                        for gpu in self.available_gpus:
                            with torch.cuda.device(gpu.id):
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                        logger.debug("🧹 GPU memory cleaned")
                        
                        # Force garbage collection
                        gc.collect()
                except Exception as e:
                    logger.warning(f"GPU memory cleanup failed: {e}")
        
        def monitor_gpu_health(self) -> Dict[str, Any]:
            """Monitor GPU health and return status"""
            
            if not self.use_gpu:
                return {"status": "no_gpu", "gpus": []}
            
            current_gpus = self.get_gpu_status(force_refresh=True)
            health_status = {"status": "healthy", "gpus": [], "warnings": []}
            
            for gpu in current_gpus:
                gpu_status = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_usage": gpu.memory_percent,
                    "utilization": gpu.utilization,
                    "temperature": gpu.temperature,
                    "power_usage": gpu.power_usage,
                    "healthy": True,
                    "issues": []
                }
                
                # Check for issues
                if gpu.temperature and gpu.temperature > self.temperature_threshold:
                    gpu_status["healthy"] = False
                    gpu_status["issues"].append(f"High temperature: {gpu.temperature}°C")
                    health_status["warnings"].append(f"GPU {gpu.id} overheating")
                
                if gpu.memory_percent > 95:
                    gpu_status["healthy"] = False
                    gpu_status["issues"].append(f"High memory usage: {gpu.memory_percent:.1f}%")
                    health_status["warnings"].append(f"GPU {gpu.id} memory critical")
                
                if gpu.utilization > 95:
                    gpu_status["issues"].append(f"High utilization: {gpu.utilization:.1f}%")
                
                health_status["gpus"].append(gpu_status)
            
            # Overall health status
            unhealthy_gpus = [gpu for gpu in health_status["gpus"] if not gpu["healthy"]]
            if unhealthy_gpus:
                health_status["status"] = "warning" if len(unhealthy_gpus) < len(current_gpus) else "critical"
            
            return health_status
        
        def get_gpu_recommendations(self) -> List[str]:
            """Get optimization recommendations"""
            
            recommendations = []
            
            if not self.use_gpu:
                recommendations.append("Consider using GPU acceleration for faster training")
                return recommendations
            
            health_status = self.monitor_gpu_health()
            
            for gpu_info in health_status["gpus"]:
                gpu_id = gpu_info["id"]
                
                if gpu_info["memory_usage"] > 90:
                    recommendations.append(f"GPU {gpu_id}: Reduce batch size or model complexity")
                
                if gpu_info["temperature"] and gpu_info["temperature"] > 80:
                    recommendations.append(f"GPU {gpu_id}: Check cooling, temperature high")
                
                if gpu_info["utilization"] < 30:
                    recommendations.append(f"GPU {gpu_id}: Increase batch size for better utilization")
            
            # General recommendations
            with self._allocation_lock:
                if len(self._gpu_allocations) > len(self.available_gpus):
                    recommendations.append("Consider load balancing across multiple GPUs")
            
            return recommendations
        
        def get_status(self) -> dict:
            """Get current GPU status"""
            return {
                'use_gpu': self.use_gpu,
                'gpu_count': self.gpu_count,
                'selected_gpu_id': self.selected_gpu_id,
                'gpu_info': self.gpu_info
            }


class OtherFunctions:
    """
    Other Functions Class - Other functions + main program
    
    Handles all remaining functions including:
    - Streamlit UI and visualization
    - Utility helpers and configuration
    - Communication modules (Zalo, etc.)
    - File I/O and data persistence
    - Main application entry points
    """
    
    def __init__(self):
        """Initialize other functions with global instances"""
        self.logger = get_logger('other_functions')
        self.settings = settings
        
    # UI and Visualization Methods (will be moved here)
    # - main() function
    # - Streamlit interface functions
    # - create_price_chart() and visualization utilities
    
    # Utility Methods (will be moved here)
    # - Configuration management functions
    # - File I/O utilities
    # - Helper functions
    
    # Communication Methods (will be moved here)
    # - ZaloSender class methods
    # - Communication and notification utilities

# ===================== REMOVED ALL DUPLICATE CODE =====================

# ❌ REMOVED: 
#   - Duplicate get_news() function in DataCollection 
#   - Duplicate MarketAnalysis methods scattered in NewsAnalysis
#   - Duplicate RSS fetching functions 
#   - Duplicate translation functions
#   - Duplicate sentiment analysis code
#   - Redundant wrapper functions
#   - Multiple class definitions for same functionality
#   - Scattered variables and imports 
#   - Incomplete class definitions
#   - Orphaned function stubs

# All functionality is now consolidated into 3 main classes:
# 1. NewsAnalysis - Complete news operations
# 2. MarketAnalysis - Complete market analysis  
# 3. DataCollection - Complete data collection operations

# ===================== GLOBAL INSTANCES =====================
data_collection = DataCollection()
news_analysis = NewsAnalysis()
market_analysis = MarketAnalysis()

# Global instances already provide all needed functionality


    # ===================== EXISTING CODE CONTINUES BELOW =====================
    # All existing functions and classes remain here temporarily during reorganization
    # They will be systematically moved into the appropriate classes above





    # Thiết lập logging thống nhất
    def setup_unified_logging(log_level=logging.INFO):
        """Setup unified logging for the entire application"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('crypto_analysis.log', encoding='utf-8')
            ]
        )
        return logging.getLogger(__name__)

    def get_logger(name):
        """Get logger instance"""
        return logging.getLogger(name)

    # Logger chính
    logger = setup_unified_logging()

    # Configuration settings
    def Settings:
        def __init__(self):
            self.DATA_DIR = os.path.join(os.getcwd(), "data")
            self.TIMEZONE = "Asia/Ho_Chi_Minh"
            self.DEFAULT_QUOTE_BINANCE = "USDT"
            self.HISTORY_REQUIREMENTS = {
                "5m": 7, "15m": 30, "30m": 60, "1h": 90, "4h": 180, "1d": 365
            }

    
    HISTORY_REQUIREMENTS = settings.HISTORY_REQUIREMENTS
    TIMEZONE = settings.TIMEZONE
    DATA_DIR = settings.DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)

    # Column list chuẩn của Binance
    _KLINES_COLS = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
    ]

    # Utilities/helper functions
    def timeframe_to_minutes(tf_str: str):
        """Chuyển chuỗi khung thời gian (vd '5m','1h','1d') thành số phút."""
        units = {'m': 1, 'h': 60, 'd': 24*60, 'w': 7*24*60, 'M': 30*24*60}
        num = ''.join([ch for ch in tf_str if ch.isdigit()]) or '1'
        unit = ''.join([ch for ch in tf_str if ch.isalpha()]) or 'm'
        return int(num) * units.get(unit, 1)    # Removed: ms_to_local_time_str function moved to DataCollection class

    # Moved to DataCollection.fetch_klines() method
    def fetch_klines(symbol: str, tf: str, start_time: int) -> pd.DataFrame:
        """Wrapper function - redirects to DataCollection.fetch_klines()"""
        data_collection = DataCollection()
        return data_collection.fetch_klines(symbol, tf, start_time)    # Removed: ensure_data_dir function moved to DataCollection class

    # Moved to DataCollection.fetch_new_data_from_binance() method


    def save_compressed(data, file_path):
        """
        Save data to a compressed .pkl.gz file.
        If data is a pandas DataFrame, it will be pickled and gzipped.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Use pickle with gzip compression for lossless compression
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_compressed(file_path):
        """    Load data from a compressed .pkl.gz file.
        Returns the original object (e.g., DataFrame).
        """
        with gzip.open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    # Khởi tạo VADER sentiment analyzer
    try:
        analyzer = SentimentIntensityAnalyzer()
    except Exception as e:
        logger = get_logger('news_manager')
        logger.error(f"Lỗi khởi tạo Sentiment Analyzer: {e}")
        analyzer = None


    # 8. reinforcement module - Môi trường và huấn luyện RL
    try:
    except ImportError:
        gym = None

    # Duplicate TradingEnv removed - consolidated to single definition below

    # Thư mục gốc của dự án
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(PROJECT_DIR)

    # Thư mục chứa dữ liệu lịch sử (.pkl.gz)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Cấu trúc đường dẫn lưu dữ liệu lịch sử cho symbol theo từng khung thời gian
    DATA_PATH = os.path.join(DATA_DIR, "{symbol}_{timeframe}.pkl.gz")

    # Cấu hình API Binance
    BINANCE_BASE_URL = "https://api.binance.com"
    DEFAULT_QUOTE_BINANCE = "USDT"

    # symbol mặc định khi khởi động chương trình
    DEFAULT_symbol = "BTC"

    # Múi giờ mặc định của chương trình (Việt Nam UTC+7)
    TIMEZONE = pytz.timezone("Asia/Ho_Chi_Minh")

    # Số ngày lịch sử tối thiểu cần tải cho từng loại khung thời gian
    HISTORY_REQUIREMENTS = {
        "5m": 180, "15m": 180, "30m": 180, "1h": 180, "4h": 180,
        "6h": 365, "12h": 365, "1d": 365, "3d": 365,
        "1w": 3650,  # ~10 năm cho dữ liệu dài hạn
        "1M": 3650   # ~10 năm cho dữ liệu dài hạn
    }

    # Thời gian lưu trữ tối đa dữ liệu cũ (ngày) – dữ liệu cũ hơn sẽ bị cắt bớt
    RETENTION_DAYS = {
        "5m": 180, "15m": 180, "30m": 180, "1h": 180, "4h": 180,
        "6h": 365, "12h": 365, "1d": 365, "2d": 365,
        "1w": 3650,
        "1M": 3650
    }

    # Bảng quy đổi khung thời gian ra giây (sử dụng cho tính chỉ báo theo cửa sổ thời gian)
    TIMEFRAME_TO_SECONDS = {
        "5m": 5 * 60,
        "15m": 15 * 60,
        "30m": 30 * 60,
        "1h": 60 * 60,
        "4h": 4 * 60 * 60,
        "6h": 6 * 60 * 60,
        "12h": 12 * 60 * 60,
        "1d": 24 * 60 * 60,
        "1w": 7 * 24 * 60 * 60,
        "1M": 30 * 24 * 60 * 60
    }

    # Danh sách khung thời gian sử dụng cho mô hình ensemble
    TIMEFRAMES = [
        "5m", "15m", "30m", "1h", "4h"
    ]

    # Logging – thư mục log
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_FILE = os.path.join(LOG_DIR, "app.log")

    # API key cho dịch vụ tin tức (News API) – để trống nếu không dùng
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    # Maximum proportion of RAM được phép sử dụng cho training (0<n<1)
    MAX_RAM_UTIL = 0.95
    # Tỷ lệ core CPU dùng cho training (0<n<=1)
    CPU_UTIL_RATIO = 0.95
    # Số luồng tối đa dùng cho đào tạo (VD: số CPU nhân với tỷ lệ CPU_UTIL_RATIO)
    MAX_WORKERS = int(os.cpu_count() * CPU_UTIL_RATIO)
    # Mức % CPU cho phép sử dụng (trên 100% nếu có nhiều core)
    MAX_CPU_PERCENT = int(CPU_UTIL_RATIO * 100)

    # GPU Configuration
    MAX_GPU_LOAD = 0.80
    MIN_FREE_GPU_MEM = 0.15

    # Model Training Configuration
    OPTUNA_TRIALS = 30
    CV_FOLDS = 3
    MIN_SAMPLES_PER_CLASS = 5
    MIN_TOTAL_SAMPLES = 20

    # File: AI_Crypto_Project/ai_optimizer/resource_optimizer.py

    # Global training status
    training_status: Dict[str, Any] = {
        "train_progress": 0.0,
        "train_status": "",
        "train_error": None,
        "predicting": False,
        "predict_done": 0,
        "predict_total_steps": 0
    }


    def ZaloSender:
        """
        Module thuần logic để:
        - Tạo PKCE codes
            - Sinh QR code / URL để user scan
        - Hoán đổi code → access_token
        - Quản lý contact (load, save, add, remove)
        - Gửi tin nhắn text
        Hoàn toàn không phụ thuộc Streamlit.
        """
        
        def __init__(self):
            self.app_id       = os.getenv("ZALO_APP_ID")
            self.secret_key   = os.getenv("ZALO_SECRET_KEY")
            self.redirect_uri = os.getenv("ZALO_REDIRECT_URI")
            self.access_token = os.getenv("ZALO_ACCESS_TOKEN")  # có thể None

            # Khóa mã hóa dữ liệu contacts
            self.enc_key  = self._get_encryption_key()
            self.contacts = self.load_contacts()

        def best_by(self, metric: str = 'f1') -> pd.Series:
            idx = self.metrics[metric].idxmax()
            return self.metrics.loc[idx, ['timeframe','model', metric]]

        def top_n_timeframes(self, n: int = 3, metric: str = 'f1') -> List[str]:
            df = self.metrics.groupby('timeframe')[metric] \
                            .mean() \
                            .sort_values(ascending=False)
            return df.head(n).index.tolist()

        def top_n_models(self, n: int = 3, metric: str = 'f1') -> List[str]:
            df = self.metrics.groupby('model')[metric] \
                            .mean() \
                            .sort_values(ascending=False)
            return df.head(n).index.tolist()

        def recommend(self, metric: str = 'f1') -> Dict[str, Any]:
            best = self.best_by(metric)
            return {
                'best_timeframe': best['timeframe'],
                'best_model':     best['model'],
                'best_score':     float(best[metric]),
                'top_timeframes': self.top_n_timeframes(3, metric),
                'top_models':     self.top_n_models(3, metric)
            }

        def filter_metrics(self,
                        timeframes: List[str] = None,
                        models:     List[str] = None
                        ) -> pd.DataFrame:
            df = self.metrics
            if timeframes is not None:
                df = df[df['timeframe'].isin(timeframes)]
            if models is not None:
                df = df[df['model'].isin(models)]
            return df.reset_index(drop=True)

        # ----- New methods for cross-timeframe comparison -----

        @staticmethod
        def _aggregate_predictions_low_to_high(df_low: pd.DataFrame,
                                            tf_high: str,
                                            pred_col: str = 'prediction',
                                            method: str = 'majority'
                                            ) -> pd.Series:
            """
            Resample df_low[pred_col] from its low timeframe up to tf_high.
            method: 'majority' or 'mean_thresh' or 'max'
            Returns a Series indexed by the tf_high timestamps.
            """
            ser = df_low.set_index('timestamp')[pred_col]
            if method == 'majority':
                agg_fn = lambda x: 1 if x.sum() > len(x)/2 else 0
            elif method == 'mean_thresh':
                agg_fn = lambda x: 1 if x.mean() >= 0.5 else 0
            elif method == 'max':
                agg_fn = lambda x: int(x.max() > 0)
            else:
                raise ValueError(f"Unknown method {method}")

            return ser.resample(tf_high).apply(agg_fn)

        def compare_timeframes(self,
                            df_low: pd.DataFrame,
                            df_high: pd.DataFrame,
                            tf_high: str,
                            pred_col: str = 'prediction',
                            label_col: str = 'label',
                            method:    str = 'majority'
                            ) -> Dict[str, float]:
            """
            So sánh accuracy giữa dự đoán khung thấp và label khung cao.
            df_low: DataFrame với ['timestamp', pred_col]
            df_high: DataFrame với ['timestamp', label_col]
            tf_high: e.g. '15T', '30T', '1H'
            method: cách aggregate ('majority','mean_thresh','max')
            Returns dict of { 'accuracy', 'precision', 'recall', 'f1' }.
            """
            # 1) Aggregate low‑frame predictions to high frame
            agg_pred = self._aggregate_predictions_low_to_high(
                df_low, tf_high, pred_col, method
            )

            # 2) Align and extract labels
            labels = (
                df_high.set_index('timestamp')[label_col]
                    .resample(tf_high)
                    .first()
            )

            # 3) Combine and drop any NA
            df_cmp = pd.concat([agg_pred, labels], axis=1)
            df_cmp.columns = ['pred','true']
            df_cmp = df_cmp.dropna()

            # 4) Compute metrics
            y_true = df_cmp['true'].astype(int)
            y_pred = df_cmp['pred'].astype(int)

            return {
                'accuracy':  accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall':    recall_score(y_true, y_pred, zero_division=0),
                'f1':        f1_score(y_true, y_pred)
            }






# Thời gian chờ giữa các lần huấn luyện (tính bằng giây) - 7 ngày = 604800 giây
    ONE_WEEK_SECONDS = 7 * 24 * 60 * 60

    def _schedule_next_run(interval_seconds: int):
        """Đặt lịch gọi lại hàm huấn luyện sau interval_seconds giây."""
        timer = threading.Timer(interval_seconds, _run_weekly_job)
        timer.daemon = True  # đặt luồng nền để không ngăn chương trình thoát
        timer.start()

    def _run_weekly_job():
        """Hàm công việc chạy hàng tuần: gọi vòng lặp huấn luyện và lên lịch lần tiếp theo."""
        try:
            trainer_loop.run_training_loop()
        except Exception as e:
            # Ghi log lỗi nếu có, sử dụng log_message từ trainer_loop nếu cần
            try:
                trainer_loop.log_message(f"Lỗi khi huấn luyện tự động: {e}")
            except:
                print(f"Lỗi khi huấn luyện tự động: {e}")
        # Sau khi hoàn thành, lên lịch chạy lần tiếp theo sau 1 tuần
        _schedule_next_run(ONE_WEEK_SECONDS)

    def start_weekly_training():
        """
        Bắt đầu quá trình huấn luyện tự động hàng tuần.
        Gọi hàm này sẽ kích hoạt việc huấn luyện ngay lập tức một lần, 
        sau đó tự động lên lịch lặp lại mỗi tuần.
        """
        print("Khởi động cơ chế huấn luyện hàng tuần...")
        # Gọi ngay một lần huấn luyện đầu tiên
        _run_weekly_job()
        # (Lưu ý: _run_weekly_job sẽ tự lên lịch lần tiếp theo)



def infer_signal(symbol: str, timeframe: str):
    """
    Dự đoán tín hiệu Long/Short hiện tại cho symbol bằng mô hình AI đã huấn luyện.
    """    # Lấy dữ liệu mới nhất
    try:
        data = update_data(symbol, timeframe)
    except Exception as e:
        logger.error(f"Không cập nhật được dữ liệu cho {symbol}, {timeframe}: {e}")
        raise
    last_price = data["close"].iloc[-1]
    features_now = prepare_features(data, training=False)
    model_path = f"ai_models/trained_models/{symbol.upper()}_ALL_best.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Chưa có mô hình AI cho symbol này. Vui lòng huấn luyện trước.")
    try:
        best_model = joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file model tại {model_path}")
    except Exception as e:
        logger.error(f"Lỗi khi load model {model_path}: {e}")
        raise
    probas = best_model.predict_proba(features_now)
    short_prob = probas[0][0]
    long_prob = probas[0][1]
    signal = 1 if long_prob >= short_prob else 0
    confidence = max(long_prob, short_prob)
    # Xác định ATR để tính TP/SL
    atr_col = f"{timeframe}_ATR"
    atr_value = None
    if atr_col in features_now.columns:
        atr_value = features_now[atr_col].iloc[0]
    else:
        # Dùng khung nhỏ nhất nếu không có ATR của khung yêu cầu
        
        atr_value = features_now[f"{timeframe}_ATR"].iloc[0]
    tp_price = None
    sl_price = None
    if atr_value is not None and last_price is not None:
        if signal == 1:  # Long
            tp_price = last_price + 2 * atr_value
            sl_price = last_price - 1 * atr_value
        else:  # Short
            tp_price = last_price - 2 * atr_value
            sl_price = last_price + 1 * atr_value
    return {
        "signal": signal,
        "probability": confidence,
        "tp_price": tp_price,
        "sl_price": sl_price
    }

# Bổ sung hàm tự động


def run_training_loop(symbol=None, timeframe=None):
    """Chạy vòng lặp huấn luyện mô hình hàng tuần."""
    if symbol is None:
        symbol = DEFAULT_symbol
    logger.info(f"Auto-learning: Bắt đầu huấn luyện tự động hàng tuần cho {symbol}")
    
    # Use Training class method instead of standalone function
    training_instance = Training()
    training_instance.train_all_models(symbol)

def log_message(message):
    """Ghi thông điệp log của quá trình huấn luyện tự động."""
    logger.info(message)
# File: ai_models/reinforcement_ai.py


def TradingEnv(gym.Env):
    """
    A simple trading environment for reinforcement learning.
    State: [current_price, holding_flag]
      - holding_flag = 1 if we currently hold the asset, 0 if not.
    Actions: 0 = Hold, 1 = Buy, 2 = Sell.
    Reward: change in portfolio value after each action.
    Episode ends when we reach the end of the price series.
    """
    def __init__(self, prices: list):
        super(TradingEnv, self).__init__()
        self.prices = prices
        self.current_step = 0
        # Start with full cash (balance = 1 unit of capital) and 0 asset
        self.balance = 1.0
        self.holding = 0.0  # amount of asset held (if any)
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # {0: hold, 1: buy, 2: sell}
        # Observation: current price and whether we're holding asset (binary flag)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self):
        """
        Reset the environment to initial state.
        """
        self.current_step = 0
        self.balance = 1.0  # reset capital
        self.holding = 0.0  # no asset held
        # Starting observation: initial price and not holding
        initial_price = self.prices[0] if len(self.prices) > 0 else 0.0
        return np.array([initial_price, 0.0], dtype=np.float32)

    def step(self, action):
        """
        Execute one time step within the environment.
        Returns: obs, reward, done, info
        """
        # Current price at this step
        price = self.prices[self.current_step]
        done = False
        reward = 0.0
        # Apply action
        if action == 1:  # Buy action
            if self.holding == 0:  # if not already holding, buy with all balance
                self.holding = self.balance / price  # buy as many units as possible
                self.balance = 0.0
        elif action == 2:  # Sell action
            if self.holding > 0:  # if holding asset, sell all
                self.balance = self.holding * price
                self.holding = 0.0
        # Move to next step
        self.current_step += 1
        if self.current_step >= len(self.prices) - 1:
            # If we've reached the end of price data, episode will finish
            done = True
        # Next price (for observation and reward calculation)
        next_price = self.prices[self.current_step] if not done else self.prices[-1]
        # Calculate reward as change in total portfolio value
        total_value_before = self.balance + self.holding * price
        total_value_after = self.balance + self.holding * next_price
        reward = total_value_after - total_value_before
        # Observation for the next step: [next_price, holding_flag]
        obs = np.array([next_price, 1.0 if self.holding > 0 else 0.0], dtype=np.float32)
        return obs, reward, done, {}

# Note: train_trading_agent and train_reinforcement functions moved to Training class
# crypto/advanced_ai_modules/tp_sl_logic.py





# -*- coding: utf-8 -*-



# Thư viện snscrape để lấy tin Twitter (nếu chưa có thì bỏ qua)
sntwitter = None


# 1. sentiment module - Phân tích tâm lý thị trường




# 5. shap module - Tính toán SHAP values
def compute_shap_values(model, X):
    """
    Tính SHAP values cho mô hình và tập X trả về numpy array hoặc None.
    """
    try:
    except ImportError:
        os.system("pip install shap")
        try:
        except ImportError:
            print("Không thể cài đặt shap.")
            return None

    try:
        explainer = shap.Explainer(model, X)
        shap_res = explainer(X)
        return shap_res.values
    except Exception as e:
        print("Lỗi khi tính SHAP values:", e)
        return None

    


# 8. reinforcement module - Môi trường và huấn luyện RL
try:
except ImportError:
    gym = None

# Duplicate TradingEnv removed - using the complete definition at line 7033

# Note: train_reinforcement function moved to Training class


def AutoOptimizer:
    """
    Tự động chọn khung thời gian và mô hình tốt nhất dựa trên log metrics,
    và so sánh cross‑timeframe accuracy giữa khung thấp và khung cao.
    """

    def __init__(self, metrics: pd.DataFrame):
        """
        metrics: DataFrame với tối thiểu các cột
          ['timeframe','model','accuracy']
        Nếu thiếu precision/recall/f1 thì fallback = accuracy.
        """
        # ——— Bổ sung fallback cho các cột thiếu ———
        for col in ('precision','recall','f1'):
            if col not in metrics.columns:
                # gán tạm bằng accuracy để AutoOptimizer vẫn hoạt động
                metrics[col] = metrics['accuracy']

        self.metrics = metrics.copy()
        # Giờ thì chắc chắn có đủ bộ
        required = {'timeframe','model','accuracy','precision','recall','f1'}
        if not required.issubset(self.metrics.columns):
            missing = required - set(self.metrics.columns)
            raise ValueError(f"Missing columns even after fallback: {missing}")

    def best_by(self, metric: str = 'f1') -> pd.Series:
        idx = self.metrics[metric].idxmax()
        return self.metrics.loc[idx, ['timeframe','model', metric]]

    def top_n_timeframes(self, n: int = 3, metric: str = 'f1') -> List[str]:
        df = self.metrics.groupby('timeframe')[metric] \
                         .mean() \
                         .sort_values(ascending=False)
        return df.head(n).index.tolist()

    def top_n_models(self, n: int = 3, metric: str = 'f1') -> List[str]:
        df = self.metrics.groupby('model')[metric] \
                         .mean() \
                         .sort_values(ascending=False)
        return df.head(n).index.tolist()

    def recommend(self, metric: str = 'f1') -> Dict[str, Any]:
        best = self.best_by(metric)
        return {
            'best_timeframe': best['timeframe'],
            'best_model':     best['model'],
            'best_score':     float(best[metric]),
            'top_timeframes': self.top_n_timeframes(3, metric),
            'top_models':     self.top_n_models(3, metric)
        }

    def filter_metrics(self,
                       timeframes: List[str] = None,
                       models:     List[str] = None
                      ) -> pd.DataFrame:
        df = self.metrics
        if timeframes is not None:
            df = df[df['timeframe'].isin(timeframes)]
        if models is not None:
            df = df[df['model'].isin(models)]
        return df.reset_index(drop=True)

    # ----- New methods for cross-timeframe comparison -----

    @staticmethod
    def _aggregate_predictions_low_to_high(df_low: pd.DataFrame,
                                           tf_high: str,
                                           pred_col: str = 'prediction',
                                           method: str = 'majority'
                                          ) -> pd.Series:
        """
        Resample df_low[pred_col] from its low timeframe up to tf_high.
        method: 'majority' or 'mean_thresh' or 'max'
        Returns a Series indexed by the tf_high timestamps.
        """
        ser = df_low.set_index('timestamp')[pred_col]
        if method == 'majority':
            agg_fn = lambda x: 1 if x.sum() > len(x)/2 else 0
        elif method == 'mean_thresh':
            agg_fn = lambda x: 1 if x.mean() >= 0.5 else 0
        elif method == 'max':
            agg_fn = lambda x: int(x.max() > 0)
        else:
            raise ValueError(f"Unknown method {method}")

        return ser.resample(tf_high).apply(agg_fn)

    def compare_timeframes(self,
                           df_low: pd.DataFrame,
                           df_high: pd.DataFrame,
                           tf_high: str,
                           pred_col: str = 'prediction',
                           label_col: str = 'label',
                           method:    str = 'majority'
                          ) -> Dict[str, float]:
        """
        So sánh accuracy giữa dự đoán khung thấp và label khung cao.
        df_low: DataFrame với ['timestamp', pred_col]
        df_high: DataFrame với ['timestamp', label_col]
        tf_high: e.g. '15T', '30T', '1H'
        method: cách aggregate ('majority','mean_thresh','max')
        Returns dict of { 'accuracy', 'precision', 'recall', 'f1' }.
        """
        # 1) Aggregate low‑frame predictions to high frame
        agg_pred = self._aggregate_predictions_low_to_high(
            df_low, tf_high, pred_col, method
        )

        # 2) Align and extract labels
        labels = (
            df_high.set_index('timestamp')[label_col]
                   .resample(tf_high)
                   .first()
        )

        # 3) Combine and drop any NA
        df_cmp = pd.concat([agg_pred, labels], axis=1)
        df_cmp.columns = ['pred','true']
        df_cmp = df_cmp.dropna()

        # 4) Compute metrics
        y_true = df_cmp['true'].astype(int)
        y_pred = df_cmp['pred'].astype(int)

        return {
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'f1':        f1_score(y_true, y_pred)
        }
# File: AI_Crypto_Project/backtesting/realtime_backtester.py






def forecast_long_term(symbol=None):
    """
    Dự báo xu hướng dài hạn (tuần, tháng, quý) cho đồng symbol đưa ra.
    Sử dụng dữ liệu lịch sử khung 1W, 1M, 1d để huấn luyện mô hình và dự đoán xu hướng (Uptrend/Downtrend) cho kỳ kế tiếp.
    Trả về kết quả dự báo cho từng khung thời gian.
    """
    if symbol is None:
        symbol = settings.DEFAULT_symbol if hasattr(settings, "DEFAULT_symbol") else "BTCUSDT"
    timeframes = ['1d', '1W', '1M']  # các khung thời gian dài hạn cần dự báo
    forecast_results = {}  # kết quả dự báo cho mỗi khung

    for tf in timeframes:
        # Xác định interval phù hợp cho API Binance
        if tf == "1d":
            interval = "1d"  # 1W -> 1w
        elif  tf.endswith("W") or tf.endswith("w"):
            interval = "1w"
        elif tf == "1M":
            interval = "1M"
        else:
            interval = tf.lower()

        # Xác định khoảng thời gian lịch sử để lấy (5 năm gần nhất)
        years_back = 5
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=years_back * 365)
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)

        data_df = pd.DataFrame()
        try:
            # Gọi API Binance để lấy dữ liệu nến lịch sử cho khoảng thời gian
            url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}"
                   f"&interval={interval}&startTime={start_ts}&endTime={end_ts}")
            response = requests.get(url, timeout=60)
            candles = response.json() if response.status_code == 200 else []
            # Chuyển đổi kết quả API thành DataFrame pandas
            if len(candles) > 0:
                # Binance trả về: [open_time, open, high, low, close, volume, close_time, ...]
                cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                        'quote_vol', 'trades', 'taker_base_vol', 'taker_quote_vol', 'ignore']
                data_df = pd.DataFrame(candles, columns=cols)
                # Chuyển kiểu dữ liệu cho các cột số
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
                # Chuyển thời gian về datetime
                data_df['date'] = pd.to_datetime(data_df['open_time'], unit='ms')
                data_df.set_index('date', inplace=True)
            else:
                # Nếu không có dữ liệu từ Binance (có thể do ký hiệu không tồn tại hoặc interval không hỗ trợ)
                print(f"Không lấy được dữ liệu {tf} cho {symbol} từ Binance.")
        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu {tf} từ API: {e}")

        # Bỏ qua khung thời gian nếu không có dữ liệu
        if data_df.empty or len(data_df) < 10:
            forecast_results[tf] = "No data"
            continue

        # Tính toán các đặc trưng kỹ thuật cho dữ liệu khung thời gian này
        try:
            features_df = feature_engineering.prepare_features(data_df.copy())
        except Exception as e:
            # Nếu xảy ra lỗi trong quá trình tính đặc trưng, sử dụng dữ liệu gốc
            features_df = data_df.copy()
            print(f"Lỗi prepare_features cho {tf}: {e}")

        # Tạo nhãn mục tiêu: 1 nếu giá tăng ở kỳ kế tiếp, 0 nếu giảm
        # Sử dụng giá đóng cửa để xác định xu hướng
        features_df['target'] = 0
        if 'close' in features_df.columns:
            close_series = features_df['close']
        else:
            close_series = data_df['close']
            features_df['close'] = close_series
        # Xác định nhãn target dựa trên giá đóng cửa
        features_df['target'] = (close_series.shift(-1) > close_series).astype(int)
        features_df = features_df[:-1]  # loại bỏ hàng cuối (không có nhãn target vì không có giá tương lai)

        # Tách tập dữ liệu thành X (đặc trưng) và y (nhãn)
        if 'target' in features_df.columns:
            y = features_df['target']
            X = features_df.drop(columns=['target'])
        else:
            # Nếu prepare_features trả về X, y riêng thì xử lý khác (trường hợp này ít khả năng)
            X = features_df
            y = (close_series.shift(-1) > close_series).astype(int)[:-1]

        # Huấn luyện mô hình Random Forest cho khung thời gian này
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        try:
            model.fit(X, y)
        except Exception as e:
            # Nếu có lỗi trong huấn luyện (dữ liệu quá ít...), bỏ qua dự báo
            print(f"Lỗi huấn luyện mô hình cho {symbol} khung {tf}: {e}")
            forecast_results[tf] = "Training failed"
            continue

        # Dự đoán xu hướng kỳ tiếp theo dựa trên dữ liệu cuối cùng hiện có
        last_features = X.iloc[[-1]]  # đặc trưng của kỳ gần nhất
        pred = model.predict(last_features)[0]  # 0 hoặc 1
        # Xác suất dự đoán (độ tự tin của mô hình)
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(last_features)[0]
        if prob is not None:
            confidence = prob[1] * 100.0 if pred == 1 else prob[0] * 100.0
        else:
            confidence = 100.0  # nếu mô hình không có predict_proba, tạm cho 100%

        trend = "Uptrend" if pred == 1 else "Downtrend"
       # ─── Thêm dự báo TP/SL và giá hiện tại ───────────────────────
       # 1) Lấy giá hiện tại từ Binance
        try:
            ticker = requests.get(
               f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
               , timeout=5).json()
            current_price = float(ticker.get("price", close_series.iloc[-1]))
        except Exception:
            current_price = close_series.iloc[-1]

       # 2) Tính % biến động lịch sử trong chính khung tf này
        period_pct = close_series.pct_change().dropna()
        avg_pct = period_pct.mean() if len(period_pct) > 0 else 0
        if len(period_pct) > 0:
            avg_pct = period_pct.mean()             # trung bình return từng kỳ (ví dụ: daily return cho tf 1d)
        else:
            avg_pct = 0

       # 3) Dự kiến giá kỳ tiếp theo dựa trên biến động trung bình lịch sử
        forecast_price = current_price * (1 + + avg_pct)
        tp_price = forecast_price
       # 4) Xác định TP và SL (ví dụ: TP = forecast_price, SL cách current_price đối xứng với TP)
        distance = tp_price - current_price
        sl_price = current_price - (forecast_price - current_price)
        if sl_price < 0:
            sl_price = 0
       # 5) Ghi vào kết quả
        forecast_results[tf] = {
            "xu hướng":         trend,
            "độ tin cậy":       f"{confidence:.1f}%",
            "tỉ lệ trung bình": f"{avg_pct*100:.2f}%",
            "giá dự báo":       f"{forecast_price:.2f}",
            "giá chốt lời":     f"{tp_price:.2f}",
            "giá cắt lỗ":       f"{sl_price:.2f}",
            "giá hiện tại":     f"{current_price:.2f}"
        }
        print(f"[{symbol} - {tf}] Dự báo: {trend} với độ tự tin ~{confidence:.1f}%")

    return forecast_results


# Danh sách các sự kiện lịch sử lớn (ngày tháng và mô tả)



load_dotenv()

def ZaloSender:
    """
    Module thuần logic để:
      - Tạo PKCE codes
      - Sinh QR code / URL để user scan
      - Hoán đổi code → access_token
      - Quản lý contact (load, save, add, remove)
      - Gửi tin nhắn text
    Hoàn toàn không phụ thuộc Streamlit.
    """

    def __init__(self):
        self.app_id       = os.getenv("ZALO_APP_ID")
        self.secret_key   = os.getenv("ZALO_SECRET_KEY")
        self.redirect_uri = os.getenv("ZALO_REDIRECT_URI")
        self.access_token = os.getenv("ZALO_ACCESS_TOKEN")  # có thể None

        # Khóa mã hóa dữ liệu contacts
        self.enc_key  = self._get_encryption_key()
        self.contacts = self.load_contacts()

    # — PKCE helpers — 
    def generate_pkce(self):
        """
        Trả về tuple (code_verifier, code_challenge, state)
        để build URL OAuth PKCE.
        """
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip("=")
        state = secrets.token_urlsafe(16)
        return code_verifier, code_challenge, state

    def build_auth_url(self, code_challenge: str, state: str) -> str:
        """
        Trả về URL để user scan QR hoặc click.
        """
        return (
            f"https://oauth.zaloapp.com/v4/oa/permission"
            f"?app_id={self.app_id}"
            f"&redirect_uri={self.redirect_uri}"
            f"&code_challenge={code_challenge}"
            f"&code_challenge_method=S256"
            f"&state={state}"
        )

    def get_qr_code_bytes(self, auth_url: str) -> bytes:
        """
        Sinh QR code PNG bytes từ auth_url.
        """
        img = qrcode.make(auth_url)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    def exchange_code_for_token(self, auth_code: str, code_verifier: str) -> Tuple[bool, str]:
        """
        Gửi request lấy access_token.
        Trả về (success, message). Nếu success=True thì
        self.access_token được cập nhật.
        """
        token_url = "https://oauth.zaloapp.com/v4/oa/access_token"
        data = {
            "app_id":        self.app_id,
            "grant_type":    "authorization_code",
            "code":          auth_code,
            "redirect_uri":  self.redirect_uri,
            "code_verifier": code_verifier,
            "app_secret":    self.secret_key
        }
        resp = requests.post(token_url, data=data).json()
        if "access_token" in resp:
            self.access_token = resp["access_token"]
            # Ghi vào .env để reuse
            set_key(".env", "ZALO_ACCESS_TOKEN", self.access_token)
            return True, "Authenticated successfully"
        else:
            return False, resp.get("error_description", "Unknown error")

    # — Encryption & contacts persistence —
    def _get_encryption_key(self) -> bytes:
        key_path = "data/key.enc"
        os.makedirs("data", exist_ok=True)
        if os.path.exists(key_path):
            return open(key_path, "rb").read()
        key = Fernet.generate_key()
        open(key_path, "wb").write(key)
        return key

    def load_contacts(self) -> list:
        """
        Đọc file data/contacts.enc (nếu có),
        giải mã và trả về list of dict.
        """
        path = "data/contacts.enc"
        if not os.path.exists(path):
            return []
        encrypted = open(path, "rb").read()
        fernet = Fernet(self.enc_key)
        data    = fernet.decrypt(encrypted)
        contacts = json.loads(data.decode("utf-8"))
        # đảm bảo có key last_sent
        for c in contacts:
            c.setdefault("last_sent", None)
        return contacts

    def save_contacts(self):
        """
        Mã hóa và ghi lại toàn bộ self.contacts vào file.
        """
        fernet   = Fernet(self.enc_key)
        data     = json.dumps(self.contacts).encode("utf-8")
        encrypted = fernet.encrypt(data)
        open("data/contacts.enc", "wb").write(encrypted)

    def add_contact(self, phone: str, name: str = None) -> Tuple[bool, str]:
        """
        Thêm 1 contact mới rồi save.
        """
        self.contacts.append({"phone": phone, "name": name, "last_sent": None})
        self.save_contacts()
        return True, "Contact added successfully"

    def remove_contact(self, phone: str) -> Tuple[bool, str]:
        """
        Xóa contact theo phone rồi save.
        """
        before = len(self.contacts)
        self.contacts = [c for c in self.contacts if c["phone"] != phone]
        if len(self.contacts) == before:
            return False, "Contact not found"
        self.save_contacts()
        return True, "Contact removed successfully"

    # — Enhanced Messaging Functions —
    def send_text_message(self, user_id: str, message: str) -> Tuple[bool, str]:
        """
        Gửi tin nhắn text qua Zalo OA API với retry logic và better error handling.
        """
        if not self.access_token:
            return False, "No access token. Please authenticate first."
        
        url = "https://openapi.zalo.me/v2.0/oa/message"
        headers = {
            "access_token": self.access_token,
            "Content-Type": "application/json"
        }
        payload = {
            "recipient": {"user_id": user_id},
            "message": {"text": message}
        }
        
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=10)
                r.raise_for_status()
                response = r.json()
                
                if response.get("error") == 0:
                    # Update last_sent for this contact
                    self._update_contact_last_sent(user_id)
                    return True, "Message sent successfully"
                else:
                    error_msg = response.get("message", "Unknown error")
                    if attempt == max_retries - 1:
                        return False, f"Zalo API error: {error_msg}"
                    # Wait before retry
                    time.sleep(2 ** attempt)
                    
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return False, f"Network error: {str(e)}"
                time.sleep(2 ** attempt)
                
        return False, "Failed after all retry attempts"

    def send_rich_message(self, user_id: str, title: str, subtitle: str, 
                         elements: list = None, buttons: list = None) -> Tuple[bool, str]:
        """
        Gửi tin nhắn phong phú với template.
        """
        if not self.access_token:
            return False, "No access token. Please authenticate first."
            
        url = "https://openapi.zalo.me/v2.0/oa/message"
        headers = {
            "access_token": self.access_token,
            "Content-Type": "application/json"
        }
        
        # Build rich message payload
        attachment = {
            "type": "template",
            "payload": {
                "template_type": "list",
                "elements": elements or [
                    {
                        "title": title,
                        "subtitle": subtitle,
                        "image_url": "https://via.placeholder.com/150x150.png"
                    }
                ]
            }
        }
        
        if buttons:
            attachment["payload"]["buttons"] = buttons
            
        payload = {
            "recipient": {"user_id": user_id},
            "message": {"attachment": attachment}
        }
        
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=10)
            r.raise_for_status()
            response = r.json()
            
            if response.get("error") == 0:
                self._update_contact_last_sent(user_id)
                return True, "Rich message sent successfully"
            else:
                return False, f"Zalo API error: {response.get('message', 'Unknown error')}"
                
        except Exception as e:
            return False, f"Error sending rich message: {str(e)}"

    def broadcast_message(self, message: str, target_group: str = "all") -> dict:
        """
        Gửi tin nhắn broadcast đến nhiều contacts.
        target_group: "all", "recent" (sent in last 7 days), "frequent" (sent >5 times)
        """
        results = {
            "success": 0,
            "failed": 0,
            "errors": [],
            "recipients": []
        }
        
        # Filter contacts based on target group
        target_contacts = self._filter_contacts_by_group(target_group)
        
        for contact in target_contacts:
            user_id = contact.get("phone") or contact.get("user_id")
            if not user_id:
                continue
                
            success, msg = self.send_text_message(user_id, message)
            if success:
                results["success"] += 1
                results["recipients"].append(contact.get("name", user_id))
            else:
                results["failed"] += 1
                results["errors"].append(f"{contact.get('name', user_id)}: {msg}")
                
            # Rate limiting - avoid spam
            time.sleep(1)
            
        return results

    def _filter_contacts_by_group(self, target_group: str) -> list:
        """
        Lọc contacts theo nhóm target.
        """
        if target_group == "all":
            return self.contacts
        elif target_group == "recent":
            # Contacts sent to in last 7 days
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            return [c for c in self.contacts 
                   if c.get("last_sent") and c["last_sent"] > cutoff]
        elif target_group == "frequent":
            # Contacts with send_count > 5
            return [c for c in self.contacts 
                   if c.get("send_count", 0) > 5]
        else:
            return self.contacts

    def _update_contact_last_sent(self, user_id: str):
        """
        Cập nhật timestamp và count cho contact vừa gửi tin nhắn.
        """
        for contact in self.contacts:
            if contact.get("phone") == user_id or contact.get("user_id") == user_id:
                contact["last_sent"] = datetime.now(timezone.utc).isoformat()
                contact["send_count"] = contact.get("send_count", 0) + 1
                break
        self.save_contacts()

    def get_contact_stats(self) -> dict:
        """
        Thống kê về contacts và hoạt động gửi tin nhắn.
        """
        total = len(self.contacts)
        active_7d = len([c for c in self.contacts 
                        if c.get("last_sent") and 
                        c["last_sent"] > (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()])
        
        frequent = len([c for c in self.contacts if c.get("send_count", 0) > 5])
        
        return {
            "total_contacts": total,
            "active_last_7_days": active_7d,
            "frequent_contacts": frequent,
            "last_broadcast": self._get_last_broadcast_time()
        }

    def _get_last_broadcast_time(self) -> str:
        """
        Lấy thời gian broadcast gần nhất.
        """
        if not self.contacts:
            return "Never"
        
        last_times = [c.get("last_sent") for c in self.contacts if c.get("last_sent")]
        if not last_times:
            return "Never"
            
        latest = max(last_times)
        dt = datetime.fromisoformat(latest.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def compose_message_from_predictions(self, predictions: list, include_chart_link: bool = True) -> str:
        """
        Tạo nội dung tin nhắn tự động từ kết quả dự đoán với enhanced formatting.
        - predictions: list of dict, mỗi dict có các key:
            'symbol'      (str)   – mã coin
            'action'      (str)   – BUY/SELL hoặc tương tự
            'entry_price' (float) – giá vào lệnh
            'take_profit' (float) – giá chốt lời
            'stop_loss'   (float) – giá cắt lỗ
            'confidence'  (float) – hệ số tin cậy (0–1)
            'timestamp'   (float) – UNIX timestamp (giây)
            'timeframe'   (str)   – khung thời gian
        """
        if not predictions:
            return "🤖 Không có tín hiệu mới hiện tại."
            
        # Header
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "🚀 CRYPTO AI SIGNALS 🚀",
            f"⏰ {timestamp}",
            "─" * 30
        ]
        
        # Group predictions by action
        buy_signals = [p for p in predictions if p.get('action', '').upper() in ['BUY', 'LONG']]
        sell_signals = [p for p in predictions if p.get('action', '').upper() in ['SELL', 'SHORT']]
        
        # Process buy signals
        if buy_signals:
            lines.append("📈 BUY SIGNALS:")
            for pr in buy_signals:
                signal_text = self._format_single_prediction(pr, include_chart_link)
                lines.append(signal_text)
            lines.append("")
            
        # Process sell signals  
        if sell_signals:
            lines.append("📉 SELL SIGNALS:")
            for pr in sell_signals:
                signal_text = self._format_single_prediction(pr, include_chart_link)
                lines.append(signal_text)
            lines.append("")
            
        # Footer
        lines.extend([
            "─" * 30,
            "⚠️ Đây chỉ là dự đoán AI, không phải lời khuyên tài chính.",
            "📊 DYOR - Tự nghiên cứu trước khi đầu tư!"
        ])
        
        return "\n".join(lines)

    def _format_single_prediction(self, pr: dict, include_chart_link: bool = True) -> str:
        """
        Format một prediction thành text đẹp.
        """
        symbol = pr.get('symbol', 'N/A')
        action = pr.get('action', '').upper()
        entry = pr.get('entry_price')
        tp = pr.get('take_profit')
        sl = pr.get('stop_loss')
        conf = pr.get('confidence')
        tf = pr.get('timeframe', '1h')
        
        # Action emoji
        emoji = "🟢" if action in ['BUY', 'LONG'] else "🔴"
        
        # Build signal text
        parts = [f"{emoji} {symbol} ({tf})"]
        
        if entry is not None:
            parts.append(f"Entry: ${entry:.4f}")
        if tp is not None:
            parts.append(f"TP: ${tp:.4f}")
        if sl is not None:
            parts.append(f"SL: ${sl:.4f}")
        if conf is not None:
            conf_emoji = "🔥" if conf > 0.8 else "⚡" if conf > 0.6 else "💡"
            parts.append(f"{conf_emoji} {conf:.1%}")
            
        signal_text = " | ".join(parts)
        
        # Add chart link if requested
        if include_chart_link:
            chart_url = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}"
            signal_text += f"\n📊 Chart: {chart_url}"
            
        return signal_text

    def create_alert_from_price_change(self, symbol: str, current_price: float, 
                                     previous_price: float, timeframe: str = "1h") -> str:
        """
        Tạo alert khi giá thay đổi đáng kể.
        """
        change_pct = ((current_price - previous_price) / previous_price) * 100
        
        if abs(change_pct) < 2:  # Chỉ alert khi thay đổi > 2%
            return ""
            
        emoji = "🚀" if change_pct > 0 else "💥"
        direction = "tăng" if change_pct > 0 else "giảm"
        
        alert = [
            f"{emoji} PRICE ALERT {emoji}",
            f"📊 {symbol}: {direction} {abs(change_pct):.2f}%",
            f"💰 Giá hiện tại: ${current_price:.4f}",
            f"⏰ Khung thời gian: {timeframe}",
            f"🕐 {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        ]
        
        return "\n".join(alert)

    def validate_access_token(self) -> Tuple[bool, str]:
        """
        Kiểm tra tính hợp lệ của access token hiện tại.
        """
        if not self.access_token:
            return False, "No access token available"
            
        url = "https://openapi.zalo.me/v2.0/oa/getoa"
        headers = {"access_token": self.access_token}
        
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            response = r.json()
            
            if response.get("error") == 0:
                oa_info = response.get("data", {})
                return True, f"Token valid for OA: {oa_info.get('name', 'Unknown')}"
            else:
                return False, f"Invalid token: {response.get('message', 'Unknown error')}"
                
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def get_oa_profile(self) -> dict:
        """
        Lấy thông tin profile của Official Account.
        """
        if not self.access_token:
            return {"error": "No access token"}
            
        url = "https://openapi.zalo.me/v2.0/oa/getoa"
        headers = {"access_token": self.access_token}
        
        try:
            r = requests.get(url, headers=headers, timeout=10)
            response = r.json()
            return response.get("data", {}) if response.get("error") == 0 else {"error": response.get("message")}
        except Exception as e:
            return {"error": str(e)}


    def SymmetricEncryption:
        """Lớp quản lý mã hóa đối xứng sử dụng AES"""

        @staticmethod
        def generate_key():
            """Tạo khóa AES ngẫu nhiên 32 byte (AES-256)"""
            return os.urandom(32)

        @staticmethod
        def generate_key_from_password(password, salt=None):
            """
            Tạo khóa từ mật khẩu sử dụng PBKDF2
            
            Args:
                password (str): Mật khẩu người dùng
                salt (bytes, optional): Salt ngẫu nhiên. Nếu None, tạo mới
                
            Returns:
                tuple: (key, salt) - Khóa và salt đã dùng
            """
            if salt is None:
                salt = os.urandom(16)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            key = kdf.derive(password.encode())
            return key, salt

        @staticmethod
        def encrypt(plaintext, key):
            """
            Mã hóa dữ liệu sử dụng AES trong chế độ CBC
            
            Args:
                plaintext (str): Dữ liệu cần mã hóa
                key (bytes): Khóa mã hóa 32 byte
                
            Returns:
                tuple: (iv, ciphertext) - IV và dữ liệu đã mã hóa
            """
            if isinstance(plaintext, str):
                plaintext = plaintext.encode()
                
            iv = os.urandom(16)
            
            # Thêm padding
            padder = padding.PKCS7(algorithms.AES.block_size).padder()
            padded_data = padder.update(plaintext) + padder.finalize()
            
            # Mã hóa
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            return iv, ciphertext

        @staticmethod
        def decrypt(iv, ciphertext, key):
            """
            Giải mã dữ liệu AES
            
            Args:
                iv (bytes): Vector khởi tạo
                ciphertext (bytes): Dữ liệu đã mã hóa
                key (bytes): Khóa mã hóa 32 byte
                
            Returns:
                bytes: Dữ liệu gốc
            """
            # Giải mã
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Loại bỏ padding
            unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
            plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
            
            return plaintext


    def AsymmetricEncryption:
        """Lớp quản lý mã hóa bất đối xứng sử dụng RSA"""

        @staticmethod
        def generate_key_pair(key_size=2048):
            """
            Tạo cặp khóa RSA
            
            Args:
                key_size (int): Kích thước khóa (mặc định: 2048 bit)
                
            Returns:
                tuple: (private_key, public_key) - Cặp khóa dạng PEM
            """
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            
            public_key = private_key.public_key()
            
            # Xuất khóa dạng PEM
            private_pem = private_key.private_bytes(
                encoding=Encoding.PEM,
                format=PrivateFormat.PKCS8,
                encryption_algorithm=NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=Encoding.PEM,
                format=PublicFormat.SubjectPublicKeyInfo
            )
            
            return private_pem, public_pem

        @staticmethod
        def encrypt(plaintext, public_key_pem):
            """
            Mã hóa dữ liệu với khóa công khai RSA
            
            Args:
                plaintext (str): Dữ liệu cần mã hóa
                public_key_pem (bytes): Khóa công khai định dạng PEM
                
            Returns:
                bytes: Dữ liệu đã mã hóa
            """
            if isinstance(plaintext, str):
                plaintext = plaintext.encode()
                
            public_key = load_pem_public_key(public_key_pem)
            
            ciphertext = public_key.encrypt(
                plaintext,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return ciphertext

        @staticmethod
        def decrypt(ciphertext, private_key_pem):
            """
            Giải mã dữ liệu với khóa riêng tư RSA
            
            Args:
                ciphertext (bytes): Dữ liệu đã mã hóa
                private_key_pem (bytes): Khóa riêng tư định dạng PEM
                
            Returns:
                bytes: Dữ liệu gốc
            """
            private_key = load_pem_private_key(
                private_key_pem,
                password=None
            )
            
            plaintext = private_key.decrypt(
                ciphertext,
                asym_padding.OAEP(
                    mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return plaintext

        @staticmethod
        def sign(message, private_key_pem):
            """
            Ký dữ liệu với khóa riêng tư RSA
            
            Args:
                message (str/bytes): Dữ liệu cần ký
                private_key_pem (bytes): Khóa riêng tư định dạng PEM
                
            Returns:
                bytes: Chữ ký số
            """
            if isinstance(message, str):
                message = message.encode()
                
            private_key = load_pem_private_key(
                private_key_pem,
                password=None
            )
            
            signature = private_key.sign(
                message,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return signature

        @staticmethod
        def verify(message, signature, public_key_pem):
            """
            Xác thực chữ ký với khóa công khai RSA
            
            Args:
                message (str/bytes): Dữ liệu đã ký
                signature (bytes): Chữ ký cần xác thực
                public_key_pem (bytes): Khóa công khai định dạng PEM
                
            Returns:
                bool: True nếu chữ ký hợp lệ, ngược lại False
            """
            if isinstance(message, str):
                message = message.encode()
                
            public_key = load_pem_public_key(public_key_pem)
            
            try:
                public_key.verify(
                    signature,
                    message,
                    asym_padding.PSS(
                        mgf=asym_padding.MGF1(hashes.SHA256()),
                        salt_length=asym_padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                return True
            except InvalidSignature:
                return False


    def HashUtils:
        """Các tiện ích tạo hàm băm"""

        @staticmethod
        def md5(data):
            """
            Tạo mã MD5 từ dữ liệu
            
            Args:
                data (str/bytes): Dữ liệu cần băm
                
            Returns:
                str: Chuỗi hex MD5
            """
            if isinstance(data, str):
                data = data.encode()
                
            hash_obj = hashlib.md5(data)
            return hash_obj.hexdigest()

        @staticmethod
        def sha1(data):
            """
            Tạo mã SHA-1 từ dữ liệu
            
            Args:
                data (str/bytes): Dữ liệu cần băm
                
            Returns:
                str: Chuỗi hex SHA-1
            """
            if isinstance(data, str):
                data = data.encode()
                
            hash_obj = hashlib.sha1(data)
            return hash_obj.hexdigest()

        @staticmethod
        def sha256(data):
            """
            Tạo mã SHA-256 từ dữ liệu
            
            Args:
                data (str/bytes): Dữ liệu cần băm
                
            Returns:
                str: Chuỗi hex SHA-256
            """
            if isinstance(data, str):
                data = data.encode()
                
            hash_obj = hashlib.sha256(data)
            return hash_obj.hexdigest()

        @staticmethod
        def sha512(data):
            """
            Tạo mã SHA-512 từ dữ liệu
            
            Args:
                data (str/bytes): Dữ liệu cần băm
                
            Returns:
                str: Chuỗi hex SHA-512
            """
            if isinstance(data, str):
                data = data.encode()
                
            hash_obj = hashlib.sha512(data)
            return hash_obj.hexdigest()

        @staticmethod
        def pbkdf2_hash(password, salt=None, iterations=100000):
            """
            Tạo hash bảo mật cho mật khẩu sử dụng PBKDF2
            
            Args:
                password (str): Mật khẩu cần hash
                salt (bytes, optional): Salt ngẫu nhiên. Nếu None, tạo mới
                iterations (int): Số vòng lặp (mặc định: 100000)
                
            Returns:
                tuple: (hash_hex, salt_hex) - Hash và salt dạng hex
            """
            if salt is None:
                salt = os.urandom(16)
            elif isinstance(salt, str):
                # Nếu salt là chuỗi hex, chuyển thành bytes
                salt = bytes.fromhex(salt)

            if isinstance(password, str):
                password = password.encode()
                
            hash_obj = hashlib.pbkdf2_hmac(
                'sha256',
                password,
                salt,
                iterations
            )
            
            return hash_obj.hex(), salt.hex()

        @staticmethod
        def verify_pbkdf2_hash(password, hash_hex, salt_hex, iterations=100000):
            """
            Xác thực mật khẩu với hash PBKDF2
            
            Args:
                password (str): Mật khẩu cần kiểm tra
                hash_hex (str): Hash dạng hex
                salt_hex (str): Salt dạng hex
                iterations (int): Số vòng lặp (mặc định: 100000)
                
            Returns:
                bool: True nếu mật khẩu khớp, ngược lại False
            """
            salt = bytes.fromhex(salt_hex)
            
            if isinstance(password, str):
                password = password.encode()
                
            hash_check = hashlib.pbkdf2_hmac(
                'sha256',
                password,
                salt,
                iterations
            ).hex()
            
            return hash_check == hash_hex


    def Base64Utils:
        """Các tiện ích mã hóa/giải mã Base64"""

        @staticmethod
        def encode(data):
            """
            Mã hóa dữ liệu sang Base64
            
            Args:
                data (str/bytes): Dữ liệu cần mã hóa
                
            Returns:
                str: Chuỗi Base64
            """
            if isinstance(data, str):
                data = data.encode()
                
            encoded = base64.b64encode(data)
            return encoded.decode()

        @staticmethod
        def decode(data):
            """
            Giải mã dữ liệu từ Base64
            
            Args:
                data (str): Chuỗi Base64
                
            Returns:
                bytes: Dữ liệu gốc
            """
            return base64.b64decode(data)

        @staticmethod
        def encode_file(file_path):
            """
            Mã hóa tệp sang Base64
            
            Args:
                file_path (str): Đường dẫn tệp
                
            Returns:
                str: Chuỗi Base64
            """
            with open(file_path, 'rb') as f:
                data = f.read()
            
            return Base64Utils.encode(data)

        @staticmethod
        def decode_to_file(base64_data, output_path):
            """
            Giải mã Base64 và lưu vào tệp
            
            Args:
                base64_data (str): Chuỗi Base64
                output_path (str): Đường dẫn tệp đích
                
            Returns:
                bool: True nếu thành công
            """
            decoded = Base64Utils.decode(base64_data)
            
            with open(output_path, 'wb') as f:
                f.write(decoded)
                
            return True


    def PasswordUtils:
        """Các tiện ích quản lý mật khẩu"""

        @staticmethod
        def generate_password(length=16, include_uppercase=True, include_lowercase=True, 
                            include_digits=True, include_symbols=True):
            """
            Tạo mật khẩu ngẫu nhiên mạnh
            
            Args:
                length (int): Độ dài mật khẩu (mặc định: 16)
                include_uppercase (bool): Bao gồm chữ hoa
                include_lowercase (bool): Bao gồm chữ thường
                include_digits (bool): Bao gồm chữ số
                include_symbols (bool): Bao gồm ký tự đặc biệt
                
            Returns:
                str: Mật khẩu ngẫu nhiên
            """
            charset = ""
            
            if include_uppercase:
                charset += string.ascii_uppercase
            if include_lowercase:
                charset += string.ascii_lowercase
            if include_digits:
                charset += string.digits
            if include_symbols:
                charset += string.punctuation
                
            if not charset:
                charset = string.ascii_letters + string.digits
                
            return ''.join(secrets.choice(charset) for _ in range(length))

        @staticmethod
        def check_password_strength(password):
            """
            Kiểm tra độ mạnh của mật khẩu
            
            Args:
                password (str): Mật khẩu cần kiểm tra
                
            Returns:
                tuple: (score, feedback) - Điểm (0-5) và phản hồi
            """
            score = 0
            feedback = []
            
            # Kiểm tra độ dài
            if len(password) < 8:
                feedback.append("Mật khẩu quá ngắn (tối thiểu 8 ký tự)")
            elif len(password) >= 12:
                score += 1
                
            # Kiểm tra ký tự viết hoa
            if re.search(r'[A-Z]', password):
                score += 1
            else:
                feedback.append("Nên bao gồm ít nhất một chữ cái viết hoa")
                
            # Kiểm tra ký tự viết thường
            if re.search(r'[a-z]', password):
                score += 1
            else:
                feedback.append("Nên bao gồm ít nhất một chữ cái viết thường")
                
            # Kiểm tra chữ số
            if re.search(r'\d', password):
                score += 1
            else:
                feedback.append("Nên bao gồm ít nhất một chữ số")
                
            # Kiểm tra ký tự đặc biệt
            if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                score += 1
            else:
                feedback.append("Nên bao gồm ít nhất một ký tự đặc biệt")
                
            # Phản hồi tổng quát
            if score < 2:
                strength = "Rất yếu"
            elif score < 3:
                strength = "Yếu"
            elif score < 4:
                strength = "Trung bình"
            elif score < 5:
                strength = "Mạnh"
            else:
                strength = "Rất mạnh"
                
            if not feedback:
                feedback.append(f"Mật khẩu {strength.lower()}")
                
            return score, (strength, feedback)


    # Các hàm tiện ích phổ biến
    def encrypt_file(input_file, output_file, password):
        """
        Mã hóa tệp sử dụng mật khẩu
        
        Args:
            input_file (str): Đường dẫn tệp đầu vào
            output_file (str): Đường dẫn tệp đầu ra
            password (str): Mật khẩu
            
        Returns:
            bool: True nếu thành công
        """
        try:
            # Đọc tệp
            with open(input_file, 'rb') as f:
                data = f.read()
                
            # Tạo khóa từ mật khẩu
            key, salt = SymmetricEncryption.generate_key_from_password(password)
            
            # Mã hóa dữ liệu
            iv, encrypted_data = SymmetricEncryption.encrypt(data, key)
            
            # Lưu salt, iv và dữ liệu đã mã hóa
            with open(output_file, 'wb') as f:
                f.write(len(salt).to_bytes(2, byteorder='big'))
                f.write(salt)
                f.write(len(iv).to_bytes(2, byteorder='big'))
                f.write(iv)
                f.write(encrypted_data)
                
            return True
        
        except Exception as e:
            print(f"Lỗi khi mã hóa tệp: {e}")
            return False


    def decrypt_file(input_file, output_file, password):
        """
        Giải mã tệp sử dụng mật khẩu
        
        Args:
            input_file (str): Đường dẫn tệp đã mã hóa
            output_file (str): Đường dẫn tệp đầu ra
            password (str): Mật khẩu
            
        Returns:
            bool: True nếu thành công
        """
        try:
            # Đọc tệp đã mã hóa
            with open(input_file, 'rb') as f:
                # Đọc salt
                salt_length = int.from_bytes(f.read(2), byteorder='big')
                salt = f.read(salt_length)
                
                # Đọc iv
                iv_length = int.from_bytes(f.read(2), byteorder='big')
                iv = f.read(iv_length)
                
                # Đọc dữ liệu đã mã hóa
                encrypted_data = f.read()
                
            # Tạo khóa từ mật khẩu và salt
            key, _ = SymmetricEncryption.generate_key_from_password(password, salt)
            
            # Giải mã dữ liệu
            decrypted_data = SymmetricEncryption.decrypt(iv, encrypted_data, key)
            
            # Lưu dữ liệu đã giải mã
            with open(output_file, 'wb') as f:
                f.write(decrypted_data)
                
            return True
        
        except Exception as e:
            print(f"Lỗi khi giải mã tệp: {e}")
            return False


    def quick_encrypt(text, password):
        """
        Mã hóa nhanh văn bản bằng mật khẩu
        
        Args:
            text (str): Văn bản cần mã hóa
            password (str): Mật khẩu
            
        Returns:
            str: Chuỗi mã hóa dạng Base64
        """
        # Tạo khóa từ mật khẩu
        key, salt = SymmetricEncryption.generate_key_from_password(password)
        
        # Mã hóa văn bản
        iv, encrypted_data = SymmetricEncryption.encrypt(text, key)
        
        # Đóng gói salt, iv và dữ liệu đã mã hóa
        result = salt + iv + encrypted_data
        
        # Mã hóa Base64
        return Base64Utils.encode(result)


    def quick_decrypt(encrypted_text, password):
        """
        Giải mã nhanh văn bản đã mã hóa
        
        Args:
            encrypted_text (str): Chuỗi đã mã hóa Base64
            password (str): Mật khẩu
            
        Returns:
            str: Văn bản gốc
        """
        try:
            # Giải mã Base64
            data = Base64Utils.decode(encrypted_text)
            
            # Trích xuất salt, iv và dữ liệu đã mã hóa
            salt = data[:16]
            iv = data[16:32]
            encrypted_data = data[32:]
            
            # Tạo khóa từ mật khẩu và salt
            key, _ = SymmetricEncryption.generate_key_from_password(password, salt)
            
            # Giải mã dữ liệu
            decrypted_data = SymmetricEncryption.decrypt(iv, encrypted_data, key)
            
            return decrypted_data.decode()
        
        except Exception as e:
            print(f"Lỗi khi giải mã: {e}")
            return None


    def generate_secure_token(length=32):
        """
        Tạo token bảo mật ngẫu nhiên
        
        Args:
            length (int): Độ dài token tính bằng byte
            
        Returns:
            str: Token dạng hex
        """
        return secrets.token_hex(length)


    # Thêm các tiện ích khác
    def FileIntegrityUtils:
        """Công cụ kiểm tra tính toàn vẹn của tệp"""
        
        @staticmethod
        def file_checksum(file_path, algorithm='sha256'):
            """
            Tính giá trị hash của tệp
            
            Args:
                file_path (str): Đường dẫn tệp
                algorithm (str): Thuật toán hash (md5, sha1, sha256, sha512)
                
            Returns:
                str: Giá trị hash dạng hex
            """
            hash_funcs = {
                'md5': hashlib.md5(),
                'sha1': hashlib.sha1(),
                'sha256': hashlib.sha256(),
                'sha512': hashlib.sha512()
            }
            
            if algorithm not in hash_funcs:
                raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}")
                
            hash_obj = hash_funcs[algorithm]
            
            with open(file_path, 'rb') as f:
                # Đọc và cập nhật từng khối dữ liệu
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_obj.update(chunk)
                    
            return hash_obj.hexdigest()
        
        @staticmethod
        def verify_file_integrity(file_path, expected_hash, algorithm='sha256'):
            """
            Xác thực tính toàn vẹn của tệp
            
            Args:
                file_path (str): Đường dẫn tệp
                expected_hash (str): Giá trị hash mong đợi
                algorithm (str): Thuật toán hash
                
            Returns:
                bool: True nếu hash khớp
            """
            actual_hash = FileIntegrityUtils.file_checksum(file_path, algorithm)
            return actual_hash.lower() == expected_hash.lower()
# app.py

# ===================== UNIFIED IMPORTS =====================
# Ignore Streamlit missing ScriptRunContext noise
warnings.filterwarnings(
    "ignore",
    message="Thread 'MainThread': missing ScriptRunContext",
    category=UserWarning,
    module="streamlit.runtime.scriptrunner_utils.script_run_context"
)
warnings.filterwarnings("ignore",message=".*tf.reset_default_graph.*",category=DeprecationWarning)


# ===================== UNIFIED LOGGING CONFIGURATION =====================
# Set environment variables to reduce thread spam and optimize performance
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'  
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only errors
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'false'

# Use unified logging configuration


# Setup unified logging
logger = setup_unified_logging(log_level=logging.INFO)
app_logger = logger  # Alias for compatibility

# ===================== END LOGGING CONFIGURATION =====================



# Bắt đầu background monitoring
resource_optimizer.start_background_monitoring(interval=10)  # Giám sát mỗi 10 giây

# --- Thư mục lưu trữ ---

# Toàn cục lưu trạng thái huấn luyện
training_status = {
    "train_progress": 0.0,
    "train_status": "",
    "train_error": None,
    "prediction_ready": False,
    "current_model": "",
    "eta": None,
    "train_markdown": "",
    "train_results": [],
    "train_results_df": pd.DataFrame(
        columns=["timeframe", "model", "accuracy", "backtest_accuracy"]
    ),
}

# —————————————————————————————————
# Global Instance Definitions (Missing Components)
# —————————————————————————————————

# ModelPredictor class for loading and using trained models


# Missing global variables and functions
resource_optimizer = global_resource_optimizer  # Reference to existing global instance
feature_engineering = FeatureEngineer()  # Create FeatureEngineer instance

# Historical events data (can be expanded as needed)
HISTORICAL_EVENTS = {
    "2024-01-01": "New Year Market Opening",
    "2023-11-30": "End of November Trading",
    "2023-12-25": "Christmas Market Effect",
    # Add more historical events as needed
}

def trainer_loop():
    """Trainer loop function for background training tasks"""
    try:
        logger.info("🔄 Starting trainer loop...")
        # This function can be expanded to handle background training tasks
        # For now, it serves as a placeholder for training coordination
        return True
    except Exception as e:
        logger.error(f"❌ Trainer loop failed: {e}")
        return False

def log_function_data(function_name: str, data: dict, level: str = "INFO"):
    """Log function data for debugging and monitoring"""
    try:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "function": function_name,
            "data": data,
            "level": level
        }
        
        if level.upper() == "ERROR":
            logger.error(f"[{function_name}] {data}")
        elif level.upper() == "WARNING":
            logger.warning(f"[{function_name}] {data}")
        else:
            logger.info(f"[{function_name}] {data}")
            
        return log_entry
    except Exception as e:
        logger.error(f"❌ Failed to log function data: {e}")
        return None


def eta(start_time: float, done: int, total: int) -> dict:
    """
    Enhanced ETA calculation with improved accuracy and edge case handling
    
    Args:
        start_time: Thời điểm bắt đầu (timestamp)
        done: Số bước đã hoàn thành
        total: Tổng số bước
    Returns:
        dict: Chứa các thông tin về tiến độ và thời gian
    """
    if done <= 0 or total <= 0:
        return {
            "remaining": "--:--:--",
            "progress": 0.0,
            "elapsed": "00:00:00",
            "status": "⏳ Đang bắt đầu...",
            "eta_seconds": 0,
            "rate": 0.0,
            "steps_remaining": total
        }
    
    # Ensure done doesn't exceed total    
    done = min(done, total)
    
    elapsed = time.time() - start_time
    
    # Avoid division by zero and handle very small elapsed times
    if elapsed < 0.1:  # Less than 100ms
        return {
            "remaining": "--:--:--",
            "progress": done / total,
            "elapsed": "00:00:00",
            "status": "⏳ Đang khởi tạo...",
            "eta_seconds": 0,
            "rate": 0.0,
            "steps_remaining": total - done
        }
    
    # Tính tốc độ trung bình (giây/bước) với smoothing
    rate = elapsed / done
    
    # Số bước và thời gian còn lại
    remaining_steps = total - done
    remaining_secs = int(rate * remaining_steps) if remaining_steps > 0 else 0
    
    # Tính % hoàn thành
    progress = done / total
    
    # Format thời gian sang HH:MM:SS
    def format_time(seconds: int) -> str:
        if seconds < 0:
            return "00:00:00"
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60  
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    # Enhanced status messages with more granular progress indicators
    if progress < 0.05:
        status = "⏳ Đang khởi tạo..."
    elif progress < 0.15:
        status = "🔄 Đang tải dữ liệu..."
    elif progress < 0.3:
        status = "🔄 Đang xử lý..."
    elif progress < 0.5:
        status = "📊 Đang phân tích..."
    elif progress < 0.7:
        status = "🧠 Đang huấn luyện..."
    elif progress < 0.9:
        status = "🔧 Đang tối ưu..."
    elif progress < 0.99:
        status = "🏁 Đang hoàn thiện..."
    else:
        status = "✅ Sắp hoàn thành..."
        
    return {
        "remaining": format_time(remaining_secs),
        "progress": progress,
        "elapsed": format_time(int(elapsed)),
        "status": status,
        "eta_seconds": remaining_secs,
        "rate": rate,
        "steps_remaining": remaining_steps,
        "total_steps": total,
        "completed_steps": done,
        "accuracy": f"{progress:.1%}"
    }

def prepare_df(symbol: str, tf: str, quick: bool) -> pd.DataFrame:
    """Chuẩn bị DataFrame cho TF cho symbol. Nếu quick=True, chỉ giữ 7 ngày gần nhất."""
    def update_and_prepare(symbol, tf):
        df = update_data(symbol, tf)
        if df is None:
            return None
        if df.empty:
            return None
    # Nếu thiếu timestamp và có close_time thì thêm
        if "timestamp" not in df.columns and "close_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["close_time"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df_valid, _ = validate_data(df, tf)
        return df_valid

    updated = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(update_and_prepare, symbol, tf): tf for tf in TIMEFRAMES}
        for fut in as_completed(futures):
            tf = futures[fut]
            try:
                df = fut.result()
                updated[tf] = df
            except Exception as e:
                logging.warning(f"[prepare_df] Cập nhật {tf} thất bại: {e}")
                return None

    df = updated.get(tf)
    if df is None or df.empty:
        return None
    if "close_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["close_time"], utc=True)
    # Đảm bảo timestamp dạng UTC
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = label_spikes(df)
    df = compute_indicators(df)
    for fn in [
        add_head_and_shoulders, add_double_top, add_double_bottom,
        add_triangle, add_flag, add_pennant, add_rising_wedge, add_falling_wedge,
        add_triple_top, add_triple_bottom, add_rectangle
    ]:
        df = fn(df)
    df = df.loc[:, ~df.columns.duplicated()]
    try:
        corr_dict = calculate_correlation(df, {})
        if isinstance(corr_dict, dict):
            for asset, corr_val in corr_dict.items():
                df[f"corr_{asset}"] = corr_val
        else:
            logging.warning(f"correlation trả về không phải dict: {corr_dict}")
    except Exception as e:
        logging.warning(f"Tính tương quan tài sản thất bại: {e}")

    if quick:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        df = df[df["timestamp"] >= cutoff]

    return df.reset_index(drop=True)

# --- SHAP Explain ---
SHAP_CACHE = os.path.join(os.path.dirname(__file__), "cache", "shap")
os.makedirs(SHAP_CACHE, exist_ok=True)
def shap_pipeline(symbol: str):
    logging.info(f"🔍 SHAP Explain cho {symbol}")
    tf0 = list(HISTORY_REQUIREMENTS)[0]
    path = os.path.join(MODEL_DIR, f"{symbol}_{tf0}.pkl.gz")
    if not os.path.exists(path):
        logging.error(f"❌ Thiếu model {tf0} cho SHAP")
        return
    data = pickle.load(gzip.open(path, "rb")) or {}
    model = data.get("model")
    df = prepare_df(symbol, tf0, quick=False)
    if df is None or df.empty:
        logging.error("❌ Không có dữ liệu để SHAP")
        return
    X = prepare_features(df, training=False)
    explainer = SHAPExplainer(model, X)
    explainer.explain()
    shap_vals = explainer.shap_values


def create_price_chart(df: pd.DataFrame, prediction: Dict[str, Any] = None, tp_sl: Dict[str, Any] = None) -> go.Figure:
    """
    Tạo biểu đồ giá với các đường chỉ báo kỹ thuật và dự đoán
    
    Params:
        df: DataFrame chứa dữ liệu giá và chỉ báo
        prediction: Dict chứa kết quả dự đoán (tùy chọn)
        tp_sl: Dict chứa mức take profit và stop loss (tùy chọn)
        
    Returns:
        Đối tượng Figure của Plotly
    """
    # Tạo subplot với 2 hàng (giá và chỉ báo)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, 
                       row_heights=[0.7, 0.3],
                       subplot_titles=("Giá Bitcoin/USDT", "Chỉ báo"))
    
    # Thêm nến
    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="OHLC",
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Thêm EMA
    if 'ema9' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['ema9'],
                name="ema 9",
                line=dict(color='#f48fb1', width=1)
            ),
            row=1, col=1
        )
    
    if 'ema21' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['ema21'],
                name="ema 21",
                line=dict(color='#90caf9', width=1.5)
            ),
            row=1, col=1
        )
    
    if 'ema55' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['ema55'],
                name="ema 55",
                line=dict(color='#ffcc80', width=2)
            ),
            row=1, col=1
        )
    
    # Thêm Bollinger Bands
    if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['bb_upper'],
                name="bb_upper",
                line=dict(color='rgba(38, 166, 154, 0.3)', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['bb_middle'],
                name="bb_middle",
                line=dict(color='rgba(128, 128, 128, 0.3)', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['bb_lower'],
                name="bb_lower",
                line=dict(color='rgba(239, 83, 80, 0.3)', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # Thêm chỉ báo RSI trong subplot dưới
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['rsi'],
                name="rsi",
                line=dict(color='#a5d6a7', width=1.5)
            ),
            row=2, col=1
        )
        
        # Thêm đường tham chiếu cho RSI (70 và 30)
        fig.add_trace(
            go.Scatter(
                x=[df['time'].iloc[0], df['time'].iloc[-1]],
                y=[70, 70],
                name="rsi 70",
                line=dict(color='rgba(239, 83, 80, 0.5)', width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[df['time'].iloc[0], df['time'].iloc[-1]],
                y=[30, 30],
                name="rsi 30",
                line=dict(color='rgba(38, 166, 154, 0.5)', width=1, dash='dash')
            ),
            row=2, col=1
        )
    
    # Thêm đường MACD vào subplot dưới (nếu có RSI rồi thì thêm mờ hơn)
    if all(col in df.columns for col in ['macd', 'macd_signal']):
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['macd'],
                name="macd",
                line=dict(color='#ba68c8', width=1.5),
                opacity=0.7 if 'rsi' in df.columns else 1
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['macd_signal'],
                name="signal",
                line=dict(color='#dce775', width=1.5),
                opacity=0.7 if 'rsi' in df.columns else 1
            ),
            row=2, col=1
        )
    
    # Thêm volume bar chart (không hiển thị mặc định nếu có RSI)
    fig.add_trace(
        go.Bar(
            x=df['time'],
            y=df['volume'],
            name="volume",
            marker_color='rgba(128, 128, 128, 0.4)',
            opacity=0.3
        ),
        row=2, col=1
    )
    
    # Thêm dự đoán (nếu có)
    if prediction is not None:
        pred_label = prediction.get('label', 0)
        pred_conf = prediction.get('confidence', 0.5)
        
        # Tính toán giá dự đoán tương lai (giả định)
        last_price = df['close'].iloc[-1]
        pred_change = 0.02 if pred_label == 1 else (-0.02 if pred_label == -1 else 0.001)
        pred_price = last_price * (1 + pred_change)
        
        # Chọn màu dựa trên dự đoán
        if pred_label == 1:  # Tăng
            pred_color = 'rgba(38, 166, 154, 0.8)'
            annotation_text = f"BUY<br>Conf: {pred_conf:.1%}"
        elif pred_label == -1:  # Giảm
            pred_color = 'rgba(239, 83, 80, 0.8)'
            annotation_text = f"SELL<br>Conf: {pred_conf:.1%}"
        else:  # Đi ngang
            pred_color = 'rgba(128, 128, 128, 0.8)'
            annotation_text = f"NEUTRAL<br>Conf: {pred_conf:.1%}"
        
        # Thời điểm cuối cùng
        last_time = df['time'].iloc[-1]
        next_time = last_time + pd.Timedelta(minutes=30)  # Giả định khoảng thời gian 30 phút
        
        # Thêm đường xu hướng dự đoán
        fig.add_trace(
            go.Scatter(
                x=[last_time, next_time],
                y=[last_price, pred_price],
                mode='lines',
                line=dict(color=pred_color, width=3, dash='dot'),
                name=f"Dự đoán {'Tăng' if pred_label == 1 else 'Giảm' if pred_label == -1 else 'Đi ngang'}"
            ),
            row=1, col=1
        )
        
        # Thêm annotation cho dự đoán
        fig.add_annotation(
            x=next_time,
            y=pred_price,
            text=annotation_text,
            showarrow=True,
            arrowhead=2,
            arrowcolor=pred_color,
            arrowsize=1,
            arrowwidth=2,
            ax=40,
            ay=-40,
            bgcolor=pred_color,
            bordercolor="#ffffff",
            borderwidth=1,
            borderpad=4,
            font=dict(color="#ffffff", size=12),
            opacity=0.8,
            row=1, col=1
        )
    
    # Thêm TP/SL (nếu có)
    if tp_sl is not None and tp_sl.get('signal') in ['BUY', 'SELL']:
        TP = tp_sl.get('TP')
        SL = tp_sl.get('SL')
        current_price = tp_sl.get('current_price', df['close'].iloc[-1])
        
        if TP is not None and SL is not None:
            # Chuẩn bị dữ liệu
            last_time = df['time'].iloc[-1]
            future_time = last_time + pd.Timedelta(days=1)  # Hiển thị TP/SL kéo dài 1 ngày
            
            # Take Profit
            fig.add_trace(
                go.Scatter(
                    x=[last_time, future_time],
                    y=[TP, TP],
                    mode='lines',
                    line=dict(color='rgba(38, 166, 154, 0.8)', width=2, dash='dash'),
                    name=f"Take Profit"
                ),
                row=1, col=1
            )
            
            # Stop Loss
            fig.add_trace(
                go.Scatter(
                    x=[last_time, future_time],
                    y=[SL, SL],
                    mode='lines',
                    line=dict(color='rgba(239, 83, 80, 0.8)', width=2, dash='dash'),
                    name=f"Stop Loss"
                ),
                row=1, col=1
            )
            
            # Tính % thay đổi của TP/SL
            tp_pct = (TP / current_price - 1) * 100
            sl_pct = (SL / current_price - 1) * 100
            
            # Thêm annotation cho TP
            fig.add_annotation(
                x=future_time,
                y=TP,
                text=f"TP: {format_price(TP)} ({format_change(tp_pct)})",
                showarrow=False,
                xanchor="left",
                bgcolor="rgba(38, 166, 154, 0.7)",
                bordercolor="#ffffff",
                borderwidth=1,
                borderpad=4,
                font=dict(color="#ffffff", size=12),
                opacity=0.8,
                row=1, col=1
            )
            
            # Thêm annotation cho SL
            fig.add_annotation(
                x=future_time,
                y=SL,
                text=f"SL: {format_price(SL)} ({format_change(sl_pct)})",
                showarrow=False,
                xanchor="left",
                bgcolor="rgba(239, 83, 80, 0.7)",
                bordercolor="#ffffff",
                borderwidth=1,
                borderpad=4,
                font=dict(color="#ffffff", size=12),
                opacity=0.8,
                row=1, col=1
            )
    
    # Cập nhật layout
    fig.update_layout(
        title_text="Biểu đồ giá Bitcoin/USDT và các chỉ báo kỹ thuật",
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Cập nhật trục y cho subplot RSI
    if 'rsi' in df.columns:
        fig.update_yaxes(title_text="rsi", range=[0, 100], row=2, col=1)
    elif all(col in df.columns for col in ['macd', 'macd_signal']):
        # Nếu không có RSI nhưng có MACD
        min_macd = min(df['macd'].min(), df['macd_signal'].min())
        max_macd = max(df['macd'].max(), df['macd_signal'].max())
        padding = (max_macd - min_macd) * 0.1
        fig.update_yaxes(title_text="macd", range=[min_macd - padding, max_macd + padding], row=2, col=1)
    
    return fig


def format_price(price: float, precision: int = 8, min_decimal: int = 2) -> str:
    """
    Định dạng giá tiền để hiển thị đẹp:
      - Giá >= 1: Cắt bớt số 0 cuối, nhưng luôn giữ tối thiểu min_decimal số lẻ.
      - Giá < 1: Hiển thị đúng số lẻ (precision), không cắt số 0 ở cuối (đảm bảo chính xác cho symbol nhỏ).
    Args:
        price: Giá trị float cần định dạng.
        precision: Số chữ số thập phân tối đa cho symbol nhỏ.
        min_decimal: Số chữ số lẻ tối thiểu cho symbol lớn (>=1).
    Returns:
        str: Chuỗi giá đã format.
    """
    if price is None:
        return "N/A"
    if price >= 1:
        # Tối đa 8 số lẻ, bỏ 0 cuối, nhưng luôn giữ tối thiểu min_decimal số lẻ
        s = f"{price:.8f}".rstrip('0').rstrip('.')
        if '.' in s:
            decimals = len(s.split('.')[1])
            if decimals < min_decimal:
                s += '0' * (min_decimal - decimals)
        else:
            # Giá trị nguyên, thêm .00 nếu muốn
            s += '.' + '0'*min_decimal
        return s
    else:
        # symbol nhỏ: giữ nguyên độ chính xác, KHÔNG cắt số 0 phía sau!
        s = f"{price:.{precision}f}"
        return s

def format_change(change: float, include_sign: bool = True) -> str:
    """
    Định dạng thay đổi phần trăm để hiển thị trực quan hơn.
    """
    if change is None:
        return "N/A"
    
    if include_sign and change > 0:
        return "+{:.2f}%".format(change)
    else:
        return "{:.2f}%".format(change)

def format_datetime(dt: pd.Timestamp, include_seconds: bool = False) -> str:
    """
    Định dạng datetime để hiển thị
    
    Params:
        dt: Đối tượng datetime cần định dạng
        include_seconds: Có hiển thị giây hay không
        
    Returns:
        Chuỗi định dạng, ví dụ: 25/12/2023 15:30
    """
    if dt is None:
        return "N/A"
    
    if include_seconds:
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    else:
        return dt.strftime("%d/%m/%Y %H:%M")
    
def train_model_thread(symbol, timeframe, epochs, threshold, batch_size, horizon, use_gpu=True, use_enhanced=True):
    """
    Simplified training thread function - redirects to unified Training class
    This replaces the massive duplicate implementation with a clean interface to existing training infrastructure
    """
    try:
        logger.info(f"[train_model_thread] Starting training for {symbol}@{timeframe}")
        
        # Update training status
        training_status.update({
            "train_running": True,
            "train_progress": 0.0,
            "train_status": "🔄 Initializing unified training...",
            "train_error": None
        })
        
        # Sync to session state
        for key, value in training_status.items():
            st.session_state[key] = value
        
        # Use existing unified Training class
        trainer = Training()
        results = trainer.train_all_models(symbol)
        
        # Update final status
        training_status.update({
            "train_progress": 1.0,
            "train_status": f"✅ Training completed! Best: {results.get('best_model_name', 'N/A')}",
            "train_running": False,
            "prediction_ready": True
        })
        
        # Sync final state
        for key, value in training_status.items():
            st.session_state[key] = value
            
        logger.info(f"[train_model_thread] ✅ Training completed successfully using unified infrastructure")
        
    except Exception as e:
        error_msg = f"❌ Training failed: {str(e)}"
        training_status.update({
            "train_status": error_msg,
            "train_error": str(e),
            "train_progress": 0.0,
            "train_running": False
        })
        
        # Sync error state
        for key, value in training_status.items():
            st.session_state[key] = value
            
        logger.error(f"[train_model_thread] Training error: {e}", exc_info=True)


# Helper function to save model results
def save_model_results(symbol, timeframe, accuracy, loss, epochs):
    """
    Save model results efficiently without intermediate data
    """
    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Tạo đối tượng kết quả nhỏ gọn
    result = {
        'symbol': symbol,
        'timeframe': timeframe,
        'accuracy': accuracy,
        'loss': loss,
        'epochs': epochs,
        'timestamp': datetime.now().isoformat()
    }
    
    # Lưu kết quả vào file riêng
    result_file = os.path.join(model_dir, f"{symbol}_{timeframe}_results.pkl.gz")
    try:
        with gzip.open(result_file, 'wb') as f:
            pickle.dump(result, f)
        logging.info(f"Đã lưu kết quả mô hình: {result_file}")
        return True
    except Exception as e:
        logging.error(f"Lỗi khi lưu kết quả mô hình: {e}")
        return False


# Thêm hàm mới để xóa cache dữ liệu trung gian
def clear_training_cache():
    """
    Xóa cache dữ liệu trung gian sau khi huấn luyện
    """
    # Xóa cache trong thư mục tạm
    temp_dir = os.path.join(os.getcwd(), "data", "temp")
    if os.path.exists(temp_dir):
        try:
            for file in os.listdir(temp_dir):
                if file.endswith('.temp') or file.endswith('.cache'):
                    os.remove(os.path.join(temp_dir, file))
            logging.info(f"Đã xóa cache dữ liệu trung gian trong {temp_dir}")
        except Exception as e:
            logging.warning(f"Không thể xóa cache: {e}")
    
    # Buộc garbage collection
    gc.collect()
def main():
    try:
        # 1) Tiêu đề trang
        st.title("🤖 AI Dự Báo Giao Dịch Crypto")
        st.markdown("Hệ thống dự báo giao dịch tiền điện tử theo thời gian thực với AI")

        # 2) KHỞI TẠO MẶC ĐỊNH CHO SESSION STATE
        defaults = {
            # --- TRAIN WORKFLOW ---
            "train_status":         "",     # thông báo text hiện tại
            "train_progress":       0.0,    # % hoàn thành
            "train_markdown":       "",     # rich text
            "train_results":        [],     # list kết quả metrics
            "train_error":          None,   # lỗi nếu có
            "prediction_ready":     False,  # đã train xong chưa
            "model_updating":       False,  # cờ đang train
            "train_running":        False,  # cờ tiến trình train

            # --- PREDICT WORKFLOW ---
            "predicting":           False,
            "predict_done":         0,
            "predict_total_steps":  0,
            "predict_start_time":   None,
            "prediction":    {},

            # --- CHUNG DỮ LIỆU & UI ---
            "symbol":               DEFAULT_symbol,  # mặc định từ config
            "selected_timeframe":   "15m",         # mặc định
            "data_loaded":          False,
            "price_data":           {},            # cache data cho chart
            "chart":                None,
            "prediction":           None,          # cờ đã predict xong chưa
            "tp_sl":                None,
            "explanation":          None,

            # --- TIN TỨC ---
            "max_news":          100,     # số lượng mặc định
            "good_news":         [],     
            "bad_news":          [],
            "market_score":      0.0,
            "show_news":         False,

            # --- META-AI RECOMMENDATION ---
            "meta_recommendation":  None,
            
            # --- SHAP EXPLAINER ---
            "shap_error":        None,
            "shap_vals":         None,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

        # ============================================================================
        # KHỞI TẠO NHANH: Chỉ load dữ liệu khi cần thiết, không tự động khi startup
        # ============================================================================
        
        # Kiểm tra và tải kết quả dự đoán cũ nếu có
        if not st.session_state.data_loaded:
            # Thử load kết quả cũ từ cache thay vì tính toán mới
            prediction_cache_path = os.path.join(MODEL_DIR, f"{st.session_state.symbol}_last_prediction.json")
            if os.path.exists(prediction_cache_path):
                try:
                    with open(prediction_cache_path, "r", encoding="utf-8") as f:
                        cached_prediction = json.load(f)
                    st.session_state.prediction = cached_prediction
                    logger.info(f"✅ Loaded cached prediction for {st.session_state.symbol}")
                except Exception as e:
                    logger.warning(f"Could not load cached prediction: {e}")
            
            # Thử load chart cũ nếu có
            chart_cache_path = os.path.join(MODEL_DIR, f"{st.session_state.symbol}_last_chart.pkl")
            if os.path.exists(chart_cache_path):
                try:
                    with open(chart_cache_path, "rb") as f:
                        cached_chart = pickle.load(f)
                    st.session_state.chart = cached_chart
                    logger.info(f"✅ Loaded cached chart for {st.session_state.symbol}")
                except Exception as e:
                    logger.warning(f"Could not load cached chart: {e}")
            
            # Đánh dấu đã "load" (thực ra chỉ load cache) để không lặp lại
            st.session_state.data_loaded = True
            logger.info(f"✅ Fast startup completed for {st.session_state.symbol} - using cached data if available")
            
            # Hiển thị thông báo khởi động nhanh
            if st.session_state.prediction or st.session_state.chart:
                st.success("⚡ **Khởi động nhanh hoàn tất!** App đã load kết quả dự đoán từ cache.")
            else:
                st.info("🚀 **Khởi động nhanh!** App sẵn sàng hoạt động. Click 'Cập nhật Dự đoán' để bắt đầu tính toán.")

        # 3) HIỂN THỊ TRẠNG THÁI TRAIN / PREDICT
        if st.session_state.train_status:
            st.info(st.session_state.train_status)
        st.progress(st.session_state.train_progress)

        if st.session_state.train_markdown:
            st.markdown(st.session_state.train_markdown, unsafe_allow_html=True)

        if st.session_state.train_results:
            df_metrics = pd.DataFrame(st.session_state.train_results)
            st.table(df_metrics)

        if st.session_state.train_error:
            st.error(st.session_state.train_error)
            st.stop()

        if st.session_state.prediction_ready:
            st.success("✅ Hoàn tất huấn luyện và đã cập nhật dự đoán!")

        # 4) Nếu đang TRAIN hoặc PREDICT, sync và show tiến độ
        if st.session_state.model_updating or st.session_state.predicting:
            # đồng bộ từ global training_status vào session_state
            for k, v in training_status.items():
                st.session_state[k] = v

            pct = st.session_state.get("train_progress", 0.0)
            eta_info = st.session_state.get("eta_info", {})
            current = st.session_state.get("current_model", "")
            
            # Hiển thị system health và tài nguyên
            health_score = resource_optimizer.get_system_health_score()
            optimal_device = resource_optimizer.get_optimal_device()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏥 System Health", f"{health_score:.0f}/100")
            with col2:
                st.metric("🖥️ Device", optimal_device.upper())
            with col3:
                if eta_info:
                    st.metric("⏱️ ETA", eta_info.get("remaining", "--:--:--"))

            # Thanh tiến độ với thông tin chi tiết
            progress_text = f"**{current}**" if current else "Processing"
            status = eta_info.get("status", "⏳ Đang xử lý...")
            
            if eta_info:
                st.info(f"{status} — {progress_text} — {pct*100:.1f}%")
                # Detailed progress bar với ETA
                remaining = eta_info.get("remaining", "--:--:--")
                elapsed = eta_info.get("elapsed", "--:--:--")
                rate = eta_info.get("rate", 0)
                
                # Progress bar với thông tin chi tiết
                st.progress(pct, text=f"⏱️ ETA: {remaining} | Elapsed: {elapsed} | Rate: {rate:.2f}s/step")
                
                # ETA details in columns for better visibility
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("⏳ Remaining", remaining)
                with col2:
                    st.metric("⏱️ Elapsed", elapsed)  
                with col3:
                    st.metric("🎯 Progress", f"{pct*100:.1f}%")
                with col4:
                    st.metric("⚡ Rate", f"{rate:.2f}s/step" if rate > 0 else "--")
            else:
                st.info(f"⏳ Tiến độ: {pct*100:.1f}%")
                st.progress(pct, text=f"Progress: {pct*100:.1f}%")
                
            # Smart autorefresh with ETA-aware timing
            if st.session_state.model_updating or st.session_state.predicting:
                # Determine refresh interval based on activity and state
                if st.session_state.get("train_running", False) or st.session_state.get("predicting", False):
                    # Khi đang thực hiện tích cực, sử dụng khoảng cách lâu hơn để không ảnh hưởng ETA
                    refresh_interval = 15_000  # 15 seconds
                else:
                    # Đã hoàn thành hoặc ở trạng thái chờ
                    refresh_interval = 8_000 if eta_info else 5_000  # 8s với ETA, 5s không ETA
            
            st_autorefresh(interval=refresh_interval, key="train_auto_refresh")

        # 5) META-AI: nếu đã có train_results, thì khởi tạo và lấy gợi ý
        if st.session_state.train_results:
            metrics_df = pd.DataFrame(st.session_state.train_results)
            meta = MetaAI(st.session_state.symbol, metrics_df)
            try:
                st.session_state.meta_recommendation = meta.recommend(metric="f1")
            except ValueError as e:
                st.warning(f"⚠️ Meta-AI recommendation skipped: {e}")

        # 6) SIDEBAR – từ đây là UI tùy chọn
        with st.sidebar:
            st.header("⚙️ Tùy chọn")

            # 6.1) Hiển thị Meta-AI gợi ý (nếu có)
            if st.session_state.meta_recommendation:
                best_tf = st.session_state.meta_recommendation["best_timeframe"]
                best_md = st.session_state.meta_recommendation["best_model"]
                st.success(f"🔮 Meta-AI gợi ý: Khung **{best_tf}**, Model **{best_md}**")
                # override khung thời gian theo gợi ý
                st.session_state.selected_timeframe = best_tf    

            symbol = st.selectbox("Cặp giao dịch", ["BTCUSDT"], index=0)
            tf = st.session_state.get("timeframe", "5m")
   # chuyển "5m","15m","1h" → số phút
            tf_min = int(tf.rstrip("m").rstrip("h")) * (60 if "h" in tf else 1)
            _ = st_autorefresh(interval=tf_min * 60_000, key="price_refresh")
            if symbol != st.session_state.symbol:
                st.session_state.symbol = symbol
                st.session_state.data_loaded = False

            def fetch_price(symbol: str) -> float:
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                return float(resp.json()["price"])
                
                # ở trong with st.sidebar:
            price_container = st.empty()
            time_container  = st.empty()
            try:
                price = fetch_price(symbol)
                ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if "base_price" not in st.session_state:
                # Tính timestamp ms cho 07:00 sáng HCM
                    vn_tz      = pytz.timezone("Asia/Ho_Chi_Minh")
                    now_vn     = datetime.now(vn_tz)
                    base_dt    = now_vn.replace(hour=7, minute=0, second=0, microsecond=0)
                    start_ms   = int(base_dt.timestamp() * 1000)
                    end_ms     = start_ms + 60*60*1000 - 1

                    resp = requests.get(
                        "https://api.binance.com/api/v3/klines",
                        params={
                        "symbol":    symbol,
                        "interval":  "1h",
                        "startTime": start_ms,
                        "endTime":   end_ms,
                        "limit":     1
                        },
                        timeout=5
                    )
                    resp.raise_for_status()
                    klines = resp.json()
                    if klines and len(klines[0]) >= 5:
                    # data[0][4] là giá đóng phiên
                        st.session_state.base_price = float(klines[0][4])
                    else:
                    # fallback nếu không có nến 7:00
                        st.session_state.base_price = price

                base_price = st.session_state.base_price

            # 3) So sánh & hiển thị
                change_pct = (price - base_price) / base_price * 100
                arrow      = "▲" if price >= base_price else "▼"
                color      = "green" if price >= base_price else "red"
                pct_str    = f"{change_pct:+.2f}%"  # dấu + cho biết lên/xuống


                price_container.markdown(
                    f"**Giá {symbol}:** "
                    f"<span style='color:{color}; font-size:1.3em'>"
                    f"{arrow} {format_price(price)} ({pct_str})"
                    "</span>",
                    unsafe_allow_html=True
                )
                time_container.caption(f"Cập nhật: {ts}")

            except Exception as e:
                price_container.error("Không thể lấy giá")
                time_container.caption(f"Lỗi: {e}")        
            # ————————————————————————————————————————————————
            # Chọn khung thời gian
            timeframe = st.selectbox(
                "Khung thời gian", 
                ["5m", "15m", "1h", "4h"],
                index=1  # Mặc định là 15m
            )
            
            # Cập nhật khung thời gian đã chọn
            if timeframe != st.session_state.selected_timeframe:
                st.session_state.selected_timeframe = timeframe
                st.session_state.data_loaded = False 
                    
                # --- Nút TẢI LẠI DỮ LIỆU ---
               # Nếu đang có thread huấn luyện chạy, disable nút reload
            train_thr = st.session_state.get("train_thread")
            running = train_thr.is_alive() if train_thr else False
            if running:
                st.button("🔄 Tải lại dữ liệu", key="reload_data", disabled=True)
                st.info("🚀 Đang huấn luyện AI – không thể tải lại dữ liệu.")
            else:
                if st.button("🔄 Tải lại dữ liệu", key="reload_data"):
                    df_new = update_data(symbol, timeframe)
                    if df_new is None or df_new.empty:
                        st.error("❌ Không thể tải dữ liệu, vui lòng thử lại.")
                        st.session_state.data_loaded = False
                    else:
                        st.session_state.price_data[timeframe] = df_new
                        st.session_state.data_loaded = True
                        st.session_state['predicting']     = False
                        st.session_state['model_updating'] = False
                        st.session_state['train_running']  = False
                        st.success("✅ Đã tải dữ liệu thành công!")

            # --- Nút CẬP NHẬT DỰ ĐOÁN ---
            if st.button(
                "🔄 Cập nhật dự đoán",
                key="update_prediction",
                disabled=st.session_state.get("predicting", False)
            ):
                st.session_state['predicting']        = True
                st.session_state['predict_done']       = 0
                st.session_state['predict_total_steps'] = 2   # chúng ta dự kiến có 2 bước: load model + fetch data
                st.session_state['predict_start_time'] = time.time()
                threading.Thread(
                    target=run_prediction_thread,
                    args=(symbol, st.session_state.selected_timeframe),
                    daemon=True
                ).start()

            # ETA khi đang predict
            if st.session_state.get('predicting', False):
               
                done  = st.session_state.get('predict_done', 0)
                total = st.session_state.get('predict_total_steps', 0)
                start = st.session_state.get('predict_start_time', time.time())
                remaining = eta(start, done, total)
                st.info(f"🚧 Predicting... ETA {remaining}")

            # Hiển thị kết quả predict
            pred = st.session_state.get("prediction")
            if isinstance(pred, dict) and 'Entry' in pred:
                st.markdown("---")
                st.subheader("📌 Kết Quả")
                st.write(f"**Entry:**     {pred['Entry']:.2f}")
                st.write(f"**TP:**        {pred['TP']:.2f}")
                st.write(f"**SL:**        {pred['SL']:.2f}")
                st.write(f"**Hướng:**     {pred['Direction']}")
            else:
                st.info("⏳ Chưa có dự đoán, vui lòng nhấn 'Cập nhật dự đoán'")

        # Đồng hồ ngày tháng chạy realtime
            st.markdown("---")
            st.subheader("⏰ Thời gian thực") 
            
            html("""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .realtime-clock {
                    background: rgba(255,255,255,0.1);
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                    font-family: monospace;
                }
            </style>
        </head>
        <body>
            <div class="realtime-clock" id="clock"></div>
            <script>
                function updateClock() {
                    var clock = document.getElementById('clock');
                    var now = new Date();
                    var options = {
                        timeZone: 'Asia/Ho_Chi_Minh',
                        year: 'numeric',
                        month: '2-digit',
                        day: '2-digit',
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                        hour12: false
                    };
                    clock.textContent = now.toLocaleString('vi-VN', options);
                }
                updateClock();
                setInterval(updateClock, 1000);
            </script>
        </body>
        </html>
    """, height=80)

    # Load Meta-AI recommendation và accuracy
            try:
                if os.path.exists(os.path.join(MODEL_DIR, f"{symbol}_meta.json")):
                    with open(os.path.join(MODEL_DIR, f"{symbol}_meta.json"), "r") as f:
                        meta_data = json.load(f)
                
                    st.markdown("---")
                    st.subheader("🔮 Meta-AI Gợi ý")
            
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Khung thời gian tốt nhất", meta_data["best_timeframe"])
                        st.metric("Độ chính xác", f"{meta_data['accuracy']*100:.1f}%")
                    with col2:
                        st.metric("Loại model", meta_data["best_model"])
                        st.metric("Backtest accuracy", f"{meta_data['backtest_accuracy']*100:.1f}%")
                
                    st.info(f"""
                👉 Gợi ý giao dịch:
                - Hướng: **{meta_data['direction']}**
                - Entry: **{meta_data['entry']:.2f}**
                - TP: **{meta_data['tp']:.2f}**
                - SL: **{meta_data['sl']:.2f}**
                    """)
            
            except Exception as e:
                st.warning("⚠️ Chưa có gợi ý từ Meta-AI")

            # Đọc độ chính xác từ cache
            accuracy_cache = os.path.join(DATA_DIR, "accuracy_cache.pkl.gz")
            if os.path.exists(accuracy_cache) and "train_results" not in st.session_state:
                try:
                    with gzip.open(accuracy_cache, "rb") as f:
                        cached_results = pickle.load(f)
                    if cached_results:
                        st.session_state.train_results = cached_results
                except Exception as e:
                    logging.warning(f"Không load được cache độ chính xác: {e}")

            # Hiển thị bảng độ chính xác nếu có
            if "train_results" in st.session_state and st.session_state.train_results:
                with st.expander("🎯 Độ chính xác hiện tại", expanded=True):
                    df_acc = pd.DataFrame(st.session_state.train_results)
                    df_acc["accuracy"] = df_acc["accuracy"].apply(
                        lambda x: f"{x*100:.2f}%" if pd.notna(x) else "–"
                    )
                    df_acc["backtest_accuracy"] = df_acc["backtest_accuracy"].apply(
                        lambda x: f"{x*100:.2f}%" if pd.notna(x) else "–"
                    )
                    st.table(df_acc[["timeframe", "model", "accuracy", "backtest_accuracy"]])

        # ====== GIAO DIỆN CHÍNH VỚI SELECTBOX THAY VÌ TABS ======
        with st.container():
            st.markdown("<div class='page-box'>", unsafe_allow_html=True)
            
            # Hiển thị bảng độ chính xác tự động khi load
            st.subheader("📊 Bảng độ chính xác hiện tại")
            accuracy_file = os.path.join(MODEL_DIR, f"{symbol}_accuracy.json")
            if os.path.exists(accuracy_file):
                try:
                    with open(accuracy_file, "r") as f:
                        accuracy_data = json.load(f)
                    
                    if accuracy_data.get("accuracy_results"):
                        df_acc = pd.DataFrame(accuracy_data["accuracy_results"])
                        # Format phần trăm để hiển thị đẹp
                        df_acc["accuracy_display"] = df_acc["accuracy"].apply(
                            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "–"
                        )
                        df_acc["backtest_display"] = df_acc["backtest_accuracy"].apply(
                            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "–"
                        )
                        
                        # Hiển thị bảng gọn
                        display_df = df_acc[["timeframe", "model", "accuracy_display", "backtest_display"]].copy()
                        display_df.columns = ["Khung thời gian", "Mô hình", "Độ chính xác", "Backtest"]
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Hiển thị thông tin meta
                        if "meta" in accuracy_data:
                            meta = accuracy_data["meta"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📈 Số mô hình", meta.get("total_models", 0))
                            with col2:
                                best = meta.get("best_model", {})
                                best_acc = best.get("accuracy", 0) * 100 if best.get("accuracy") else 0
                                st.metric("🏆 Tốt nhất", f"{best_acc:.1f}%")
                            with col3:
                                avg_acc = meta.get("average_accuracy", 0) * 100 if meta.get("average_accuracy") else 0
                                st.metric("📊 Trung bình", f"{avg_acc:.1f}%")
                except Exception as e:
                    st.info("⚠️ Không thể load dữ liệu độ chính xác: " + str(e))
            else:
                st.info("ℹ️ Chưa có dữ liệu độ chính xác. Vui lòng huấn luyện mô hình trước.")
            
            st.markdown("---")
            
            # Hộp chọn chức năng
            choice = st.selectbox(
                "📑 Chọn chức năng",
                ["Trade View", "Phân tích kỹ thuật", "Tin tức", 
                 "Huấn luyện", "Gửi Zalo", "Cấu hình", "Dự báo dài hạn"],
                index=0
            )
            st.markdown("---")

            # 3) Xử lý nội dung theo lựa chọn
            if choice == "Trade View":
                st.header("📊 Trade View")
                
                # Log dữ liệu Trade View
                trade_data = {
                    "symbol": st.session_state.get("symbol"),
                    "timeframe": st.session_state.get("selected_timeframe"),
                    "prediction": st.session_state.get("prediction"),
                    "tp_sl": st.session_state.get("tp_sl"),
                    "explanation": st.session_state.get("explanation"),
                    "chart_available": st.session_state.get("chart") is not None,
                    "price_data_keys": list(st.session_state.get("price_data", {}).keys())
                }
                log_function_data("TRADE_VIEW", trade_data)
                
                # Hiển thị biểu đồ giá
                if st.session_state.chart:
                    st.plotly_chart(st.session_state.chart, use_container_width=True)
                else:
                    # Thử tạo chart từ dữ liệu đã cache nếu có prediction
                    if st.session_state.prediction:
                        try:
                            # TRÁNH GỌI update_data() để không trigger heavy loading!
                            # Thay vào đó, chỉ đọc từ file processed nếu có
                            
                            symbol_name = f"{st.session_state.symbol}USDT" if not st.session_state.symbol.endswith("USDT") else st.session_state.symbol
                            data_file = os.path.join(DATA_DIR, f"{symbol_name}@{st.session_state.selected_timeframe}.pkl.gz")
                            
                            if os.path.exists(data_file):
                                # Đọc dữ liệu đã có từ file cache
                                with gzip.open(data_file, 'rb') as f:
                                    df_temp = pickle.load(f)
                                
                                if df_temp is not None and not df_temp.empty:
                                    # Chỉ tính indicators cơ bản, không làm full processing
                                    df_temp = compute_indicators(df_temp)
                                    
                                    # Thêm cột time nếu chưa có
                                    if 'time' not in df_temp.columns:
                                        if 'close_time' in df_temp.columns:
                                            df_temp['time'] = pd.to_datetime(df_temp['close_time'], utc=True)
                                        elif 'timestamp' in df_temp.columns:
                                            df_temp['time'] = pd.to_datetime(df_temp['timestamp'], utc=True)
                                        else:
                                            df_temp['time'] = pd.to_datetime(df_temp.index)
                                    
                                    temp_chart = create_price_chart(
                                        df_temp,
                                        prediction=st.session_state.prediction,
                                        tp_sl=st.session_state.tp_sl
                                    )
                                    st.plotly_chart(temp_chart, use_container_width=True)
                                    st.info("📊 Biểu đồ được tạo từ dữ liệu cache cục bộ (fast mode)")
                                else:
                                    st.info("⏳ Chưa có biểu đồ, vui lòng cập nhật dự đoán để tạo biểu đồ")
                            else:
                                st.info("⏳ Chưa có dữ liệu cache, vui lòng cập nhật dự đoán để tải dữ liệu và tạo biểu đồ")
                        except Exception as e:
                            logger.warning(f"Could not create temporary chart from cache: {e}")
                            st.info("⏳ Chưa có biểu đồ, vui lòng cập nhật dự đoán để tạo biểu đồ")
                    else:
                        st.info("⏳ Chưa có biểu đồ, vui lòng cập nhật dự đoán để tạo biểu đồ")
                
                # Hiển thị thông tin dự báo
                if st.session_state.prediction:
                    # Hiển thị thông báo nếu đang sử dụng cache
                    prediction_cache_path = os.path.join(MODEL_DIR, f"{st.session_state.symbol}_last_prediction.json")
                    if os.path.exists(prediction_cache_path):
                        try:
                            cache_time = datetime.fromtimestamp(os.path.getmtime(prediction_cache_path))
                            time_diff = datetime.now() - cache_time
                            if time_diff.total_seconds() < 86400:  # Trong vòng 24h
                                st.info(f"📋 Hiển thị kết quả dự đoán từ cache ({cache_time.strftime('%H:%M %d/%m')})")
                        except:
                            st.info("📋 Hiển thị kết quả dự đoán từ cache")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Dự báo")
                        direction = st.session_state.prediction.get("Direction", "Unknown")
                        probability = st.session_state.prediction.get("Probability", 0)
                        st.markdown(f"**Hướng:** {direction}")
                        st.markdown(f"**Xác suất:** {probability:.2f}%")
                    
                    with col2:
                        st.subheader("Mức giá")
                        entry = st.session_state.prediction.get("Entry", 0)
                        tp = st.session_state.prediction.get("TP", 0)
                        sl = st.session_state.prediction.get("SL", 0)
                        st.markdown(f"**Entry:** {entry:.2f}")
                        st.markdown(f"**TP:** {tp:.2f}")
                        st.markdown(f"**SL:** {sl:.2f}")
                else:
                    st.info("⏳ Chưa có dự đoán, vui lòng nhấn 'Cập nhật dự đoán'")
                
                # Hiển thị giải thích
                if st.session_state.explanation:
                    st.subheader("Giải thích")
                    for feature, impact in st.session_state.explanation.items():
                        st.markdown(f"- {feature}: {impact}")

            elif choice == "Phân tích kỹ thuật":
                st.header("📈 Phân tích kỹ thuật")
                
                # 1) Chọn khung thời gian (tf) luôn được gán
                tf = st.session_state.selected_timeframe
                
                # Log dữ liệu Phân tích kỹ thuật
                tech_data = {
                    "timeframe": tf,
                    "symbol": st.session_state.get("symbol"),
                    "shap_error": st.session_state.get("shap_error"),
                    "shap_vals_available": st.session_state.get("shap_vals") is not None,
                    "price_data_available": tf in st.session_state.get("price_data", {}),
                }
                log_function_data("TECHNICAL_ANALYSIS", tech_data)
                
                if st.button("🔍 Tính SHAP cho khung " + tf):
                    st.session_state.shap_error  = None
                    st.session_state.shap_vals   = None
                    try:
                        vals = shap_pipeline(st.session_state.symbol)
                        if vals is None:
                            st.session_state.shap_error = "Không tạo được SHAP values."
                        else:
                            st.session_state.shap_vals = vals
                            log_function_data("SHAP_CALCULATION", {"shap_values": vals})
                    except Exception as e:
                        st.session_state.shap_error = str(e)
                        app_logger.error(f"SHAP calculation error: {e}")

                # ——— Hiển thị kết quả SHAP ———
                if st.session_state.shap_error:
                    st.error(st.session_state.shap_error)
                elif st.session_state.shap_vals is not None:
                    st.subheader(f"SHAP Values – {tf}")
                    st.write(st.session_state.shap_vals)
            
                # 2) Lấy dict price_data an toàn
                price_dict = st.session_state.get("price_data", {})
                # 2.1) Nếu tf chưa có key này, thử load từ cache file trên đĩa
                if tf not in price_dict:
                    cache_file = os.path.join(DATA_DIR, f"{st.session_state.symbol}@{tf}.pkl.gz")
                    if os.path.exists(cache_file):
                        try:
                            with gzip.open(cache_file, "rb") as f:
                                df_cached = pickle.load(f)
                            if df_cached is not None and not df_cached.empty:
                                # ghi lại vào session_state để lần sau có sẵn
                                st.session_state.price_data[tf] = df_cached
                                price_dict[tf] = df_cached
                                app_logger.info(f"Loaded cache data for {tf}: {df_cached.shape}")
                        except Exception as e:
                            st.warning(f"⚠️ Không load được cache {tf}: {e}")
                            app_logger.error(f"Cache load error for {tf}: {e}")
                df = price_dict.get(tf)
                
                # Log thông tin DataFrame
                if df is not None:
                    log_function_data("TECHNICAL_DATA", {
                        "dataframe_shape": df.shape,
                        "columns": list(df.columns),
                        "latest_close": df["close"].iloc[-1] if "close" in df.columns else None,
                        "date_range": f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else None
                    })
                
                # 3) Nếu tf chưa có key hoặc df None/empty → cảnh báo và dừng
                if df is None or df.empty:
                    st.warning(
                        "⏳ Chưa có dữ liệu kỹ thuật cho khung thời gian này.\n"
                        "Vui lòng quay lại Tab Giao dịch và nhấn 'Cập nhật dự đoán' để tải về dữ liệu."
                    )
                    app_logger.warning(f"No technical data available for timeframe {tf}")
                else:     
                    # 5) Danh sách (label, col_name) để hiển thị metric
                    indicators = [
                        ("RSI",             "rsi"),
                        ("MACD",            "macd"),
                        ("ATR",             "atr"),
                        ("Volume",          "volume"),
                        ("BB Upper",        "bb_upper"),
                        ("BB Mid",          "bb_middle"),
                        ("BB Lower",        "bb_lower"),
                        ("CCI",             "cci"),
                        ("ADX",             "adx"),
                        ("Ichimoku A",      "ichimoku_a"),
                        ("Ichimoku B",      "ichimoku_b"),
                        ("Ichimoku Base",   "ichimoku_base"),
                        ("Ichimoku Conv.",  "ichimoku_conv"),
                        ("Stoch %K",        "stochk"),
                        ("Stoch %D",        "stochd"),
                        ("Williams %R",     "willr"),
                        ("OBV",             "obv"),
                        ("MFI",             "mfi"),
                        ("PSAR",            "psar"),
                        ("KC Up",           "kc_up"),
                        ("KC Mid",          "kc_mid"),
                        ("KC Lo",           "kc_lo"),
                        ("Vortex +",        "vi_plus"),
                        ("Vortex -",        "vi_minus"),
                        ("Ult. Osc.",       "ult_osc"),
                        ("TSI",             "tsi"),
                        ("EOM",             "eom"),
                        ("VWAP",            "vwap"),
                        ("ROC",             "roc"),
                    ]

                    # 6) Hiển thị metric 4 cột
                    cols = st.columns(5)
                    for i, (label, col_name) in enumerate(indicators):
                        if col_name in df.columns:
                            val = df[col_name].iloc[-1]
                            disp = f"{val:.2f}"
                        else:
                            disp = "N/A"
                        cols[i % 5].metric(label, disp)

                    # chuyển khung ngang X thành khung thời gian trên biểu đồ
                    if "timestamp" in df.columns:
                        x = pd.to_datetime(df["timestamp"])
                                      


                    elif "time" in df.columns:
                        x = pd.to_datetime(df["time"])
                    else:
                        # fallback: nếu index có thể parse thành datetime
                        try:
                            x = pd.to_datetime(df.index)
                        except Exception:
                            x = df.index
                    # 7) Vẽ ví dụ biểu đồ Giá + Bollinger + ATR
                    fig = go.Figure()
                    # Giá đóng
                    fig.add_trace(
                        go.Scatter(x=x, y=df["close"], name="Close", line=dict(width=1.5))
                    )

                    # Bollinger Bands
                    if {"bb_upper", "bb_middle", "bb_lower"}.issubset(df.columns):
                        fig.add_trace(go.Scatter(x=x, y=df["bb_upper"], name="BB Up",    line=dict(dash="dash")))
                        fig.add_trace(go.Scatter(x=x, y=df["bb_middle"], name="BB Mid",  line=dict(dash="dot")))
                        fig.add_trace(go.Scatter(x=x, y=df["bb_lower"], name="BB Low",   line=dict(dash="dash")))

                    # ATR
                    if "atr" in df.columns:
                        fig.add_trace(go.Scatter(x=x, y=df["atr"], name="ATR", fill="tozeroy", opacity=0.2))

                    # Cấu hình trục X hiển thị datetime
                    fig.update_xaxes(
                        type="date",
                        tickformat="%d/%m %H:%M",  # ngày/tháng Giờ:Phút
                        nticks=8,
                        ticks="outside",
                    )
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=30, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            elif choice == "Tin tức":
                st.header("📰 Tin tức")
                
                # Log dữ liệu Tin tức
                news_data = {
                    "symbol": st.session_state.get("symbol"),
                    "max_news": st.session_state.get("max_news"),
                    "show_news": st.session_state.get("show_news"),
                    "good_news_count": len(st.session_state.get("good_news", [])),
                    "bad_news_count": len(st.session_state.get("bad_news", [])),
                    "market_score": st.session_state.get("market_score"),
                    "news_table_available": "news_table" in st.session_state and not st.session_state.get("news_table", pd.DataFrame()).empty
                }
                log_function_data("NEWS", news_data)
                
                cache_path = os.path.join(DATA_DIR, "news_cache.csv")
                if os.path.exists(cache_path) and not st.session_state.show_news:
                    df_cache = pd.read_csv(cache_path, parse_dates=["publishedAt"])
                    # Chuyển thành list of dict để reuse phần hiển thị
                    st.session_state.news_table  = df_cache
                    st.session_state.good_news   = df_cache[df_cache["sentiment"] > 0].to_dict("records")
                    st.session_state.bad_news    = df_cache[df_cache["sentiment"] < 0].to_dict("records")
                    st.session_state.market_score = df_cache["sentiment"].mean()
                    st.session_state.show_news    = True
                    app_logger.info(f"Loaded news cache: {len(df_cache)} articles")
                
                # Số tin tối đa do user config (sidebar hoặc tab6)
                max_news = st.session_state.max_news

                # Khi đổi symbol hoặc số tin, reset flag để cần bấm lại
                if symbol != st.session_state.symbol or max_news != st.session_state.max_news:
                    st.session_state.show_news = False

                if st.button("🔄 Cập nhật tin tức", key="refresh_news"):
                    with st.spinner(f"Đang lấy {max_news} tin cho {symbol}..."):
                        good, bad, score = get_news_summary(symbol, max_items=max_news)
                        all_news = good + bad
                        for n in all_news:
                            if "publishedAt" not in n:
                                # dùng lần lượt các key khả dĩ
                                n["publishedAt"] = n.get("published_at") or n.get("date") or ""
                        st.session_state.good_news = good
                        st.session_state.bad_news = bad
                        st.session_state.market_score = score
                        # Gộp thành DataFrame duy nhất
                        st.session_state.news_table = pd.DataFrame(all_news)
                        st.session_state.show_news = True
                        
                        # # Log kết quả cập nhật tin tức
                        news_update_data = {
                            "total_news": len(all_news),
                            "good_news": len(good),
                            "bad_news": len(bad),
                            "market_score": score
                        }
                        log_function_data("NEWS_UPDATE", news_update_data)

                st.subheader("📋 Bảng tổng hợp tất cả tin tức")

                if "news_table" in st.session_state and not st.session_state.news_table.empty:
                    df_news = st.session_state.news_table.copy()
                    df_news = df_news[["publishedAt", "title", "sentiment", "summary", "url"]]
                    df_news = df_news.sort_values("publishedAt", ascending=False)           
                    df_news.columns = ["Thời gian", "Tiêu đề", "Cảm xúc", "Tóm tắt", "Nguồn"]
                    st.dataframe(df_news, use_container_width=True)

                    # Optional: Nút tải CSV
                    csv = df_news.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Tải bảng tin dạng CSV", csv, "news_summary.csv", "text/csv", key="download_news")

                if not st.session_state.show_news:
                    st.info("Chưa có dữ liệu tin tức, nhấn nút Cập nhật.")
                else:
                    # Hiển thị điểm sentiment
                    st.sidebar.metric("🔍 Sentiment thị trường", f"{st.session_state.market_score:.2f}")

                    st.subheader(f"🟢 Tin tích cực ({len(st.session_state.good_news)})")
                    for n in st.session_state.good_news:
                        st.markdown(f"**{n['title']}** *(+{n['sentiment']:.2f})*")
                        st.markdown(f"> {n['summary']}")
                        st.markdown(f"[🔗]({n['url']})  🕒 {n['publishedAt']}")
                        st.write("---")

                    st.subheader(f"🔴 Tin tiêu cực ({len(st.session_state.bad_news)})")
                    for n in st.session_state.bad_news:
                        st.markdown(f"**{n['title']}** *({n['sentiment']:.2f})*")
                        st.markdown(f"> {n['summary']}")
                        st.markdown(f"[🔗]({n['url']})  🕒 {n['publishedAt']}")
                        st.write("---")
                gc.collect()

            elif choice == "Huấn luyện":
                st.header("🧠 Huấn luyện mô hình")
                
                # Log dữ liệu Huấn luyện
                train_data = {
                    "model_updating": st.session_state.get("model_updating"),
                    "train_running": st.session_state.get("train_running"),
                    "train_progress": st.session_state.get("train_progress"),
                    "train_results_count": len(st.session_state.get("train_results", [])),
                    "prediction_ready": st.session_state.get("prediction_ready"),
                    "train_error": st.session_state.get("train_error"),
                    "symbol": st.session_state.get("symbol"),
                    "selected_timeframe": st.session_state.get("selected_timeframe")
                }
                log_function_data("TRAINING", train_data)
                
                # Hiển thị trạng thái hệ thống
                col1, col2, col3 = st.columns(3)
                with col1:
                    health_score = resource_optimizer.get_system_health_score()
                    st.metric("🏥 System Health", f"{health_score:.0f}/100")
                with col2:
                    optimal_device = resource_optimizer.get_optimal_device()
                    st.metric("🖥️ Optimal Device", optimal_device.upper())
                with col3:
                    stats = resource_optimizer.monitor_cpu()
                    st.metric("💾 RAM Usage", f"{stats['memory']:.1f}%")
                
                # Log system stats
                system_stats = {
                    "health_score": health_score,
                    "optimal_device": optimal_device,
                    "ram_usage": stats['memory'] if stats else None
                }
                log_function_data("SYSTEM_STATS", system_stats)
                
                # ► Hiển thị kết quả (nếu có)
                st.subheader("🔍 Độ chính xác các mô hình hiện tại")
                st.markdown(training_status["train_markdown"], unsafe_allow_html=True)
                st.progress(training_status["train_progress"])
                
                # Hiển thị ETA với thông tin chi tiết và robust handling
                eta_info = training_status.get("eta_info", {})
                if eta_info and eta_info.get("remaining") != "--:--:--":
                    # Rich ETA display with multiple metrics
                    eta_col1, eta_col2, eta_col3 = st.columns(3)
                    with eta_col1:
                        st.metric("⏱️ ETA", eta_info.get("remaining", "--:--:--"))
                    with eta_col2:
                        st.metric("⏲️ Elapsed", eta_info.get("elapsed", "00:00:00"))
                    with eta_col3:
                        progress_pct = eta_info.get("progress", 0) * 100
                        st.metric("📊 Progress", f"{progress_pct:.1f}%")
                    
                    # Detailed status with rate information
                    rate = eta_info.get("rate", 0)
                    total_steps = eta_info.get("total_steps", 0)
                    completed_steps = eta_info.get("completed_steps", 0)
                    
                    st.info(
                        f"{eta_info.get('status', '⏳')} — "
                        f"Step {completed_steps}/{total_steps} — "
                        f"Rate: {rate:.2f}s/step"
                    )
                else:
                    # Fallback display for missing or incomplete ETA info
                    st.write(training_status["train_status"], training_status.get("eta", ""))
                    
                if training_status["train_error"]:
                    st.error(training_status["train_error"])
                    st.stop()
                    
                if st.session_state.get("train_results"):
                    df_acc = pd.DataFrame(st.session_state.train_results)
                    # Format phần trăm
                    df_acc["accuracy"] = df_acc["accuracy"].apply(
                        lambda x: f"{x*100:.2f}%" if pd.notna(x) else "–"
                    )
                    df_acc["backtest_accuracy"] = df_acc["backtest_accuracy"].apply(
                        lambda x: f"{x*100:.2f}%" if pd.notna(x) else "–"
                    )
                    st.table(df_acc[["timeframe", "model", "accuracy", "backtest_accuracy"]])
                else:
                    st.info("Chưa có kết quả huấn luyện nào.")

                # Tùy chọn huấn luyện
                st.subheader("Tùy chọn huấn luyện")
                col1, col2 = st.columns(2)
                with col1:
                    horizon = st.slider("Horizon (số nến)", 1, 24, 12)
                    threshold = st.slider("Ngưỡng biến động (%)", 0.1, 5.0, 1.0)
                    # Thêm biến epochs vào đây
                    epochs = st.slider("Số epochs", 5, 100, 20)
                with col2:
                     # Thêm biến batch_size vào đây
                    batch_size = st.slider("Batch size", 16, 256, 64, step=16)
                    # Thêm biến use_gpu vào đây nếu có GPU
                    use_gpu = st.checkbox("Sử dụng GPU (nếu có)", value=USE_GPU)
                    # Thêm lựa chọn chế độ huấn luyện enhanced
                    use_enhanced = st.checkbox("🚀 Huấn luyện nâng cao (Multi-threaded)", value=True, 
                                             help="Sử dụng hệ thống huấn luyện đa luồng với tối ưu hóa GPU và tài nguyên")
                    # Thêm lựa chọn chế độ huấn luyện
                    mode_options = ["Huấn luyện nhanh (7 ngày)", "Huấn luyện chuyên sâu (365 ngày)"]
                    train_mode = st.radio("Chế độ huấn luyện", mode_options, index=0)
                    quick_mode = (train_mode == mode_options[0])
                    use_sentiment = st.checkbox("Sử dụng sentiment", value=True)
                    use_technical = st.checkbox("Sử dụng chỉ báo kỹ thuật", value=True)
                
                # --- Bảng độ chính xác hiện tại ---
                st.subheader("🔍 Độ chính xác các mô hình hiện tại")
                
                # Nút bắt đầu huấn luyện
                if st.button("🚀 Bắt đầu huấn luyện", key="train_button", disabled=st.session_state.get("model_updating", False)):
                    st.session_state.model_updating = True
                    st.session_state.train_running = True
                    
                    # Đảm bảo thư mục mô hình tồn tại
                    model_dir = os.path.join(os.getcwd(), "models")
                    os.makedirs(model_dir, exist_ok=True)
                    
                    base_symbol = symbol.replace("USDT", "") if symbol.endswith("USDT") else symbol

                    # Kiểm tra và tải dữ liệu trước khi huấn luyện
                    if not st.session_state.data_loaded:
                        st.info("⏳ Đang chuẩn bị dữ liệu cho huấn luyện...")
                        try:
                            # Dùng hàm load_and_process_data thay vì update_data
                            df = load_and_process_data(
                                symbol=base_symbol, 
                                timeframe=st.session_state.selected_timeframe,
                                apply_indicators=True,
                                apply_patterns=True,
                                apply_labels=True,
                                horizon=horizon,
                                threshold=threshold/100,
                                force_reload=True
                            )
                            
                            if df is not None and not df.empty:
                                st.session_state.price_data[st.session_state.selected_timeframe] = df
                                st.session_state.data_loaded = True
                                logging.info(f"✅ Đã tải dữ liệu cho {symbol}@{st.session_state.selected_timeframe} "
                                           f"({len(df)} dòng, {len(df.columns)} cột)")
                                
                                # Kiểm tra cấu trúc dữ liệu cần thiết cho huấn luyện
                                if "timestamp" not in df.columns:
                                    st.error("❌ Dữ liệu thiếu cột timestamp. Không thể huấn luyện.")
                                    st.session_state.model_updating = False
                                    st.session_state.train_running = False
                                    return
                                if "spike" not in df.columns:
                                    st.error("❌ Dữ liệu thiếu cột spike (gán nhãn). Không thể huấn luyện.")
                                    st.session_state.model_updating = False
                                    st.session_state.train_running = False
                                    return
                            else:
                                st.error("❌ Không thể tải dữ liệu. Vui lòng thử lại.")
                                st.session_state.model_updating = False
                                st.session_state.train_running = False
                                return
                        except Exception as e:
                            st.error(f"❌ Lỗi khi tải dữ liệu: {e}")
                            logging.error(f"Lỗi tải dữ liệu: {e}", exc_info=True)
                            st.session_state.model_updating = False
                            st.session_state.train_running = False
                            return
                    
                    # Khởi chạy thread huấn luyện
                    t = threading.Thread(
                        target=train_model_thread,
                        args=(symbol, st.session_state.selected_timeframe, epochs, threshold/100, batch_size, horizon, use_gpu, use_enhanced),
                        daemon=True
                    )
                    st.session_state["train_thread"] = t
                    t.start()
                    
                    # Hiển thị thông báo
                    st.info(f"🚀 Đã bắt đầu huấn luyện mô hình cho {symbol}@{st.session_state.selected_timeframe}...")

            elif choice == "Gửi Zalo":
                st.header("📱 Gửi thông báo Zalo - Enhanced Version")
                zalo = ZaloSender()

                # Log dữ liệu Gửi Zalo  
                zalo_data = {
                    "has_access_token": bool(zalo.access_token),
                    "contacts_count": len(zalo.contacts),
                    "contacts": [{"name": c.get('name', 'Unknown'), "phone": c.get('phone', 'Unknown')} for c in zalo.contacts],
                    "zalo_pkce_available": "zalo_pkce" in st.session_state,
                    "symbol": st.session_state.get("symbol"),
                    "prediction": st.session_state.get("prediction"),
                    "train_results_count": len(st.session_state.get("train_results", []))
                }
                log_function_data("ZALO_SENDER", zalo_data)

                # Display connection status and stats
                if zalo.access_token:
                    # Validate token and show profile
                    is_valid, validation_msg = zalo.validate_access_token()
                    if is_valid:
                        st.success(f"✅ Kết nối Zalo thành công: {validation_msg}")
                        
                        # Show OA profile info
                        profile = zalo.get_oa_profile()
                        if "error" not in profile:
                            st.info(f"📱 OA: {profile.get('name', 'Unknown')} | Followers: {profile.get('followers_count', 'N/A')}")
                        
                        # Contact statistics
                        stats = zalo.get_contact_stats()
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Tổng Contacts", stats["total_contacts"])
                        with col2:
                            st.metric("Hoạt động 7 ngày", stats["active_last_7_days"])
                        with col3:
                            st.metric("Contacts thường xuyên", stats["frequent_contacts"])
                        with col4:
                            st.metric("Broadcast cuối", stats["last_broadcast"][:10] if stats["last_broadcast"] != "Never" else "Never")
                    else:
                        st.error(f"❌ Token không hợp lệ: {validation_msg}")
                        st.info("🔄 Vui lòng xác thực lại Zalo OA")
                        zalo.access_token = None  # Reset token

                # 1) XÁC THỰC ZALO OA nếu chưa có access_token
                if not zalo.access_token:
                    st.subheader("🔐 Xác thực Zalo OA")
                    # Sinh PKCE và lưu vào session
                    if "zalo_pkce" not in st.session_state:
                        cv, cc, stt = zalo.generate_pkce()
                        st.session_state.zalo_pkce = {"verifier": cv, "challenge": cc, "state": stt}
                    cv = st.session_state.zalo_pkce["verifier"]
                    cc = st.session_state.zalo_pkce["challenge"]
                    stt = st.session_state.zalo_pkce["state"]

                    # Hiển thị QR + link
                    auth_url = zalo.build_auth_url(cc, stt)
                    qr_png = zalo.get_qr_code_bytes(auth_url)
                    st.image(qr_png, caption="Scan QR để xác thực", use_container_width=True)
                    st.markdown(f"[Mở link xác thực trực tiếp]({auth_url})", unsafe_allow_html=True)

                    # Nhập code và xác thực
                    code = st.text_input("Nhập code từ callback URL", key="zalo_auth_code")
                   
                    if st.button("🔄 Xác thực", key="zalo_auth_button"):
                        ok, msg = zalo.exchange_code_for_token(code, cv)
                        if ok:
                            st.success(msg)
                            try:
                                try:
                                    st.rerun()
                                except AttributeError:
                                    st.experimental_rerun()
                            except AttributeError:
                                st.experimental_rerun()
                        else:
                            st.error(msg)
                    # Dừng ở đây cho đến khi auth xong
                    st.stop()

                # 2) Quản lý danh bạ
                st.subheader("Danh sách liên hệ")
                contacts = zalo.contacts
                if contacts:
                    for c in contacts:
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.markdown(f"**{c['name'] or 'Không tên'}**  –  {c['phone']}")
                        with col2:
                            if c.get("last_sent"):
                                st.markdown(f"Lần cuối: {c['last_sent']}")
                        with col3:
                            if st.button("Xóa", key=f"zalo_del_{c['phone']}"):
                                zalo.remove_contact(c["phone"])
                                st.success("Đã xóa liên hệ")
                                st.rerun()
                else:
                    st.info("Chưa có liên hệ nào")

                st.subheader("Thêm liên hệ mới")
                ph = st.text_input("Số điện thoại", key="zalo_new_phone")
                nm = st.text_input("Tên (tùy chọn)", key="zalo_new_name")
                if st.button("Thêm", key="zalo_add_contact"):
                    if ph:
                        ok, msg = zalo.add_contact(ph, nm)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("Vui lòng nhập số điện thoại")

                # 3) Enhanced messaging features
                st.markdown("---")
                st.subheader("📬 Tính năng gửi tin nhắn nâng cao")
                
                # Message type selection
                message_type = st.selectbox(
                    "Loại tin nhắn",
                    ["Dự đoán AI", "Broadcast", "Tin nhắn tùy chỉnh", "Rich Message", "Price Alert"]
                )
                
                if message_type == "Dự đoán AI":
                    pred = st.session_state.get("prediction")
                    if not isinstance(pred, dict) or "Entry" not in pred:
                        st.info("⏳ Chưa có dự đoán, hãy chạy 'Cập nhật dự đoán' trước")
                    elif not contacts:
                        st.warning("⚠️ Chưa có liên hệ nào để gửi")
                    else:
                        phones = [c["phone"] for c in contacts]
                        selected = st.multiselect(
                            "Chọn người nhận",
                            options=phones,
                            default=phones[:3] if len(phones) > 3 else phones  # Limit default selection
                        )

                        # Enhanced message with chart links
                        include_chart = st.checkbox("Bao gồm link chart TradingView", value=True)
                        msg_text = zalo.compose_message_from_predictions([pred], include_chart_link=include_chart)
                        st.text_area("Nội dung tin nhắn", value=msg_text, height=200, key="ai_pred_msg")

                        if st.button("🚀 Gửi dự đoán AI", key="zalo_send_ai"):
                            with st.spinner("Đang gửi tin nhắn..."):
                                success_count = 0
                                errors = []
                                for phone in selected:
                                    ok, rsp = zalo.send_text_message(phone, msg_text)
                                    if ok:
                                        success_count += 1
                                    else:
                                        errors.append(f"{phone}: {rsp}")
                                
                                # Show results
                                if success_count > 0:
                                    st.success(f"✅ Đã gửi thành công đến {success_count}/{len(selected)} liên hệ")
                                if errors:
                                    st.error("❌ Lỗi gửi:\n" + "\n".join(errors))

                elif message_type == "Broadcast":
                    st.subheader("📢 Gửi tin nhắn tập thể")
                    
                    # Target group selection
                    target_group = st.selectbox(
                        "Nhóm đối tượng",
                        ["all", "recent", "frequent"],
                        format_func=lambda x: {
                            "all": "Tất cả contacts", 
                            "recent": "Hoạt động trong 7 ngày", 
                            "frequent": "Liên hệ thường xuyên (>5 lần)"
                        }[x]
                    )
                    
                    # Show target count
                    target_contacts = zalo._filter_contacts_by_group(target_group)
                    st.info(f"📊 Sẽ gửi đến {len(target_contacts)} contacts")
                    
                    broadcast_msg = st.text_area(
                        "Nội dung broadcast", 
                        placeholder="Nhập tin nhắn cần gửi đến tất cả...",
                        height=150
                    )
                    
                    if st.button("📢 Gửi Broadcast", key="zalo_broadcast") and broadcast_msg:
                        with st.spinner("Đang gửi broadcast..."):
                            results = zalo.broadcast_message(broadcast_msg, target_group)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.success(f"✅ Thành công: {results['success']}")
                            with col2:
                                st.error(f"❌ Thất bại: {results['failed']}")
                                
                            if results['recipients']:
                                st.info(f"📤 Đã gửi đến: {', '.join(results['recipients'][:5])}" + 
                                       (f" và {len(results['recipients'])-5} khác..." if len(results['recipients']) > 5 else ""))

                elif message_type == "Rich Message":
                    st.subheader("✨ Tin nhắn phong phú")
                    
                    if contacts:
                        phone = st.selectbox("Chọn người nhận", [c["phone"] for c in contacts])
                        title = st.text_input("Tiêu đề", value=f"Tín hiệu {st.session_state.get('symbol', 'BTC')}")
                        subtitle = st.text_input("Phụ đề", value="Cập nhật từ AI Crypto Bot")
                        
                        if st.button("📱 Gửi Rich Message", key="zalo_rich"):
                            ok, msg = zalo.send_rich_message(phone, title, subtitle)
                            if ok:
                                st.success(f"✅ {msg}")
                            else:
                                st.error(f"❌ {msg}")
                    else:
                        st.warning("⚠️ Chưa có liên hệ nào")

                elif message_type == "Price Alert":
                    st.subheader("🚨 Cảnh báo giá")
                    
                    symbol = st.session_state.get("symbol", "BTC")
                    
                    # Simulate price alert (in real app, this would come from price monitoring)
                    col1, col2 = st.columns(2)
                    with col1:
                        current_price = st.number_input("Giá hiện tại", value=50000.0, step=100.0)
                    with col2:
                        previous_price = st.number_input("Giá trước đó", value=48000.0, step=100.0)
                        
                    alert_msg = zalo.create_alert_from_price_change(symbol, current_price, previous_price)
                    
                    if alert_msg:
                        st.text_area("Alert sẽ được gửi", value=alert_msg, height=120)
                        
                        if st.button("🚨 Gửi Price Alert", key="zalo_alert") and contacts:
                            # Send to recent contacts only
                            recent_contacts = zalo._filter_contacts_by_group("recent")
                            if recent_contacts:
                                results = zalo.broadcast_message(alert_msg, "recent")
                                st.success(f"✅ Đã gửi alert đến {results['success']} liên hệ hoạt động gần đây")
                            else:
                                st.info("📭 Không có liên hệ hoạt động gần đây để gửi alert")
                    else:
                        st.info("💡 Thay đổi giá < 2%, không tạo alert")

                elif message_type == "Tin nhắn tùy chỉnh":
                    st.subheader("✏️ Tin nhắn tùy chỉnh")
                    
                    if contacts:
                        phones = [c["phone"] for c in contacts]
                        selected = st.multiselect("Chọn người nhận", options=phones)
                        
                        custom_msg = st.text_area(
                            "Nội dung tin nhắn", 
                            placeholder="Nhập tin nhắn tùy chỉnh...",
                            height=150
                        )
                        
                        if st.button("📤 Gửi tin nhắn tùy chỉnh", key="zalo_custom") and custom_msg and selected:
                            with st.spinner("Đang gửi..."):
                                success_count = 0
                                for phone in selected:
                                    ok, _ = zalo.send_text_message(phone, custom_msg)
                                    if ok:
                                        success_count += 1
                                
                                st.success(f"✅ Đã gửi thành công đến {success_count}/{len(selected)} liên hệ")
                    else:
                        st.warning("⚠️ Chưa có liên hệ nào")
                        
                # Contact management (moved to bottom)
                st.markdown("---")
                st.subheader("👥 Quản lý liên hệ")
                contacts = zalo.contacts
                if contacts:
                    # Enhanced contact display with stats
                    contact_data = []
                    for c in contacts:
                        last_sent = c.get("last_sent", "Chưa gửi")
                        if last_sent and last_sent != "Chưa gửi":
                            try:
                                last_sent = datetime.fromisoformat(last_sent.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M")
                            except:
                                pass
                        contact_data.append({
                            "Tên": c.get("name", "Chưa đặt tên"),
                            "Số điện thoại": c["phone"],
                            "Lần gửi cuối": last_sent,
                            "Số lần gửi": c.get("send_count", 0)
                        })
                    
                    df_contacts = pd.DataFrame(contact_data)
                    st.dataframe(df_contacts, use_container_width=True)
                    
                    # Quick remove contact
                    phone_to_remove = st.selectbox("Xóa liên hệ", [""] + [c["phone"] for c in contacts])
                    if st.button("🗑️ Xóa liên hệ", key="remove_contact") and phone_to_remove:
                        ok, msg = zalo.remove_contact(phone_to_remove)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                else:
                    st.info("Chưa có liên hệ nào")

                # Add new contact (enhanced)
                st.subheader("➕ Thêm liên hệ mới")
                col1, col2 = st.columns(2)
                with col1:
                    ph = st.text_input("Số điện thoại", key="zalo_new_phone", placeholder="Ví dụ: 0987654321")
                with col2:
                    nm = st.text_input("Tên (tùy chọn)", key="zalo_new_name", placeholder="Ví dụ: Nguyễn Văn A")
                    
                if st.button("➕ Thêm liên hệ", key="zalo_add_contact"):
                    if ph:
                        # Validate phone format (basic)
                        if ph.isdigit() and len(ph) >= 10:
                            ok, msg = zalo.add_contact(ph, nm or f"Contact_{ph[-4:]}")
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                        else:
                            st.error("❌ Số điện thoại không hợp lệ (chỉ số, ít nhất 10 chữ số)")
                    else:
                        st.warning("Vui lòng nhập số điện thoại")

            elif choice == "Cấu hình":
                st.header("⚙️ Cấu hình")
                
                # Log dữ liệu Cấu hình
                config_data = {
                    "current_symbol": st.session_state.get("symbol"),
                    "current_timeframe": st.session_state.get("selected_timeframe"),
                    "current_max_news": st.session_state.get("max_news"),
                    "data_loaded": st.session_state.get("data_loaded"),
                    "price_data_keys": list(st.session_state.get("price_data", {}).keys()),
                    "config_file_exists": os.path.exists("data/config.json"),
                    "raw_data_exists": os.path.exists("data/raw"),
                    "processed_data_exists": os.path.exists("data/processed")
                }
                log_function_data("CONFIGURATION", config_data)
                
                # Cấu hình chung
                st.subheader("Cấu hình chung")
                col1, col2 = st.columns(2)
                with col1:
                    update_interval = st.slider(
                        "Thời gian cập nhật (phút)",
                        min_value=1,
                        max_value=60,
                        value=5
                    )
                with col2:
                    max_news = st.slider(
                        "Số tin tức tối đa",
                        min_value=10,
                        max_value=100,
                        value=50
                    )
                
                # Cấu hình mô hình
                st.subheader("Cấu hình mô hình")
                col1, col2 = st.columns(2)
                with col1:
                    min_confidence = st.slider(
                        "Ngưỡng tin cậy tối thiểu (%)",
                        min_value=50,
                        max_value=90,
                        value=70
                    )
                with col2:
                    prediction_horizon = st.slider(
                        "Tầm nhìn dự báo (nến)",
                        min_value=1,
                        max_value=24,
                        value=12
                    )
                
                # Lưu cấu hình
                if st.button("💾 Lưu cấu hình"):
                    config = {
                        "update_interval": update_interval,
                        "max_news": max_news,
                        "min_confidence": min_confidence,
                        "prediction_horizon": prediction_horizon
                    }
                    with open("data/config.json", "w") as f:
                        json.dump(config, f)
                    st.session_state.max_news = max_news
                    st.success("Đã lưu cấu hình!")
                
                # Xóa dữ liệu
                st.subheader("Quản lý dữ liệu")
                if st.button("🗑️ Xóa dữ liệu cũ"):
                    if os.path.exists("data/raw"):
                        shutil.rmtree("data/raw")
                        os.makedirs("data/raw")
                    if os.path.exists("data/processed"):
                        shutil.rmtree("data/processed")
                        os.makedirs("data/processed")
                    st.success("Đã xóa dữ liệu cũ!")
                    st.session_state.data_loaded = False
                    update_data(st.session_state.symbol, st.session_state.selected_timeframe)

            elif choice == "Dự báo dài hạn":
                st.header("📈 Dự báo dài hạn")
                symbol = st.session_state.symbol

                # Log dữ liệu Dự báo dài hạn
                forecast_data = {
                    "symbol": symbol,
                    "selected_timeframe": st.session_state.get("selected_timeframe"),
                    "timeframes_available": TIMEFRAMES,
                    "smallest_timeframe": TIMEFRAMES[3] if len(TIMEFRAMES) > 3 else None,
                    "price_data_keys": list(st.session_state.get("price_data", {}).keys()),
                    "train_results_available": bool(st.session_state.get("train_results")),
                    "model_files_exist": any(os.path.exists(os.path.join(MODEL_DIR, f"{symbol}_{tf}.pkl.gz")) for tf in TIMEFRAMES),
                    "data_loaded": st.session_state.get("data_loaded")
                }
                log_function_data("LONG_TERM_FORECAST", forecast_data)

                if st.button("⚡ Cập nhật dự báo dài hạn"):
                    results = forecast_long_term(symbol)  # chỉ chạy khi bấm nút
                    for tf, res in results.items():
                        st.markdown(f"**{tf}:** {res}")

                st.header("🤖 Reinforcement Learning Agent")
                symbol = st.session_state.get("symbol", DEFAULT_symbol)
                # ─── Lấy DataFrame khung nhỏ nhất (5m) trực tiếp ───────────────
                # Giả sử TIMEFRAMES = ["5m","15m","1h","4h"] đã định nghĩa ở trên
                
                tf_smallest = TIMEFRAMES[3]
                df_small = update_data(symbol, tf_smallest)
                if df_small is None or df_small.empty:
                    st.error(f"❌ Không thể tải dữ liệu khung {tf_smallest}")
                    return
                prices = df_small["close"].tolist()                if st.button("🚀 Train PPO Agent"):
                    # Use Training class method instead of standalone function
                    training_instance = Training()
                    model = training_instance.train_trading_agent(prices, timesteps=1000000)
                    st.success("Đã train xong RL Agent")
                    
                if st.button("🚀 Chạy Trainer Loop ngay"):
                    try:
                        run_training_loop(st.session_state.symbol, st.session_state.selected_timeframe)  # gọi hàm tự động huấn luyện hàng tuần
                        st.success("✅ Đã hoàn thành chạy Trainer Loop.")
                    except Exception as e:
                        st.error(f"❌ Lỗi khi chạy Trainer Loop: {e}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    finally:
        resource_optimizer.cleanup_resources()

# Khởi chạy ứng dụng
if __name__ == "__main__":
    main()
