from __future__ import annotations

import re
import datetime as dt
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import yfinance as yf

from src.config import NEWS_SOURCE, MAX_HEADLINES_PER_TICKER


_GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)

# Output schema

_COLUMNS = ["ticker", "ts", "headline", "source", "link"]

# helper functions

def _normalize_title(text: str) -> str:
    """Collapse weird whitespace and trim."""
    return re.sub(r"\s+", " ", (text or "")).strip()

def _to_utc_datetime(t) -> dt.datetime | None:
    """Convert a feedparser time-struct to a tz-aware UTC datetime."""
    if t is None:
        return None
    return dt.datetime(*t[:6], tzinfo=dt.timezone.utc)

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLUMNS)

# Google news rss

def fetch_headlines_google(ticker: str, days: int) -> pd.DataFrame:
    """Fetch recent headlines for a ticker using Google News RSS."""
    q = quote_plus(f"{ticker} when:{days}d")
    url = _GOOGLE_NEWS_RSS.format(query=q)
    
    feed = feedparser.parse(url)
    rows = []
    for e in feed.entries:
        title = _normalize_title(getattr(e, "title", ""))
        if not title:
            continue
        
        # pick published or updated time; convert to UTC
        pub = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
        ts = _to_utc_datetime(pub)
        if ts is None:
            continue
        
        link = getattr(e, "link", "") or ""
        src = ""
        src_obj = getattr(e, "source", None)
        if isinstance(src_obj, dict):
            src = src_obj.get("title", "") or src
            
        rows.append(
            {"ticker": ticker.upper(), "ts": ts, "headline": title, "source": src, "link": link}
        )
    
    if not rows:
        return _empty_df()
    
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["headline"]).sort_values("ts").reset_index(drop=True)
    
    if MAX_HEADLINES_PER_TICKER:
    # keep most recent N, then put back in chronological order
        df = (
            df.sort_values("ts", ascending=False)
            .head(MAX_HEADLINES_PER_TICKER)
            .sort_values("ts")
            .reset_index(drop=True)
        )
    
    return df[_COLUMNS]

# yfinance

def fetch_headlines_yfinance(ticker: str, days: int) -> pd.DataFrame:
    """Fetch recent headlines via yfinance's Ticker.news"""
    try:
        items = (yf.Ticker(ticker).news or [])
    except Exception:
        items = []
        
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    rows = []
    for it in items:
        title = _normalize_title(it.get("title", ""))
        if not title:
            continue
        
        # providerPublishTime is epoch seconds
        ts_raw = it.get("providerPublishTime")
        if ts_raw is None:
            continue
        ts = dt.datetime.fromtimestamp(int(ts_raw), tz=dt.timezone.utc)
        if ts < cutoff:
            continue
        
        rows.append(
            {"ticker": ticker.upper(), 
             "ts": ts, "headline": title, 
             "source": it.get("publisher", "") or "", 
             "link": it.get("link", "") or ""}
        )
    
    if not rows:
        return _empty_df()
    
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["headline"]).sort_values("ts").reset_index(drop=True)
    
    if MAX_HEADLINES_PER_TICKER:
        df = (
            df.sort_values("ts", ascending=False)
            .head(MAX_HEADLINES_PER_TICKER)
            .sort_values("ts")
            .reset_index(drop=True)
        )
    
    return df[_COLUMNS]

# public api

def fetch_headlines(ticker: str, days: int, source: str | None = None) -> pd.DataFrame:
    """Fetch headlines for one ticker using the chosen source."""
    use = (source or NEWS_SOURCE or "auto").lower()
    if use == "google":
        return fetch_headlines_google(ticker, days)
    if use == "yfinance":
        return fetch_headlines_yfinance(ticker, days)
    
    # auto: try google first, then yfinance if empty
    df = fetch_headlines_google(ticker, days)
    if df.empty:
        return fetch_headlines_yfinance(ticker, days)
    return df

def fetch_all_headlines(tickers: list[str], days: int, source: str | None = None) -> pd.DataFrame:
    """Fetch and concatenate headlines for multiple tickers."""
    frames = [fetch_headlines(t, days, source=source) for t in tickers]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty_df()
    
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["headline"]).sort_values(["ticker","ts"]).reset_index(drop=True)
    return out[_COLUMNS]