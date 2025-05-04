import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stock_data')

# Cache for stock data to avoid frequent API calls
stock_data_cache = {}
cache_expiry = 300  # Cache expiry in seconds (5 minutes)
last_fetch_time = {}

def get_stock_data(symbol, period="1mo", interval="1d"):
    """
    Get stock data from Yahoo Finance with caching
    
    Parameters:
    ----------
    symbol : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    period : str, optional (default="1mo")
        Time period to fetch data for (e.g., '1d', '5d', '1mo', '3mo', '1y')
    interval : str, optional (default="1d")
        Data interval (e.g., '1m', '5m', '1h', '1d', '1wk')
    
    Returns:
    -------
    pandas.DataFrame
        Stock price data
    """
    # Check cache first
    cache_key = f"{symbol}_{period}_{interval}"
    current_time = time.time()
    
    if (cache_key in stock_data_cache and 
        cache_key in last_fetch_time and 
        current_time - last_fetch_time[cache_key] < cache_expiry):
        return stock_data_cache[cache_key]
    
    try:
        # Get stock data from Yahoo Finance
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        
        # Basic validation
        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        # Store in cache
        stock_data_cache[cache_key] = data
        last_fetch_time[cache_key] = current_time
        
        return data
    
    except Exception as e:
        # If there's an error with the API, try to use cached data if available
        if cache_key in stock_data_cache:
            return stock_data_cache[cache_key]
        else:
            raise Exception(f"Error fetching stock data for {symbol}: {str(e)}")

def get_stock_info(symbol):
    """
    Get detailed stock information
    
    Parameters:
    ----------
    symbol : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
    -------
    dict
        Stock information
    """
    try:
        # Get stock info
        logger.info(f"Fetching detailed information for {symbol}")
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Extract relevant information
        relevant_info = {
            'symbol': symbol,
            'name': info.get('shortName', 'Unknown'),
            'long_name': info.get('longName', 'Unknown'),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'beta': info.get('beta', 0),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
            'average_volume': info.get('averageVolume', 0),
            'price_to_book': info.get('priceToBook', 0),
            'earnings_growth': info.get('earningsGrowth', 0),
            'revenue_growth': info.get('revenueGrowth', 0),
            'profit_margins': info.get('profitMargins', 0),
            'analyst_target_price': info.get('targetMeanPrice', 0),
            'recommendation': info.get('recommendationKey', 'Unknown'),
            'currency': info.get('currency', 'USD'),
            'website': info.get('website', 'Unknown'),
            'business_summary': info.get('longBusinessSummary', 'No information available')
        }
        
        return relevant_info
    
    except Exception as e:
        logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
        raise Exception(f"Error fetching stock info for {symbol}: {str(e)}")

def get_financial_data(symbol):
    """
    Get financial data for a stock
    
    Parameters:
    ----------
    symbol : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    
    Returns:
    -------
    dict
        Dictionary containing financial data
    """
    try:
        logger.info(f"Fetching financial data for {symbol}")
        stock = yf.Ticker(symbol)
        
        # Get financial data
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cashflow
        
        # Validate data
        if balance_sheet.empty and income_stmt.empty and cash_flow.empty:
            raise ValueError(f"No financial data available for {symbol}")
        
        # Format and extract key metrics
        financial_data = {
            'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
            'income_statement': income_stmt.to_dict() if not income_stmt.empty else {},
            'cash_flow': cash_flow.to_dict() if not cash_flow.empty else {},
        }
        
        # Extract key financial metrics if available
        metrics = {}
        
        if not income_stmt.empty and not income_stmt.empty:
            try:
                latest_year = income_stmt.columns[0]
                metrics['total_revenue'] = income_stmt.loc['Total Revenue', latest_year] if 'Total Revenue' in income_stmt.index else None
                metrics['net_income'] = income_stmt.loc['Net Income', latest_year] if 'Net Income' in income_stmt.index else None
            except (KeyError, IndexError) as e:
                logger.warning(f"Could not extract income statement metrics for {symbol}: {str(e)}")
        
        if not balance_sheet.empty:
            try:
                latest_year = balance_sheet.columns[0]
                metrics['total_assets'] = balance_sheet.loc['Total Assets', latest_year] if 'Total Assets' in balance_sheet.index else None
                metrics['total_liabilities'] = balance_sheet.loc['Total Liabilities Net Minority Interest', latest_year] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
            except (KeyError, IndexError) as e:
                logger.warning(f"Could not extract balance sheet metrics for {symbol}: {str(e)}")
        
        financial_data['key_metrics'] = metrics
        return financial_data
        
    except Exception as e:
        logger.error(f"Error fetching financial data for {symbol}: {str(e)}")
        raise Exception(f"Error fetching financial data for {symbol}: {str(e)}")

def get_news(symbol, limit=5):
    """
    Get latest news for a stock
    
    Parameters:
    ----------
    symbol : str
        Stock symbol (e.g., 'AAPL', 'MSFT')
    limit : int, optional (default=5)
        Maximum number of news items to return
    
    Returns:
    -------
    list
        List of news items
    """
    try:
        logger.info(f"Fetching news for {symbol}")
        stock = yf.Ticker(symbol)
        news = stock.news
        
        if not news:
            return []
        
        # Format news data
        formatted_news = []
        for item in news[:limit]:
            formatted_news.append({
                'title': item.get('title', 'No title'),
                'publisher': item.get('publisher', 'Unknown'),
                'link': item.get('link', '#'),
                'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                'summary': item.get('summary', 'No summary available')
            })
        
        return formatted_news
        
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        return []  # Return empty list on error to avoid breaking the app

def create_interactive_chart(stock_data, title="Stock Price Analysis"):
    """
    Create an interactive stock chart with multiple indicators
    
    Parameters:
    ----------
    stock_data : pandas.DataFrame
        Stock price data from Yahoo Finance
    title : str, optional (default="Stock Price Analysis")
        Chart title
    
    Returns:
    -------
    plotly.graph_objects.Figure
        Interactive stock chart
    """
    try:
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           row_heights=[0.7, 0.3],
                           vertical_spacing=0.03,
                           subplot_titles=('Price', 'Volume'))
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name="Price",
            increasing_line_color='#00cc96', 
            decreasing_line_color='#ef553b'
        ), row=1, col=1)
        
        # Add volume bar chart
        fig.add_trace(go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            name="Volume",
            marker_color='rgba(128, 128, 128, 0.5)'
        ), row=2, col=1)
        
        # Calculate and add 20-day moving average
        if len(stock_data) >= 20:
            ma20 = stock_data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=ma20,
                name="20-day MA",
                line=dict(color='rgba(255, 207, 102, 1)', width=2)
            ), row=1, col=1)
        
        # Calculate and add 50-day moving average
        if len(stock_data) >= 50:
            ma50 = stock_data['Close'].rolling(window=50).mean()
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=ma50,
                name="50-day MA",
                line=dict(color='rgba(120, 99, 255, 1)', width=2)
            ), row=1, col=1)
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            yaxis=dict(
                autorange=True,
                fixedrange=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating interactive chart: {str(e)}")
        # Return a simple line chart as fallback
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock_data.index, 
            y=stock_data['Close'],
            mode='lines',
            name='Close Price'
        ))
        fig.update_layout(title="Stock Price (Simplified View)", template="plotly_dark")
        return fig
