import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import time

# Configure page
st.set_page_config(
    page_title="Enhanced Stock Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .positive { 
        color: #00C851; 
        font-weight: bold;
    }
    
    .negative { 
        color: #ff4444; 
        font-weight: bold;
    }
    
    .neutral { 
        color: #33b5e5; 
        font-weight: bold;
    }
    
    .live-indicator {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        animation: pulse 2s infinite;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(ticker, period="1y"):
    """Enhanced stock data fetching with error handling"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        
        if hist.empty:
            return None, None, "No historical data found for this ticker"
        
        return stock, hist, None
    except Exception as e:
        return None, None, f"Error fetching data: {str(e)}"

@st.cache_data(ttl=60)  # Cache for 1 minute for real-time feel
def get_realtime_data(ticker):
    """Get latest market data"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d", interval="1h")  # Last 5 days hourly data
        return data
    except Exception as e:
        st.warning(f"Real-time data unavailable: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(data):
    """Calculate comprehensive technical indicators"""
    if data.empty:
        return data
    
    # Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    return data

def create_enhanced_candlestick_chart(data, title, show_volume=True):
    """Create an enhanced candlestick chart"""
    if data.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    # Create subplots
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=[title, "Volume"]
        )
    else:
        fig = go.Figure()
    
    # Candlestick chart
    candlestick = go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350'
    )
    
    if show_volume:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Add moving averages if available
    if 'SMA_20' in data.columns and not data['SMA_20'].isna().all():
        ma_trace = go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
        )
        if show_volume:
            fig.add_trace(ma_trace, row=1, col=1)
        else:
            fig.add_trace(ma_trace)
    
    if 'SMA_50' in data.columns and not data['SMA_50'].isna().all():
        ma_trace = go.Scatter(
            x=data.index,
            y=data['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=2)
        )
        if show_volume:
            fig.add_trace(ma_trace, row=1, col=1)
        else:
            fig.add_trace(ma_trace)
    
    # Volume chart
    if show_volume:
        colors = ['#ef5350' if close < open else '#26a69a' 
                  for close, open in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        template="plotly_white",
        height=600 if show_volume else 400,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    if not show_volume:
        fig.update_layout(title=title)
    
    return fig

def create_technical_indicators_dashboard(data):
    """Create comprehensive technical indicators dashboard"""
    if data.empty or 'RSI' not in data.columns:
        return go.Figure().add_annotation(text="Insufficient data for technical analysis", showarrow=False)
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['RSI', 'MACD', 'Bollinger Bands', 'Volume Analysis', 'Price vs Moving Averages', 'Support/Resistance'],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
        row=1, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')),
        row=1, col=2
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='red', dash='dash')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='black')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', line=dict(color='red', dash='dash')),
        row=2, col=1
    )
    
    # Volume Analysis
    volume_ma = data['Volume'].rolling(window=20).mean()
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', opacity=0.7),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=volume_ma, name='Volume MA', line=dict(color='orange')),
        row=2, col=2
    )
    
    # Price vs Moving Averages
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='black')),
        row=3, col=1
    )
    if 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')),
            row=3, col=1
        )
    if 'SMA_50' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='blue')),
            row=3, col=1
        )
    
    # Support/Resistance (simplified)
    recent_high = data['High'].rolling(window=20).max()
    recent_low = data['Low'].rolling(window=20).min()
    fig.add_trace(
        go.Scatter(x=data.index, y=recent_high, name='Resistance', line=dict(color='red', dash='dot')),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=recent_low, name='Support', line=dict(color='green', dash='dot')),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='black')),
        row=3, col=2
    )
    
    fig.update_layout(
        height=900,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

def display_key_metrics(stock, data):
    """Display key financial metrics in an attractive format"""
    info = stock.info
    
    # Live indicator
    st.markdown('<div class="live-indicator">🔴 LIVE DATA</div>', unsafe_allow_html=True)
    
    # Current price info
    current_price = data['Close'].iloc[-1] if not data.empty else info.get('currentPrice', 0)
    prev_close = info.get('regularMarketPreviousClose', current_price)
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color = "positive" if change >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Current Price</h4>
            <h2>${current_price:.2f}</h2>
            <p class="{color}">
                {'▲' if change >= 0 else '▼'} ${abs(change):.2f} ({change_pct:+.2f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        market_cap = info.get('marketCap', 0)
        if market_cap:
            if market_cap >= 1e12:
                cap_str = f"${market_cap/1e12:.1f}T"
            elif market_cap >= 1e9:
                cap_str = f"${market_cap/1e9:.1f}B"
            else:
                cap_str = f"${market_cap/1e6:.1f}M"
        else:
            cap_str = "N/A"
            
        st.markdown(f"""
        <div class="metric-card">
            <h4>Market Cap</h4>
            <h2>{cap_str}</h2>
            <p>Total Value</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pe_ratio = info.get('trailingPE', 'N/A')
        if isinstance(pe_ratio, (int, float)):
            pe_str = f"{pe_ratio:.1f}"
        else:
            pe_str = "N/A"
            
        st.markdown(f"""
        <div class="metric-card">
            <h4>P/E Ratio</h4>
            <h2>{pe_str}</h2>
            <p>Price/Earnings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        volume = data['Volume'].iloc[-1] if not data.empty else 0
        avg_volume = info.get('averageVolume', 1)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Volume</h4>
            <h2>{volume:,.0f}</h2>
            <p>{volume_ratio:.1f}x Average</p>
        </div>
        """, unsafe_allow_html=True)

def display_financial_summary(stock):
    """Display comprehensive financial summary"""
    info = stock.info
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Valuation Metrics</h4>
        """, unsafe_allow_html=True)
        
        metrics = {
            "P/E Ratio": info.get('trailingPE', 'N/A'),
            "Forward P/E": info.get('forwardPE', 'N/A'),
            "PEG Ratio": info.get('pegRatio', 'N/A'),
            "Price/Book": info.get('priceToBook', 'N/A'),
            "Price/Sales": info.get('priceToSalesTrailing12Months', 'N/A')
        }
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and value != 0:
                st.write(f"**{metric}:** {value:.2f}")
            else:
                st.write(f"**{metric}:** N/A")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>💰 Profitability</h4>
        """, unsafe_allow_html=True)
        
        profitability = {
            "Profit Margin": info.get('profitMargins', 0),
            "Operating Margin": info.get('operatingMargins', 0),
            "ROE": info.get('returnOnEquity', 0),
            "ROA": info.get('returnOnAssets', 0),
            "Revenue Growth": info.get('revenueGrowth', 0)
        }
        
        for metric, value in profitability.items():
            if isinstance(value, (int, float)) and value != 0:
                st.write(f"**{metric}:** {value*100:.1f}%")
            else:
                st.write(f"**{metric}:** N/A")
        
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="header-gradient">🚀 Enhanced Stock Market Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("*Advanced stock analysis with real-time data, technical indicators, and comprehensive financial metrics*")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Stock input with validation
        ticker = st.text_input(
            "📊 Enter Stock Symbol", 
            value="AAPL",
            help="Enter a valid stock ticker (e.g., AAPL, GOOGL, TSLA)",
            max_chars=10
        ).upper().strip()
        
        # Period selection
        period_options = {
            "1 Month": "1mo",
            "3 Months": "3mo", 
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y"
        }
        
        selected_period = st.selectbox(
            "📅 Time Period", 
            list(period_options.keys()),
            index=3
        )
        period = period_options[selected_period]
        
        # Auto refresh
        auto_refresh = st.checkbox("🔄 Auto Refresh (30s)", value=False)
        
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        # Quick links
        st.markdown("---")
        st.markdown("**📚 Quick Access:**")
        st.markdown("• [SEC Filings](https://www.sec.gov/edgar)")
        st.markdown("• [Yahoo Finance](https://finance.yahoo.com)")
        st.markdown("• [Market Watch](https://www.marketwatch.com)")
    
    if not ticker:
        st.warning("⚠️ Please enter a stock ticker to begin analysis")
        st.stop()
    
    # Auto refresh logic
    if auto_refresh:
        if 'refresh_time' not in st.session_state:
            st.session_state.refresh_time = time.time()
        
        if time.time() - st.session_state.refresh_time > 30:
            st.session_state.refresh_time = time.time()
            st.cache_data.clear()
            st.rerun()
    
    # Main content
    try:
        with st.spinner(f"🔍 Analyzing {ticker}..."):
            stock, data, error = get_stock_data(ticker, period)
            
            if error:
                st.error(f"❌ {error}")
                st.info("💡 **Suggestions:**")
                st.info("• Check if the ticker symbol is correct")
                st.info("• Try major exchange symbols (NYSE, NASDAQ)")
                st.info("• Ensure the stock is actively traded")
                st.stop()
            
            # Get real-time data
            realtime_data = get_realtime_data(ticker)
            if not realtime_data.empty:
                display_data = realtime_data
            else:
                display_data = data
            
            # Calculate technical indicators
            data_with_indicators = calculate_technical_indicators(data.copy())
        
        # Company header
        info = stock.info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'Unknown')
        
        st.markdown(f"## 🏢 {company_name} ({ticker})")
        st.markdown(f"**Sector:** {sector} | **Industry:** {info.get('industry', 'Unknown')}")
        
        # Key metrics section
        st.markdown("---")
        st.markdown("### 📊 Key Metrics")
        display_key_metrics(stock, display_data)
        
        # Main analysis tabs
        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Price Analysis", 
            "🔬 Technical Analysis", 
            "📋 Financial Overview",
            "📰 News & Events",
            "📊 Advanced Data"
        ])
        
        with tab1:
            st.subheader("Price Action & Volume Analysis")
            
            # Main price chart
            price_chart = create_enhanced_candlestick_chart(
                data, f"{ticker} Stock Price - {selected_period}", show_volume=True
            )
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Price statistics
            col1, col2, col3, col4 = st.columns(4)
            
            if not data.empty:
                with col1:
                    high_52w = info.get('fiftyTwoWeekHigh', data['High'].max())
                    st.metric("52W High", f"${high_52w:.2f}")
                
                with col2:
                    low_52w = info.get('fiftyTwoWeekLow', data['Low'].min())
                    st.metric("52W Low", f"${low_52w:.2f}")
                
                with col3:
                    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                    st.metric("Volatility (1Y)", f"{volatility:.1f}%")
                
                with col4:
                    avg_volume = data['Volume'].mean()
                    st.metric("Avg Volume", f"{avg_volume:,.0f}")
        
        with tab2:
            st.subheader("Technical Analysis Dashboard")
            
            if 'RSI' in data_with_indicators.columns:
                # Technical indicators chart
                tech_chart = create_technical_indicators_dashboard(data_with_indicators)
                st.plotly_chart(tech_chart, use_container_width=True)
                
                # Technical summary
                st.markdown("#### 📋 Technical Summary")
                col1, col2, col3 = st.columns(3)
                
                latest_data = data_with_indicators.iloc[-1]
                
                with col1:
                    rsi = latest_data.get('RSI', None)
                    if pd.notna(rsi):
                        if rsi > 70:
                            rsi_signal = "🔴 Overbought"
                        elif rsi < 30:
                            rsi_signal = "🟢 Oversold"  
                        else:
                            rsi_signal = "🟡 Neutral"
                        st.metric("RSI Signal", rsi_signal, f"{rsi:.1f}")
                    else:
                        st.metric("RSI Signal", "N/A", "Insufficient data")
                
                with col2:
                    macd = latest_data.get('MACD', None)
                    macd_signal = latest_data.get('MACD_Signal', None)
                    if pd.notna(macd) and pd.notna(macd_signal):
                        trend = "🟢 Bullish" if macd > macd_signal else "🔴 Bearish"
                        st.metric("MACD Trend", trend, f"{macd - macd_signal:.4f}")
                    else:
                        st.metric("MACD Trend", "N/A", "Insufficient data")
                
                with col3:
                    bb_upper = latest_data.get('BB_Upper', None)
                    bb_lower = latest_data.get('BB_Lower', None)
                    close_price = latest_data.get('Close', None)
                    
                    if all(pd.notna(val) for val in [bb_upper, bb_lower, close_price]):
                        if close_price > bb_upper:
                            bb_signal = "🔴 Above Upper Band"
                        elif close_price < bb_lower:
                            bb_signal = "🟢 Below Lower Band"
                        else:
                            bb_signal = "🟡 Within Bands"
                        st.metric("Bollinger Bands", bb_signal)
                    else:
                        st.metric("Bollinger Bands", "N/A")
            else:
                st.warning("⚠️ Insufficient data for technical analysis. Try a longer time period.")
        
        with tab3:
            st.subheader("Financial Overview")
            display_financial_summary(stock)
            
            # Financial statements
            st.markdown("#### 📊 Financial Statements")
            statement_tabs = st.tabs(["Balance Sheet", "Income Statement", "Cash Flow"])
            
            with statement_tabs[0]:
                try:
                    balance_sheet = stock.balance_sheet
                    if not balance_sheet.empty:
                        st.dataframe(balance_sheet.head(10), use_container_width=True)
                    else:
                        st.info("Balance sheet data not available")
                except:
                    st.info("Balance sheet data not available")
            
            with statement_tabs[1]:
                try:
                    financials = stock.financials
                    if not financials.empty:
                        st.dataframe(financials.head(10), use_container_width=True)
                    else:
                        st.info("Income statement data not available")
                except:
                    st.info("Income statement data not available")
            
            with statement_tabs[2]:
                try:
                    cashflow = stock.cashflow
                    if not cashflow.empty:
                        st.dataframe(cashflow.head(10), use_container_width=True)
                    else:
                        st.info("Cash flow data not available")
                except:
                    st.info("Cash flow data not available")
        
        with tab4:
            st.subheader("Latest News & Market Events")
            
            try:
                news = stock.news
                if news:
                    for i, article in enumerate(news[:8]):  # Show top 8 articles
                        if article.get('title'):
                            with st.container():
                                st.markdown(f"**{article['title']}**")
                                
                                # Publisher and date
                                publisher = article.get('publisher', 'Unknown')
                                if 'providerPublishTime' in article:
                                    try:
                                        pub_time = datetime.fromtimestamp(article['providerPublishTime'])
                                        time_str = pub_time.strftime('%Y-%m-%d %H:%M')
                                        st.caption(f"📰 {publisher} • {time_str}")
                                    except:
                                        st.caption(f"📰 {publisher}")
                                else:
                                    st.caption(f"📰 {publisher}")
                                
                                # Link
                                if article.get('link'):
                                    st.markdown(f"[Read Full Article →]({article['link']})")
                                
                                if i < len(news) - 1:
                                    st.divider()
                else:
                    st.info("📰 No recent news available for this stock")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not fetch news: {e}")
        
        with tab5:
            st.subheader("Advanced Data & Analytics")
            
            # Historical data table
            st.markdown("#### 📈 Historical Price Data")
            display_data_formatted = data.copy()
            
            # Format the data for better display
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in display_data_formatted.columns:
                    display_data_formatted[col] = display_data_formatted[col].round(2)
            
            if 'Volume' in display_data_formatted.columns:
                display_data_formatted['Volume'] = display_data_formatted['Volume'].astype(int)
            
            st.dataframe(
                display_data_formatted.tail(20)[::-1],  # Show last 20 days, most recent first
                use_container_width=True
            )
            
            # Download data
            csv = data.to_csv()
            st.download_button(
                label="📥 Download Historical Data (CSV)",
                data=csv,
                file_name=f"{ticker}_historical_data.csv",
                mime="text/csv"
            )
            
            # Company information
            st.markdown("#### 🏢 Company Information")
            if info.get('longBusinessSummary'):
                with st.expander("📖 Business Summary", expanded=False):
                    st.write(info['longBusinessSummary'])
            
            # Key company details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📍 Company Details:**")
                details = {
                    "Industry": info.get('industry', 'N/A'),
                    "Sector": info.get('sector', 'N/A'),
                    "Country": info.get('country', 'N/A'),
                    "Website": info.get('website', 'N/A'),
                    "Full Time Employees": f"{info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else 'N/A'
                }
                
                for key, value in details.items():
                    if key == "Website" and value != 'N/A':
                        st.write(f"**{key}:** [{value}]({value})")
                    else:
                        st.write(f"**{key}:** {value}")
            
            with col2:
                st.markdown("**💼 Key Officers:**")
                try:
                    officers = stock.info.get('companyOfficers', [])
                    if officers:
                        for officer in officers[:5]:  # Show top 5 officers
                            name = officer.get('name', 'N/A')
                            title = officer.get('title', 'N/A')
                            st.write(f"**{title}:** {name}")
                    else:
                        st.write("Officer information not available")
                except:
                    st.write("Officer information not available")
        
        # Footer with additional info
        st.markdown("---")
        st.markdown("#### ⚠️ Important Disclaimer")
        st.info(
            "📊 **Investment Disclaimer:** This tool is for informational purposes only and should not be considered as financial advice. "
            "Always conduct your own research and consult with qualified financial advisors before making investment decisions. "
            "Past performance does not guarantee future results."
        )
        
        # Performance metrics
        st.markdown("#### 🚀 App Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Data Points", f"{len(data):,}")
        
        with col2:
            last_update = datetime.now().strftime('%H:%M:%S')
            st.metric("🔄 Last Update", last_update)
        
        with col3:
            data_quality = "Excellent" if len(data) > 100 else "Good" if len(data) > 50 else "Limited"
            st.metric("✅ Data Quality", data_quality)
    
    except Exception as e:
        st.error(f"🚨 An unexpected error occurred: {str(e)}")
        st.info("💡 **Troubleshooting Tips:**")
        st.info("• Verify the stock ticker is correct (e.g., AAPL for Apple)")
        st.info("• Check your internet connection")
        st.info("• Try refreshing the page")
        st.info("• Some stocks may have limited data availability")
        
        # Error details for debugging (can be removed in production)
        with st.expander("🔧 Technical Details (for debugging)", expanded=False):
            st.code(f"Error Type: {type(e).__name__}\nError Message: {str(e)}")

if __name__ == "__main__":
    main()
