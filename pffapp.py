import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import ML models
from ml_models import train_linear_regression, train_logistic_regression, train_kmeans_clustering

# Set page configuration
st.set_page_config(
    page_title="FinFit - Your Financial Health Dashboard",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'intro'

if 'data' not in st.session_state:
    st.session_state.data = None

if 'model' not in st.session_state:
    st.session_state.model = None
    
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
    
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
    
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
    
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
    
if 'stock_symbol' not in st.session_state:
    st.session_state.stock_symbol = None

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# Helper functions
def load_data(file_path):
    """Load financial profile data from a CSV file"""
    try:
        data = pd.read_csv(file_path)
        
        # Basic validation of the dataset structure
        required_columns = ['budget', 'risk_tolerance', 'investment_horizon', 'investment_type', 'fitness']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
        
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def preprocess_data(data):
    """Preprocess the financial profile data for ML modeling"""
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Fill missing numeric values with median
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill missing categorical values with mode
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Convert categorical variables to numeric
    # One-hot encode categorical columns except the target variable
    categorical_cols_to_encode = [col for col in categorical_columns if col != 'fitness']
    
    if categorical_cols_to_encode:
        df = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)
    
    # Ensure target variable is binary (0 or 1)
    if 'fitness' in df.columns:
        # If fitness is categorical (e.g., 'Fit'/'Unfit'), convert to binary
        if df['fitness'].dtype == 'object':
            fitness_mapping = {'Fit': 1, 'Unfit': 0}
            df['fitness'] = df['fitness'].map(fitness_mapping)
        
        # Ensure fitness is either 0 or 1
        df['fitness'] = df['fitness'].astype(int)
    
    return df

def get_stock_data(symbol, period="1mo", interval="1d"):
    """Get stock data from Yahoo Finance"""
    try:
        # Get stock data from Yahoo Finance
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        
        # Basic validation
        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        return data
    
    except Exception as e:
        raise Exception(f"Error fetching stock data for {symbol}: {str(e)}")

def plot_stock_data(stock_data):
    """Create a candlestick chart for stock price data"""
    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        increasing_line_color='#00cc96',
        decreasing_line_color='#ff4b4b'
    )])
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=stock_data.index,
        y=stock_data['Volume'],
        marker_color='rgba(128, 128, 128, 0.3)',
        name='Volume',
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Stock Price Movement',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        )
    )
    
    return fig

# Navigation functions
def go_to_page(page):
    st.session_state.page = page
    st.rerun()

# Helper functions for UI
def display_success(message):
    st.success(message)
    
def display_error(message):
    st.error(message)

def display_info(message):
    st.info(message)

# Pages
def intro_page():
    # Add one-time intro animation with fitness-themed elements
    if 'show_intro_animation' not in st.session_state:
        st.session_state.show_intro_animation = True
    
    if st.session_state.show_intro_animation:
        st.markdown("""
        <style>
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .animated-title {
            animation: pulse 2s infinite;
            display: inline-block;
        }
        
        .gradient-text {
            background: linear-gradient(90deg, #ff4b4b, #00cc96);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        
        /* Fitness icons animation - appears only once */
        .fitness-icon {
            position: fixed;
            top: -50px;
            z-index: 10;
            animation: icon-fall 4s forwards;
        }
        
        @keyframes icon-fall {
            0% { 
                transform: translateY(-50px) scale(0.8); 
                opacity: 0; 
            }
            30% { 
                opacity: 1; 
            }
            80% {
                opacity: 1;
            }
            100% { 
                transform: translateY(80vh) scale(1); 
                opacity: 0; 
            }
        }
        
        /* Welcome message flashing */
        .welcome-flash {
            animation: flash-text 2s ease-in-out infinite;
        }
        
        @keyframes flash-text {
            0% { color: #ff4b4b; }
            50% { color: #00cc96; }
            100% { color: #ff4b4b; }
        }
        
        /* Button animation */
        .stButton button {
            background: linear-gradient(45deg, #ff4b4b, #00cc96);
            color: white;
            font-size: 18px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(0, 204, 150, 0.5);
        }
        </style>
        
        <!-- Create fitness-themed animation (appears only once) -->
        <div>
            <!-- Fitness icons falling from top -->
            <div class="fitness-icon" style="left: 10%; animation-delay: 0s;">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="#ff4b4b">
                    <path d="M20.57 14.86L22 13.43 20.57 12 17 15.57 8.43 7 12 3.43 10.57 2 9.14 3.43 7.71 2 5.57 4.14 4.14 2.71 2.71 4.14l1.43 1.43L2 7.71l1.43 1.43L2 10.57 3.43 12 7 8.43 15.57 17 12 20.57 13.43 22l1.43-1.43L16.29 22l2.14-2.14 1.43 1.43 1.43-1.43-1.43-1.43L22 16.29z"/>
                </svg>
            </div>
            <div class="fitness-icon" style="left: 25%; animation-delay: 0.5s;">
                <svg width="50" height="50" viewBox="0 0 24 24" fill="#00cc96">
                    <path d="M20.57 14.86L22 13.43 20.57 12 17 15.57 8.43 7 12 3.43 10.57 2 9.14 3.43 7.71 2 5.57 4.14 4.14 2.71 2.71 4.14l1.43 1.43L2 7.71l1.43 1.43L2 10.57 3.43 12 7 8.43 15.57 17 12 20.57 13.43 22l1.43-1.43L16.29 22l2.14-2.14 1.43 1.43 1.43-1.43-1.43-1.43L22 16.29z"/>
                </svg>
            </div>
            <div class="fitness-icon" style="left: 40%; animation-delay: 1s;">
                <svg width="45" height="45" viewBox="0 0 24 24" fill="#ffcc00">
                    <path d="M20 6h-4V4c0-1.1-.9-2-2-2h-4c-1.1 0-2 .9-2 2v2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zM10 4h4v2h-4V4zm10 16H4V8h16v12z"/>
                    <path d="M11 10h2v8h-2z"/>
                    <path d="M7 12h10v2H7z"/>
                </svg>
            </div>
            <div class="fitness-icon" style="left: 55%; animation-delay: 1.5s;">
                <svg width="42" height="42" viewBox="0 0 24 24" fill="#32cd32">
                    <path d="M11 5.08V2c-5 .5-9 4.81-9 10s4 9.5 9 10v-3.08c-3-.48-6-3.4-6-6.92s3-6.44 6-6.92zM18.97 11H22c-.47-5-4-8.53-9-9v3.08C16 5.51 18.54 8 18.97 11zM13 18.92V22c5-.47 8.53-4 9-9h-3.03c-.43 3-2.97 5.49-5.97 5.92z"/>
                </svg>
            </div>
            <div class="fitness-icon" style="left: 70%; animation-delay: 2s;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="#ff6347">
                    <path d="M20.57 14.86L22 13.43 20.57 12 17 15.57 8.43 7 12 3.43 10.57 2 9.14 3.43 7.71 2 5.57 4.14 4.14 2.71 2.71 4.14l1.43 1.43L2 7.71l1.43 1.43L2 10.57 3.43 12 7 8.43 15.57 17 12 20.57 13.43 22l1.43-1.43L16.29 22l2.14-2.14 1.43 1.43 1.43-1.43-1.43-1.43L22 16.29z"/>
                </svg>
            </div>
            <div class="fitness-icon" style="left: 85%; animation-delay: 2.5s;">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="#7b68ee">
                    <path d="M20.57 14.86L22 13.43 20.57 12 17 15.57 8.43 7 12 3.43 10.57 2 9.14 3.43 7.71 2 5.57 4.14 4.14 2.71 2.71 4.14l1.43 1.43L2 7.71l1.43 1.43L2 10.57 3.43 12 7 8.43 15.57 17 12 20.57 13.43 22l1.43-1.43L16.29 22l2.14-2.14 1.43 1.43 1.43-1.43-1.43-1.43L22 16.29z"/>
                </svg>
            </div>
        </div>
        
        <h1 class="animated-title gradient-text">FinFit Dashboard</h1>
        """, unsafe_allow_html=True)
        # Turn off the intro animation after first load
        st.session_state.show_intro_animation = False
    else:
        # Regular title without animation
        st.markdown("""
        <style>
        .gradient-text {
            background: linear-gradient(90deg, #ff4b4b, #00cc96);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(45deg, #ff4b4b, #00cc96);
            color: white;
            font-size: 18px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(0, 204, 150, 0.5);
        }
        
        /* Welcome message flashing */
        .welcome-flash {
            animation: flash-text 2s ease-in-out infinite;
        }
        
        @keyframes flash-text {
            0% { color: #ff4b4b; }
            50% { color: #00cc96; }
            100% { color: #ff4b4b; }
        }
        </style>
        
        <h1 class="gradient-text">FinFit Dashboard</h1>
        """, unsafe_allow_html=True)
    
    # Display multiple columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Use financial fitness-themed SVG instead of inappropriate GIF
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 200px;">
            <div style="background: linear-gradient(135deg, #ff4b4b, #00cc96); border-radius: 50%; width: 150px; height: 150px; display: flex; justify-content: center; align-items: center; box-shadow: 0 0 25px rgba(255, 75, 75, 0.5);">
                <svg width="80" height="80" viewBox="0 0 24 24" fill="white">
                    <path d="M2,22 L6.5,12 L12.5,12 L8,22 L2,22 Z M12,22 L16.5,12 L22.5,12 L18,22 L12,22 Z M5,10 L7,10 L7,3 L5,3 L5,10 Z M11,10 L13,10 L13,3 L11,3 L11,10 Z M17,10 L19,10 L19,3 L17,3 L17,10 Z" />
                </svg>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.title("FinFit – Your Financial Health Dashboard")
        st.markdown('<p class="welcome-flash" style="font-size: 24px; font-weight: bold;">Where Financial Fitness meets Machine Learning</p>', unsafe_allow_html=True)
        st.write("AF3005 – Programming for Finance | FAST-NUCES Islamabad | Spring 2025")

    # Replace loading animation with professionally appropriate finance-themed graphic
    st.markdown("""
    <div style="display: flex; justify-content: center; margin: 20px 0;">
        <div style="background: linear-gradient(to right, rgba(0, 204, 150, 0.1), rgba(0, 204, 150, 0.3)); border-radius: 10px; padding: 20px; width: 80%; box-shadow: 0 0 20px rgba(0, 204, 150, 0.2); text-align: center;">
            <div style="display: flex; justify-content: space-around; margin-bottom: 15px;">
                <div style="height: 40px; width: 15px; background-color: #ff4b4b; animation: chart-bar 2s infinite;"></div>
                <div style="height: 60px; width: 15px; background-color: #00cc96; animation: chart-bar 2s infinite 0.2s;"></div>
                <div style="height: 30px; width: 15px; background-color: #ff4b4b; animation: chart-bar 2s infinite 0.4s;"></div>
                <div style="height: 70px; width: 15px; background-color: #00cc96; animation: chart-bar 2s infinite 0.6s;"></div>
                <div style="height: 50px; width: 15px; background-color: #ff4b4b; animation: chart-bar 2s infinite 0.8s;"></div>
                <div style="height: 80px; width: 15px; background-color: #00cc96; animation: chart-bar 2s infinite 1s;"></div>
                <div style="height: 40px; width: 15px; background-color: #ff4b4b; animation: chart-bar 2s infinite 1.2s;"></div>
            </div>
            <div style="font-weight: bold; color: white;">Financial Performance Tracking</div>
            <style>
                @keyframes chart-bar {
                    0% { transform: scaleY(0.8); }
                    50% { transform: scaleY(1.1); }
                    100% { transform: scaleY(0.8); }
                }
            </style>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Improved content with financial fitness-themed content
    st.markdown("""
    <div style="padding: 20px; background: rgba(0, 0, 0, 0.2); border-radius: 10px; margin: 20px 0;">
    <h3>Welcome to Your Financial Gym! 💪</h3>
    
    <p>Just like physical fitness, financial fitness requires assessment, training, and monitoring.</p>
    
    <p>With FinFit, you can:</p>
    <ul>
        <li>Upload your financial profile data</li>
        <li>Analyze real-time stock information</li>
        <li>Run a machine learning pipeline to assess your investment fitness</li>
        <li>Get personalized recommendations for your financial health</li>
    </ul>
    
    <p style="font-weight: bold; color: #00cc96;">Let's start your financial fitness journey today!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a more prominent call to action button
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-top: 30px;">
        <div style="position: relative; width: 100%;">
            <div style="position: absolute; top: -15px; right: -15px; background: #ff4b4b; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center; font-weight: bold; animation: pulse 2s infinite;">GO</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.button("Start Your Financial Fitness Check 🏋️", on_click=go_to_page, args=('data_loading',), use_container_width=True)

def data_loading_page():
    st.title("Step 1: Financial Health Check-In 📋")
    st.write("Upload your financial profile or use our sample data to get started.")
    
    st.markdown("""
    ### What is a Kragle Financial Profile?
    
    A financial profile includes information such as:
    - Your budget allocation
    - Risk tolerance level
    - Investment time horizon
    - Investment types and preferences
    - Income stability and other financial metrics
    
    Our machine learning model will use this data to evaluate your fitness for specific investments.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload your Kragle financial profile (CSV)", type=["csv"])
        if uploaded_file is not None:
            try:
                data = load_data(uploaded_file)
                st.session_state.data = data
                display_success("Data loaded successfully! Your financial profile is ready for analysis.")
                st.dataframe(data.head(), use_container_width=True)
                st.button("Continue to Stock Selection ➡️", on_click=go_to_page, args=('stock_selection',), use_container_width=True)
            except Exception as e:
                display_error(f"Error loading data: {str(e)}")
        
    with col2:
        st.write("Or use our sample dataset to explore the app")
        if st.button("Use Sample Data", use_container_width=True):
            try:
                # Loading sample data
                data = load_data("kragle_investor_fitness.csv")
                st.session_state.data = data
                display_success("Sample data loaded successfully!")
                st.dataframe(data.head(), use_container_width=True)
                st.button("Continue to Stock Selection ➡️", on_click=go_to_page, args=('stock_selection',), use_container_width=True)
            except Exception as e:
                display_error(f"Error loading sample data: {str(e)}")
                
    st.button("◀️ Back to Intro", on_click=go_to_page, args=('intro',))

def stock_selection_page():
    st.title("Step 2: Choose Your Investment Target 🎯")
    
    if st.session_state.data is None:
        display_error("No data loaded! Please go back and load your financial profile first.")
        st.button("◀️ Go to Data Loading", on_click=go_to_page, args=('data_loading',), use_container_width=True)
        return
        
    # Replace inappropriate GIF with a professionally appropriate financial target graphic
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <div style="background: linear-gradient(45deg, #ff4b4b20, #00cc9620); border-radius: 10px; width: 200px; height: 120px; display: flex; justify-content: center; align-items: center;">
            <div style="position: relative; width: 80px; height: 80px;">
                <!-- Target circles -->
                <div style="position: absolute; top: 0; left: 0; width: 80px; height: 80px; border-radius: 50%; border: 4px solid #ff4b4b; opacity: 0.3;"></div>
                <div style="position: absolute; top: 10px; left: 10px; width: 60px; height: 60px; border-radius: 50%; border: 4px solid #ff4b4b; opacity: 0.5;"></div>
                <div style="position: absolute; top: 20px; left: 20px; width: 40px; height: 40px; border-radius: 50%; border: 4px solid #ff4b4b; opacity: 0.7;"></div>
                <div style="position: absolute; top: 30px; left: 30px; width: 20px; height: 20px; border-radius: 50%; background-color: #00cc96; animation: pulse 2s infinite;"></div>
                <!-- Arrow pointing to target -->
                <div style="position: absolute; top: -10px; left: 100px; transform: rotate(-45deg);">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="#00cc96">
                        <path d="M20 4h-4.05l1.83-1.83c.39-.39.39-1.02 0-1.41s-1.02-.39-1.41 0l-2.34 2.34c-.2.2-.31.47-.31.76V8c0 .55.45 1 1 1h5.28c.55 0 1-.45 1-1V5c0-.55-.45-1-1-1zm-9.5 8.65l1.32 1.32c.2.2.2.51 0 .71-.1.1-.2.15-.35.15s-.26-.05-.35-.15l-1.32-1.32c-.2-.2-.2-.51 0-.71.19-.2.51-.2.7 0zm.45-1.95L9.88 13.3c.2.2.2.51 0 .71-.1.1-.2.15-.35.15s-.26-.05-.35-.15L7.1 11.94c-.2-.2-.2-.51 0-.71s.51-.2.71 0l2.14 2.14c.2.2.2.51 0 .71-.1.1-.2.15-.35.15s-.26-.05-.35-.15L7.11 11.94c-.2-.2-.2-.51 0-.71.2-.19.51-.19.71 0l2.12 2.12c.2.2.2.51 0 .71-.1.1-.2.15-.35.15s-.26-.05-.35-.15l-2.12-2.12c-.2-.2-.2-.51 0-.71.2-.2.51-.2.71 0l2.12 2.12zm-.71-8.35l-8 8c-.2.2-.2.51 0 .71l4.5 4.5c.2.2.51.2.71 0l8-8c.2-.2.2-.51 0-.71l-4.5-4.5c-.2-.2-.51-.2-.71 0zM3.91 17.86c-.39.39-.39 1.02 0 1.41.39.39 1.02.39 1.41 0l2.34-2.34c.2-.2.31-.47.31-.76v-4.24c0-.55-.45-1-1-1H2.69c-.55 0-1 .45-1 1v1c0 .55.45 1 1 1h4.05l-1.83 1.83c-.2.2-.31.47-.31.76v2.07c0 .55.45 1 1 1h4.24c.29 0 .56-.11.76-.31l2.34-2.34c.39-.39.39-1.02 0-1.41-.39-.39-1.02-.39-1.41 0l-1.83 1.83v-4.05c0-.55-.45-1-1-1h-1c-.55 0-1 .45-1 1v4.05l-1.83-1.83c-.39-.39-1.02-.39-1.41 0s-.39 1.02 0 1.41l2.34 2.34c.2.2.47.31.76.31h4.24c.55 0 1-.45 1-1v-1c0-.55-.45-1-1-1H7.85l1.83-1.83c.39-.39.39-1.02 0-1.41s-1.02-.39-1.41 0l-2.34 2.34c-.2.2-.31.47-.31.76v2.07c0 .55.45 1 1 1h4.24c.29 0 .56-.11.76-.31l2.34-2.34c.39-.39.39-1.02 0-1.41-.39-.39-1.02-.39-1.41 0l-1.83 1.83v-4.05c0-.55-.45-1-1-1h-1c-.55 0-1 .45-1 1v4.05l-1.83-1.83z"/>
                    </svg>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("Select a stock to analyze and determine your investment fitness.")
    
    # Popular stock options
    popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "WMT"]
    
    # Create animated stock selection section
    st.markdown("""
    <style>
    .stock-selection {
        animation: fadeIn 1s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="stock-selection">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Quick Selection")
        stock_symbol = st.selectbox("Choose from popular stocks:", 
                                  popular_stocks,
                                  index=None,
                                  placeholder="Select a stock...")
    
    with col2:
        st.subheader("Custom Search")
        custom_symbol = st.text_input("Enter stock symbol (e.g., AAPL):", "")
        
        if st.button("Use Custom Symbol", use_container_width=True):
            if custom_symbol:
                stock_symbol = custom_symbol.upper()
            else:
                display_error("Please enter a stock symbol")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if stock_symbol:
        st.session_state.stock_symbol = stock_symbol
        with st.spinner(f"Fetching latest data for {stock_symbol}..."):
            try:
                # Import the enhanced Yahoo Finance utilities
                from utils.stock_data import get_stock_data, get_stock_info, get_news, create_interactive_chart
                
                # Fetch stock data
                stock_data = get_stock_data(stock_symbol)
                st.session_state.stock_data = stock_data
                
                # Get additional stock information
                try:
                    stock_info = get_stock_info(stock_symbol)
                except Exception as e:
                    st.warning(f"Could not fetch detailed stock information. Basic price data will still be displayed.")
                    stock_info = {'name': stock_symbol, 'sector': 'Unknown', 'industry': 'Unknown'}
                
                # Create tabs for different stock information
                stock_tabs = st.tabs(["📈 Price Data", "ℹ️ Company Info", "📰 Latest News"])
                
                with stock_tabs[0]:
                    # Display stock price information
                    st.subheader(f"{stock_symbol} Stock Overview")
                    
                    # Get the latest price data
                    latest_price = stock_data['Close'].iloc[-1]
                    prev_price = stock_data['Close'].iloc[-2]
                    price_change = latest_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${latest_price:.2f}", 
                                f"{price_change:.2f} ({price_change_pct:.2f}%)",
                                delta_color="normal" if price_change >= 0 else "inverse")
                    with col2:
                        st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,}")
                    with col3:
                        high_low_diff = stock_data['High'].iloc[-1] - stock_data['Low'].iloc[-1]
                        st.metric("Day Range", f"${stock_data['Low'].iloc[-1]:.2f} - ${stock_data['High'].iloc[-1]:.2f}", 
                                f"Spread: ${high_low_diff:.2f}")
                    
                    # Plot stock data using the enhanced interactive chart
                    st.subheader("Interactive Stock Chart")
                    try:
                        fig = create_interactive_chart(stock_data, title=f"{stock_info.get('name', stock_symbol)} Stock Analysis")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        # Fallback to basic chart
                        fig = plot_stock_data(stock_data)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Add period selector
                    st.subheader("Change Time Period")
                    period_col1, period_col2, period_col3, period_col4 = st.columns(4)
                    
                    with period_col1:
                        if st.button("1 Month", use_container_width=True):
                            with st.spinner("Updating chart..."):
                                new_data = get_stock_data(stock_symbol, period="1mo")
                                st.session_state.stock_data = new_data
                                st.rerun()
                    
                    with period_col2:
                        if st.button("3 Months", use_container_width=True):
                            with st.spinner("Updating chart..."):
                                new_data = get_stock_data(stock_symbol, period="3mo")
                                st.session_state.stock_data = new_data
                                st.rerun()
                    
                    with period_col3:
                        if st.button("6 Months", use_container_width=True):
                            with st.spinner("Updating chart..."):
                                new_data = get_stock_data(stock_symbol, period="6mo")
                                st.session_state.stock_data = new_data
                                st.rerun()
                    
                    with period_col4:
                        if st.button("1 Year", use_container_width=True):
                            with st.spinner("Updating chart..."):
                                new_data = get_stock_data(stock_symbol, period="1y")
                                st.session_state.stock_data = new_data
                                st.rerun()
                
                with stock_tabs[1]:
                    # Display company information
                    if 'name' in stock_info and stock_info['name'] != 'Unknown':
                        st.subheader(f"About {stock_info.get('name', stock_symbol)}")
                        
                        # Company profile
                        st.markdown(f"""
                        **Sector:** {stock_info.get('sector', 'Unknown')}  
                        **Industry:** {stock_info.get('industry', 'Unknown')}  
                        **Website:** {stock_info.get('website', 'Unknown')}
                        """)
                        
                        # Business summary
                        if 'business_summary' in stock_info and stock_info['business_summary'] != 'No information available':
                            st.markdown("### Business Summary")
                            st.write(stock_info.get('business_summary', 'No information available'))
                        
                        # Financial metrics in columns
                        st.markdown("### Key Financial Metrics")
                        metrics_col1, metrics_col2 = st.columns(2)
                        
                        with metrics_col1:
                            st.metric("Market Cap", f"${stock_info.get('market_cap', 0):,}")
                            st.metric("P/E Ratio", f"{stock_info.get('pe_ratio', 0):.2f}")
                            st.metric("Beta", f"{stock_info.get('beta', 0):.2f}")
                        
                        with metrics_col2:
                            st.metric("Dividend Yield", f"{stock_info.get('dividend_yield', 0):.2f}%")
                            st.metric("52-Week Range", 
                                    f"${stock_info.get('fifty_two_week_low', 0):.2f} - ${stock_info.get('fifty_two_week_high', 0):.2f}")
                            st.metric("Analyst Target", f"${stock_info.get('analyst_target_price', 0):.2f}")
                    else:
                        st.info(f"Detailed information for {stock_symbol} is not available.")
                
                with stock_tabs[2]:
                    # Display latest news
                    st.subheader(f"Latest News for {stock_symbol}")
                    
                    try:
                        news_items = get_news(stock_symbol)
                        
                        if news_items:
                            for i, news in enumerate(news_items):
                                with st.container():
                                    st.markdown(f"""
                                    ### {news['title']}
                                    **{news['publisher']}** | {news['published'].strftime('%Y-%m-%d %H:%M')}
                                    
                                    {news['summary']}
                                    
                                    [Read more]({news['link']})
                                    """)
                                    if i < len(news_items) - 1:
                                        st.divider()
                        else:
                            st.info(f"No recent news found for {stock_symbol}")
                    
                    except Exception as e:
                        st.warning(f"Could not fetch news for {stock_symbol}.")
                
                # Continue button
                st.button("Continue to Data Preparation ➡️", on_click=go_to_page, args=('data_prep',), use_container_width=True)
                
            except Exception as e:
                display_error(f"Error fetching stock data: {str(e)}")
    
    st.button("◀️ Back to Data Loading", on_click=go_to_page, args=('data_loading',))

def data_prep_page():
    st.title("Step 3: Financial Data Detox 🧹")
    
    if st.session_state.data is None or st.session_state.stock_data is None:
        display_error("Missing data! Please go back and complete previous steps.")
        st.button("◀️ Go to Data Loading", on_click=go_to_page, args=('data_loading',), use_container_width=True)
        return
    
    st.write("Let's prepare your financial data for the fitness assessment.")
    
    with st.expander("About Data Preparation", expanded=True):
        st.markdown("""
        ### Why Data Preparation Matters
        
        Just like warming up before a workout, proper data preparation is essential for accurate results:
        
        - **Cleaning**: Removing missing or invalid values
        - **Preprocessing**: Scaling and normalizing data for optimal model performance
        - **Feature Engineering**: Creating new insights from existing data
        
        This step ensures our machine learning model can effectively assess your financial fitness.
        """)
    
    # Show original data
    st.subheader("Your Original Financial Profile")
    st.dataframe(st.session_state.data.head(), use_container_width=True)
    
    # Process data when button is clicked
    if st.button("Start Data Preparation", use_container_width=True):
        with st.spinner("Preparing your financial data..."):
            try:
                # Data preprocessing
                processed_data = preprocess_data(st.session_state.data)
                
                # Data splitting
                X = processed_data.drop('fitness', axis=1)
                y = processed_data['fitness']
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Show progress
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Display processed data
                st.subheader("Processed Financial Profile")
                st.dataframe(processed_data.head(), use_container_width=True)
                
                # Show statistics
                st.subheader("Data Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Training data size:** {len(X_train)} profiles")
                    st.write(f"**Testing data size:** {len(X_test)} profiles")
                with col2:
                    st.write(f"**Features used:** {', '.join(X_train.columns)}")
                    st.write(f"**Target variable:** Investment Fitness (1 = Fit, 0 = Unfit)")
                
                display_success("Data preparation complete! Your financial data is now ready for the fitness model.")
                st.button("Continue to Model Training ➡️", on_click=go_to_page, args=('model_training',), use_container_width=True)
                
            except Exception as e:
                display_error(f"Error in data preparation: {str(e)}")
    
    st.button("◀️ Back to Stock Selection", on_click=go_to_page, args=('stock_selection',))

def model_training_page():
    # Create a header with custom fitness image
    col1, col2 = st.columns([1, 5])
    with col1:
        # Fitness icon instead of "Content Not Available" placeholder
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; background: linear-gradient(45deg, #ff4b4b, #00cc96); border-radius: 50%; width: 70px; height: 70px; box-shadow: 0 0 15px rgba(255, 255, 255, 0.2);">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="white">
                <path d="M20.57 14.86L22 13.43 20.57 12 17 15.57 8.43 7 12 3.43 10.57 2 9.14 3.43 7.71 2 5.57 4.14 4.14 2.71 2.71 4.14l1.43 1.43L2 7.71l1.43 1.43L2 10.57 3.43 12 7 8.43 15.57 17 12 20.57 13.43 22l1.43-1.43L16.29 22l2.14-2.14 1.43 1.43 1.43-1.43-1.43-1.43L22 16.29z"/>
            </svg>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.title("Step 4: Financial Fitness Training 🏋️‍♀️")
    
    if (st.session_state.X_train is None or st.session_state.y_train is None or 
        st.session_state.X_test is None or st.session_state.y_test is None):
        display_error("Missing processed data! Please go back and complete the data preparation step.")
        st.button("◀️ Go to Data Preparation", on_click=go_to_page, args=('data_prep',), use_container_width=True)
        return
    
    # Financial training workout visualization - professionally appropriate
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <div style="background: linear-gradient(to right, rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.3)); border-radius: 10px; width: 250px; height: 150px; box-shadow: 0 0 15px rgba(255, 255, 255, 0.1); padding: 20px; text-align: center;">
            <div style="font-weight: bold; color: white; margin-bottom: 15px;">Financial Model Training</div>
            
            <!-- Animated graph bars representing training progress -->
            <div style="display: flex; justify-content: space-around; align-items: flex-end; height: 80px; margin-bottom: 10px;">
                <div style="width: 8px; background: linear-gradient(to top, #00cc96, #00cc9630); height: 30%; animation: train-up-down 1.2s infinite alternate;"></div>
                <div style="width: 8px; background: linear-gradient(to top, #00cc96, #00cc9630); height: 50%; animation: train-up-down 1.2s infinite alternate 0.2s;"></div>
                <div style="width: 8px; background: linear-gradient(to top, #00cc96, #00cc9630); height: 20%; animation: train-up-down 1.2s infinite alternate 0.4s;"></div>
                <div style="width: 8px; background: linear-gradient(to top, #00cc96, #00cc9630); height: 70%; animation: train-up-down 1.2s infinite alternate 0.6s;"></div>
                <div style="width: 8px; background: linear-gradient(to top, #00cc96, #00cc9630); height: 40%; animation: train-up-down 1.2s infinite alternate 0.8s;"></div>
                <div style="width: 8px; background: linear-gradient(to top, #00cc96, #00cc9630); height: 60%; animation: train-up-down 1.2s infinite alternate 1s;"></div>
                <div style="width: 8px; background: linear-gradient(to top, #00cc96, #00cc9630); height: 80%; animation: train-up-down 1.2s infinite alternate 1.2s;"></div>
                <div style="width: 8px; background: linear-gradient(to top, #00cc96, #00cc9630); height: 35%; animation: train-up-down 1.2s infinite alternate 1.4s;"></div>
            </div>
            
            <!-- Progress line representing model fitting -->
            <div style="width: 100%; height: 4px; background: #ff4b4b20; border-radius: 2px; overflow: hidden;">
                <div style="height: 100%; width: 30%; background: #ff4b4b; animation: progress-move 3s infinite;"></div>
            </div>
            
            <style>
                @keyframes train-up-down {
                    from { transform: scaleY(0.8); }
                    to { transform: scaleY(1.1); }
                }
                @keyframes progress-move {
                    0% { width: 0%; }
                    50% { width: 70%; }
                    100% { width: 100%; }
                }
            </style>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("Now we'll train a machine learning model to assess your investment fitness.")
    
    # Add styles for the expander
    st.markdown("""
    <style>
    .enhanced-expander {
        background: linear-gradient(90deg, rgba(255, 75, 75, 0.1), rgba(0, 204, 150, 0.1));
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    
    .feature-row {
        animation: slide-in 0.5s ease-out;
    }
    
    @keyframes slide-in {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .metric-box {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #00cc96;
    }
    
    .training-button {
        position: relative;
        overflow: hidden;
    }
    
    .training-button::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="enhanced-expander">', unsafe_allow_html=True)
    with st.expander("About the Fitness Model", expanded=True):
        st.markdown("""
        ### Logistic Regression for Financial Fitness
        
        We use Logistic Regression to determine your investment fitness:
        
        - **Binary Classification**: Identifies if you're "Fit" (1) or "Unfit" (0) for the selected investment
        - **Probability Based**: Provides confidence scores for the fitness assessment
        - **Feature Importance**: Reveals which financial factors most impact your fitness level
        
        This is similar to how a personal trainer assesses your physical fitness for different workout types.
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show training data overview with enhanced styling
    st.subheader("Training Data Overview")
    
    # Create metrics with enhanced styling
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <p style="margin: 0; font-size: 14px; color: #aaa;">Number of features</p>
            <p style="margin: 0; font-size: 24px; font-weight: bold;">{st.session_state.X_train.shape[1]}</p>
        </div>
        
        <div class="metric-box" style="margin-top: 15px;">
            <p style="margin: 0; font-size: 14px; color: #aaa;">Training samples</p>
            <p style="margin: 0; font-size: 24px; font-weight: bold;">{len(st.session_state.X_train)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fit_percentage = (st.session_state.y_train.sum() / len(st.session_state.y_train)) * 100
        unfit_percentage = 100 - fit_percentage
        
        st.markdown(f"""
        <div class="metric-box" style="border-left: 3px solid #00cc96;">
            <p style="margin: 0; font-size: 14px; color: #aaa;">Fit investors</p>
            <p style="margin: 0; font-size: 24px; font-weight: bold; color: #00cc96;">{fit_percentage:.1f}%</p>
        </div>
        
        <div class="metric-box" style="margin-top: 15px; border-left: 3px solid #ff4b4b;">
            <p style="margin: 0; font-size: 14px; color: #aaa;">Unfit investors</p>
            <p style="margin: 0; font-size: 24px; font-weight: bold; color: #ff4b4b;">{unfit_percentage:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display feature names with animated styling
    st.subheader("Features for Training")
    
    feature_cols = st.columns(3)
    for i, feature in enumerate(st.session_state.X_train.columns):
        col_idx = i % 3
        with feature_cols[col_idx]:
            delay = i * 0.1
            st.markdown(f"""
            <div class="feature-row" style="animation-delay: {delay}s;">
                <span style="display: inline-block; width: 10px; height: 10px; background: {'#00cc96' if i % 2 == 0 else '#ff4b4b'}; border-radius: 50%; margin-right: 8px;"></span>
                {feature}
            </div>
            """, unsafe_allow_html=True)
    
    # Add a divider before the training button
    st.markdown("<hr style='margin: 30px 0; border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);'>", unsafe_allow_html=True)
    
    # Train model when button is clicked
    st.markdown('<div class="training-button">', unsafe_allow_html=True)
    if st.button("Train Financial Fitness Model 🏋️", use_container_width=True):
        with st.spinner("Training your financial fitness model..."):
            try:
                # Model training with enhanced progress simulation
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    sub_status = st.empty()
                    
                    # Simulate training progress with more detailed steps
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        
                        if i < 20:
                            status_text.markdown("**Step 1: Initializing model parameters...**")
                            sub_status.text(f"Setting up logistic regression with balanced class weights... {i*5}%")
                        elif i < 40:
                            status_text.markdown("**Step 2: Data preprocessing...**")
                            sub_status.text(f"Scaling features and handling coefficients... {(i-20)*5}%")
                        elif i < 60:
                            status_text.markdown("**Step 3: Model fitting...**")
                            sub_status.text(f"Training on {len(st.session_state.X_train)} samples... {(i-40)*5}%")
                        elif i < 80:
                            status_text.markdown("**Step 4: Optimizing and validating...**")
                            sub_status.text(f"Optimizing convergence and validating results... {(i-60)*5}%")
                        else:
                            status_text.markdown("**Step 5: Finalizing model...**")
                            sub_status.text(f"Computing feature importance and preparing results... {(i-80)*5}%")
                        
                        time.sleep(0.03)
                
                # Actual model training using our ml_models module
                model, feature_importance = train_logistic_regression(st.session_state.X_train, st.session_state.y_train)
                
                # Store model in session state
                st.session_state.model = model
                
                # Display success message
                status_text.markdown("**Model training complete!** 🎉")
                sub_status.empty()
                
                # Add animated success message
                st.markdown("""
                <div style="padding: 15px; background: rgba(0, 204, 150, 0.1); border-radius: 5px; border-left: 4px solid #00cc96; margin: 20px 0; animation: fade-in 0.5s ease-out;">
                    <h3 style="margin-top: 0; color: #00cc96;">Training Complete!</h3>
                    <p>Your financial fitness model is ready. The model has learned patterns from your data and can now predict investment fitness.</p>
                </div>
                <style>
                @keyframes fade-in {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Show model information with enhanced styling
                st.subheader("Model Information")
                
                # Display model accuracy on training data
                train_accuracy = model.score(st.session_state.X_train, st.session_state.y_train) * 100
                
                st.markdown(f"""
                <div style="background: rgba(0, 0, 0, 0.2); border-radius: 5px; padding: 15px; margin-bottom: 20px;">
                    <p style="margin: 0; font-size: 14px; color: #aaa;">Training Accuracy</p>
                    <p style="margin: 0; font-size: 28px; font-weight: bold; color: {'#00cc96' if train_accuracy > 70 else '#ff4b4b'};">{train_accuracy:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show coefficients with enhanced styling
                st.markdown("### Feature Importance")
                st.write("These features have the biggest impact on financial fitness:")
                
                # Create enhanced feature importance plot with custom styling
                # Sort feature importance for better visualization
                feature_importance = feature_importance.sort_values('Coefficient', key=abs, ascending=False)
                
                # Use custom colors based on coefficient sign
                colors = ['#00cc96' if x > 0 else '#ff4b4b' for x in feature_importance['Coefficient']]
                
                fig = px.bar(
                    feature_importance,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance for Financial Fitness',
                    color='Coefficient',
                    color_continuous_scale=['#ff4b4b', '#aaaaaa', '#00cc96'],
                )
                
                fig.update_layout(
                    template='plotly_dark',
                    title_font_size=20,
                    title_x=0.5,
                    xaxis_title="Impact on Fitness (Coefficient Magnitude)",
                    yaxis_title=None,
                    coloraxis_showscale=False,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation of feature importance
                top_positive = feature_importance[feature_importance['Coefficient'] > 0].head(1)
                top_negative = feature_importance[feature_importance['Coefficient'] < 0].head(1)
                
                if not top_positive.empty and not top_negative.empty:
                    st.markdown(f"""
                    <div style="background: rgba(0, 0, 0, 0.2); border-radius: 5px; padding: 15px; margin: 20px 0;">
                        <h4 style="margin-top: 0;">Key Insights:</h4>
                        <p>📈 <strong style="color: #00cc96;">{top_positive['Feature'].values[0]}</strong> has the strongest positive impact on fitness</p>
                        <p>📉 <strong style="color: #ff4b4b;">{top_negative['Feature'].values[0]}</strong> has the strongest negative impact on fitness</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Create a clear call-to-action button to continue
                st.markdown('<div style="margin-top: 30px;">', unsafe_allow_html=True)
                st.button("Continue to Model Evaluation ➡️", on_click=go_to_page, args=('model_eval',), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                display_error(f"Error in model training: {str(e)}")
                st.markdown(f"""
                <div style="padding: 15px; background: rgba(255, 75, 75, 0.1); border-radius: 5px; border-left: 4px solid #ff4b4b; margin: 20px 0;">
                    <h3 style="margin-top: 0; color: #ff4b4b;">Training Error</h3>
                    <p>There was an issue while training the model. Here's what went wrong:</p>
                    <code>{str(e)}</code>
                    <p style="margin-top: 10px;">Try going back to the data preparation step to ensure your data is properly formatted.</p>
                </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add navigation back button with improved styling
    st.markdown("<div style='margin-top: 30px;'>", unsafe_allow_html=True)
    st.button("◀️ Back to Data Preparation", on_click=go_to_page, args=('data_prep',))
    st.markdown("</div>", unsafe_allow_html=True)

def model_eval_page():
    st.title("Step 5: Fitness Performance Check 📊")
    
    # Check if we have the necessary data and model
    if (st.session_state.model is None or 
        st.session_state.X_test is None or 
        st.session_state.y_test is None):
        display_error("Missing model or test data! Please go back and complete the training step.")
        st.button("◀️ Go to Model Training", on_click=go_to_page, args=('model_training',), use_container_width=True)
        return
    
    # Replace with professionally appropriate evaluation visualization
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <div style="background: linear-gradient(to right, rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.3)); border-radius: 10px; width: 250px; height: 150px; box-shadow: 0 0 15px rgba(255, 255, 255, 0.1); padding: 20px; position: relative; overflow: hidden;">
            <!-- Financial evaluation gauges -->
            <div style="text-align: center; font-weight: bold; color: white; margin-bottom: 15px;">Model Evaluation Results</div>
            
            <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                <!-- Accuracy gauge -->
                <div style="text-align: center; position: relative; width: 80px;">
                    <div style="position: relative; width: 60px; height: 60px; margin: 0 auto; background: rgba(0,0,0,0.2); border-radius: 50%; overflow: hidden;">
                        <div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 75%; background: linear-gradient(to top, #00cc96, #00cc9650); transform-origin: bottom center; animation: gauge-fill 2s ease-out forwards;"></div>
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 14px; color: white; font-weight: bold;">75%</div>
                    </div>
                    <div style="font-size: 12px; color: #aaa; margin-top: 5px;">Accuracy</div>
                </div>
                
                <!-- Confidence gauge -->
                <div style="text-align: center; position: relative; width: 80px;">
                    <div style="position: relative; width: 60px; height: 60px; margin: 0 auto; background: rgba(0,0,0,0.2); border-radius: 50%; overflow: hidden;">
                        <div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 85%; background: linear-gradient(to top, #ff4b4b, #ff4b4b50); transform-origin: bottom center; animation: gauge-fill 2s ease-out forwards;"></div>
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 14px; color: white; font-weight: bold;">85%</div>
                    </div>
                    <div style="font-size: 12px; color: #aaa; margin-top: 5px;">Confidence</div>
                </div>
            </div>
            
            <!-- Animated checkmark for completion -->
            <div style="position: absolute; bottom: 10px; right: 10px; width: 20px; height: 20px; border-radius: 50%; background: #00cc96; display: flex; justify-content: center; align-items: center; animation: pulse-check 2s infinite;">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="white">
                    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path>
                </svg>
            </div>
            
            <style>
                @keyframes gauge-fill {
                    0% { height: 0%; }
                    100% { height: var(--end-height, 75%); }
                }
                @keyframes pulse-check {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.2); }
                    100% { transform: scale(1); }
                }
            </style>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("Let's evaluate how well your financial fitness model performs.")
    
    # Add CSS styles for this page
    st.markdown("""
    <style>
    .eval-section {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid #00cc96;
    }
    
    .metric-card {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    
    .fitness-prediction {
        animation: pulse-bg 2s infinite;
        border-radius: 10px;
        padding: 20px;
        margin: 30px 0;
    }
    
    @keyframes pulse-bg {
        0% { background: rgba(0, 204, 150, 0.05); }
        50% { background: rgba(0, 204, 150, 0.15); }
        100% { background: rgba(0, 204, 150, 0.05); }
    }
    
    .fit-indicator {
        font-size: 70px;
        text-align: center;
        margin: 0;
        line-height: 1;
    }
    
    .probability-bar {
        height: 20px;
        background: linear-gradient(to right, #ff4b4b, #00cc96);
        border-radius: 10px;
        position: relative;
        margin: 10px 0 30px 0;
        overflow: hidden;
    }
    
    .probability-indicator {
        position: absolute;
        top: -10px;
        width: 10px;
        height: 40px;
        background: white;
        transform: translateX(-50%);
    }
    
    .eval-button {
        position: relative;
        overflow: hidden;
    }
    
    .eval-button::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create tabs for different evaluation aspects
    eval_tabs = st.tabs(["📈 Performance Metrics", "🎯 Fitness Prediction", "🔄 Confusion Matrix"])
    
    with eval_tabs[0]:
        st.markdown('<div class="eval-section">', unsafe_allow_html=True)
        st.markdown("### Model Performance")
        st.write("How well does your model identify fit and unfit investors?")
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # Make predictions
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(st.session_state.y_test, y_pred)
        precision = precision_score(st.session_state.y_test, y_pred, zero_division=0)
        recall = recall_score(st.session_state.y_test, y_pred, zero_division=0)
        f1 = f1_score(st.session_state.y_test, y_pred, zero_division=0)
        
        # Display metrics in a nice grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p style="margin: 0; font-size: 14px; color: #aaa;">Accuracy</p>
                <p style="margin: 0; font-size: 32px; font-weight: bold; color: {'#00cc96' if accuracy >= 0.7 else '#ff4b4b'};">
                    {accuracy:.2f}
                </p>
                <p style="margin: 0; font-size: 12px; color: #888;">
                    Percentage of correct predictions
                </p>
            </div>
            
            <div class="metric-card">
                <p style="margin: 0; font-size: 14px; color: #aaa;">Precision</p>
                <p style="margin: 0; font-size: 32px; font-weight: bold; color: {'#00cc96' if precision >= 0.7 else '#ff4b4b'};">
                    {precision:.2f}
                </p>
                <p style="margin: 0; font-size: 12px; color: #888;">
                    Of all predicted fit, how many were actually fit
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p style="margin: 0; font-size: 14px; color: #aaa;">Recall</p>
                <p style="margin: 0; font-size: 32px; font-weight: bold; color: {'#00cc96' if recall >= 0.7 else '#ff4b4b'};">
                    {recall:.2f}
                </p>
                <p style="margin: 0; font-size: 12px; color: #888;">
                    Of all actually fit, how many were identified
                </p>
            </div>
            
            <div class="metric-card">
                <p style="margin: 0; font-size: 14px; color: #aaa;">F1 Score</p>
                <p style="margin: 0; font-size: 32px; font-weight: bold; color: {'#00cc96' if f1 >= 0.7 else '#ff4b4b'};">
                    {f1:.2f}
                </p>
                <p style="margin: 0; font-size: 12px; color: #888;">
                    Balance between precision and recall
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Add interpretation of results
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background: rgba(0, 0, 0, 0.2); border-radius: 5px;">
            <h4 style="margin-top: 0;">What do these numbers mean?</h4>
            <p>
                <strong>Accuracy</strong> shows how often the model is correct overall. <br>
                <strong>Precision</strong> indicates how reliable the "Fit" predictions are. <br>
                <strong>Recall</strong> shows how well the model finds all the truly "Fit" investors. <br>
                <strong>F1 Score</strong> balances precision and recall - higher is better.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
            
    with eval_tabs[1]:
        st.markdown('<div class="eval-section">', unsafe_allow_html=True)
        st.markdown("### Investment Fitness Prediction")
        st.write(f"Based on your {st.session_state.stock_symbol if st.session_state.stock_symbol else 'selected stock'} and financial profile...")
        
        # Make a prediction with probability
        probabilities = st.session_state.model.predict_proba(st.session_state.X_test)
        average_prob_fit = probabilities[:, 1].mean() * 100
        
        # Determine overall fitness status
        overall_fitness = "Fit" if average_prob_fit >= 50 else "Unfit"
        emoji = "💪" if overall_fitness == "Fit" else "⚠️"
        
        st.markdown(f"""
        <div class="fitness-prediction">
            <p style="text-align: center; font-size: 18px; margin-bottom: 10px;">
                Your financial fitness prediction:
            </p>
            <p class="fit-indicator" style="color: {'#00cc96' if overall_fitness == 'Fit' else '#ff4b4b'};">
                {emoji} {overall_fitness}
            </p>
            <p style="text-align: center; margin-bottom: 20px;">
                with <strong>{average_prob_fit:.1f}%</strong> confidence
            </p>
            
            <div class="probability-bar">
                <div class="probability-indicator" style="left: {average_prob_fit}%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: -20px;">
                <span style="color: #ff4b4b;">Unfit (0%)</span>
                <span style="color: #00cc96;">Fit (100%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add recommendations based on fitness
        st.markdown(f"""
        <div style="background: rgba(0, 0, 0, 0.2); border-radius: 10px; padding: 20px; margin-top: 20px;">
            <h4 style="margin-top: 0;">Financial Fitness Recommendation</h4>
            
            <p>{'Based on your profile, you appear to be <strong style="color: #00cc96;">FIT</strong> for investing in this stock. Consider the following tips:' if overall_fitness == 'Fit' else 'Based on your profile, you appear to be <strong style="color: #ff4b4b;">UNFIT</strong> for investing in this stock. Consider the following tips:'}</p>
            
            <ul>
                {'<li>Your financial profile shows strength for this investment</li><li>Consider a balanced portfolio approach</li><li>Regularly monitor market conditions</li><li>Maintain your current financial fitness</li>' if overall_fitness == 'Fit' else '<li>You may want to reconsider this particular investment</li><li>Build stronger financial fundamentals first</li><li>Consider lower-risk alternatives</li><li>Consult with a financial advisor</li>'}
            </ul>
            
            <p style="font-size: 12px; color: #aaa; margin-top: 20px;">
                Note: This prediction is based on historical data and should not be your only factor in making investment decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    with eval_tabs[2]:
        st.markdown('<div class="eval-section">', unsafe_allow_html=True)
        st.markdown("### Confusion Matrix")
        st.write("Visualizing prediction errors and successes")
        
        # Calculate confusion matrix
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        
        # Create a nicer confusion matrix visualization using plotly
        import plotly.figure_factory as ff
        import numpy as np
        
        # Create annotated heatmap
        x = ['Predicted Unfit', 'Predicted Fit']
        y = ['Actually Unfit', 'Actually Fit']
        
        # Ensure we have a valid confusion matrix (at least 2x2)
        if cm.shape != (2, 2):
            # Adjust the matrix if needed
            if cm.shape == (1, 1):
                # Only one class present, expand to 2x2
                if st.session_state.y_test.iloc[0] == 0:
                    # Only unfit class present
                    cm = np.array([[cm[0, 0], 0], [0, 0]])
                else:
                    # Only fit class present
                    cm = np.array([[0, 0], [0, cm[0, 0]]])
            elif cm.shape == (1, 2):
                # Only one class in actual, but two in predicted
                if st.session_state.y_test.iloc[0] == 0:
                    # Only unfit class present in actual
                    cm = np.array([[cm[0, 0], cm[0, 1]], [0, 0]])
                else:
                    # Only fit class present in actual
                    cm = np.array([[0, 0], [cm[0, 0], cm[0, 1]]])
            elif cm.shape == (2, 1):
                # Two classes in actual, only one in predicted
                if y_pred[0] == 0:
                    # Only unfit class present in predicted
                    cm = np.array([[cm[0, 0], 0], [cm[1, 0], 0]])
                else:
                    # Only fit class present in predicted
                    cm = np.array([[0, cm[0, 0]], [0, cm[1, 0]]])
        
        # Create the figure
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=x,
            y=y,
            annotation_text=cm,
            colorscale=[[0, '#ff4b4b'], [1, '#00cc96']],
            showscale=True
        )
        
        # Customize layout
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            template='plotly_dark',
            title_x=0.5,
            width=600,
            height=500,
            margin=dict(t=100, b=0, l=0, r=0)
        )
        
        # Make the heatmap square
        fig.update_xaxes(constrain='domain')
        fig.update_yaxes(autorange='reversed')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation of confusion matrix
        st.markdown("""
        <div style="background: rgba(0, 0, 0, 0.2); border-radius: 5px; padding: 15px; margin-top: 20px;">
            <h4 style="margin-top: 0;">Understanding the Confusion Matrix</h4>
            <ul>
                <li><strong>Top-Left:</strong> Correctly identified as Unfit (True Negatives)</li>
                <li><strong>Top-Right:</strong> Incorrectly identified as Fit when actually Unfit (False Positives)</li>
                <li><strong>Bottom-Left:</strong> Incorrectly identified as Unfit when actually Fit (False Negatives)</li>
                <li><strong>Bottom-Right:</strong> Correctly identified as Fit (True Positives)</li>
            </ul>
            <p>A good model will have higher numbers on the diagonal (top-left to bottom-right).</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create custom component for the fitness assessment card using Streamlit's native components
    st.markdown('<div class="eval-button">', unsafe_allow_html=True)
    if st.button("View Investment Fitness Insights 💡", use_container_width=True):
        # Create styled container using st.container() instead of HTML
        with st.container():
            # Header with icon
            col1, col2 = st.columns([1, 6])
            with col1:
                if overall_fitness == "Fit":
                    st.markdown("### 💪")
                else:
                    st.markdown("### ⚠️")
            with col2:
                st.markdown("### Your Financial Strength Assessment")
            
            # Metrics in cards
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.markdown("#### Fitness Score")
                st.markdown(f"<h2 style='color: #00cc96;'>{average_prob_fit:.1f}%</h2>", unsafe_allow_html=True)
                st.caption("Confidence in your investment readiness")
            
            with metrics_col2:
                st.markdown("#### Model Accuracy")
                st.markdown(f"<h2 style='color: #7b68ee;'>{accuracy:.2f}</h2>", unsafe_allow_html=True)
                st.caption("Prediction reliability factor")
            
            # Key takeaways
            with st.container():
                st.markdown("#### Key Takeaways")
                
                # Use native Streamlit components for clean display
                if overall_fitness == "Fit":
                    st.success("Based on your profile strengths, this investment appears to align well with your financial situation.")
                    st.success("Your risk tolerance and investment horizon are well suited for this stock.")
                else:
                    st.error("Your current profile suggests some caution may be needed with this investment.")
                    st.error("Consider investments that better match your risk tolerance and investment horizon.")
                
                st.info("Remember that machine learning models provide guidance, but financial decisions should involve multiple factors.")
            
            # Summary banner
            if overall_fitness == "Fit":
                st.success("FINANCIALLY FIT FOR THIS INVESTMENT")
            else:
                st.error("NEEDS MORE FINANCIAL PREPARATION")
            
            # Footer
            st.caption(f"Analyzed by FinFit • {datetime.now().strftime('%Y-%m-%d')}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("◀️ Back to Model Training", on_click=go_to_page, args=('model_training',), use_container_width=True)
    with col2:
        st.button("🏠 Return to Home", on_click=go_to_page, args=('intro',), use_container_width=True)

# Main function
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f3cb.svg", width=50)
        st.title("FinFit")
        st.write("Your Financial Fitness Dashboard")
        st.divider()
        
        st.subheader("Navigation")
        
        # Simplified navigation buttons
        if st.button("⭐ Intro", use_container_width=True):
            go_to_page('intro')
        if st.button("📋 Data Loading", use_container_width=True):
            go_to_page('data_loading')
        if st.button("🎯 Stock Selection", use_container_width=True):
            go_to_page('stock_selection')
        if st.button("🧹 Data Preparation", use_container_width=True):
            go_to_page('data_prep')
        if st.button("🏋️‍♀️ Model Training", use_container_width=True):
            go_to_page('model_training')
        if st.button("📊 Model Evaluation", use_container_width=True):
            go_to_page('model_eval')
        
        st.divider()
        st.write("AF3005 – Programming for Finance")
        st.write("FAST-NUCES Islamabad | Spring 2025")
    
    # Render the current page
    if st.session_state.page == 'intro':
        intro_page()
    elif st.session_state.page == 'data_loading':
        data_loading_page()
    elif st.session_state.page == 'stock_selection':
        stock_selection_page()
    elif st.session_state.page == 'data_prep':
        data_prep_page()
    elif st.session_state.page == 'model_training':
        model_training_page()
    elif st.session_state.page == 'model_eval':
        model_eval_page()

if __name__ == "__main__":
    main()