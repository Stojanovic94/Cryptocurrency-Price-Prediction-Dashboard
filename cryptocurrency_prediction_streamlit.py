# Cryptocurrency Price Prediction Dashboard
import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autots import AutoTS
from datetime import timedelta, datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import time
import psutil
import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')

# PAGE SETUP
st.set_page_config(
    page_title="Cryptocurrency Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cryptocurrency options
crypto_options = {
    "Bitcoin (BTC)": "BTC",
    "Ethereum (ETH)": "ETH",
    "Dogecoin (DOGE)": "DOGE",
    "Litecoin (LTC)": "LTC",
    "Cardano (ADA)": "ADA",
    "Ripple (XRP)": "XRP",
    "Solana (SOL)": "SOL",
    "Polkadot (DOT)": "DOT",
    "Binance Coin (BNB)": "BNB",
    "Chainlink (LINK)": "LINK",
}

# Fiat currencies
fiat_options = ["USD", "EUR", "GBP", "JPY", "RUB", "CAD", "AUD", "CNY"]

# Title and description
st.title("📈 Cryptocurrency Price Prediction")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Option for CSV file upload
    st.subheader("Data Source")
    use_csv = st.checkbox("Use custom CSV file", value=False)
    
    if use_csv:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
    else:
        # Cryptocurrency and currency selection (only if not using CSV)
        crypto_name = st.selectbox('Cryptocurrency:', list(crypto_options.keys()))
        fiat_curr = st.selectbox('Fiat currency:', fiat_options)
        selected_crypto = crypto_options[crypto_name]
        ticker = f"{selected_crypto}-{fiat_curr}"
        st.session_state.ticker = ticker
    
    # CPU resource control
    st.subheader("Resources")
    available_cores = psutil.cpu_count(logical=False)
    cpu_cores = st.slider("Number of CPU cores", min_value=1, max_value=available_cores, 
                         value=min(4, available_cores))
    
    # Date range (only if not using CSV)
    if not use_csv:
        st.subheader("Data Range")
        use_default_dates = st.checkbox("Use last 3 years", value=True)
        
        if use_default_dates:
            end_date = datetime.today().date()
            start_date = end_date - timedelta(days=3*365)
        else:
            start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=3*365))
            end_date = st.date_input("End date", value=datetime.today())
        
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
    
    # Forecast length
    forecast_days = st.slider("Days for forecast", min_value=30, max_value=365, value=90)
    
    # Model selection
    st.subheader("Models")
    use_arima = st.checkbox("Use ARIMA", value=True)
    use_holtwinters = st.checkbox("Use Holt-Winters", value=True)
    use_prophet = st.checkbox("Use Prophet", value=True)
    use_lstm = st.checkbox("Use LSTM", value=True)
    use_ets = st.checkbox("Use ETS", value=True)
    use_theta = st.checkbox("Use Theta", value=True)
    use_linear = st.checkbox("Use Linear Regression", value=True)
    
    # LSTM hyperparameters
    if use_lstm:
        st.subheader("LSTM Hyperparameters")
        lstm_sequence_length = st.slider("Sequence Length", min_value=7, max_value=90, value=30)
        lstm_epochs = st.slider("Epochs", min_value=10, max_value=500, value=100)
        lstm_hidden_size = st.slider("Hidden Size", min_value=10, max_value=100, value=50)
        lstm_num_layers = st.slider("Number of Layers", min_value=1, max_value=5, value=2)
        lstm_learning_rate = st.slider("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        lstm_batch_size = st.slider("Batch Size", min_value=8, max_value=64, value=16)
    
    # Technical indicators
    st.subheader("Technical Indicators")
    use_technical_indicators = st.checkbox("Add technical indicators", value=False)
    if use_technical_indicators:
        use_sma = st.checkbox("SMA (Simple Moving Average)", value=True)
        use_ema = st.checkbox("EMA (Exponential Moving Average)", value=True)
        use_rsi = st.checkbox("RSI (Relative Strength Index)", value=False)
        use_macd = st.checkbox("MACD (Moving Average Convergence Divergence)", value=False)

# PyTorch LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Time Series Dataset for PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM Training and Forecasting Function
def lstm_forecast_pytorch(df, forecast_days, sequence_length, hidden_size, num_layers, 
                         epochs, learning_rate, batch_size):
    # Prepare data
    data = df['y'].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length])
        y.append(data_scaled[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data (use last sequence_length points for validation)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=1, hidden_size=hidden_size, 
                     num_layers=num_layers, output_size=1).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Prepare for forecasting
    model.eval()
    with torch.no_grad():
        # Use the last sequence_length points to make the first prediction
        last_sequence = data_scaled[-sequence_length:].reshape(1, sequence_length, 1)
        last_sequence = torch.FloatTensor(last_sequence).to(device)
        
        forecasts = []
        current_sequence = last_sequence
        
        for _ in range(forecast_days):
            # Predict next point
            next_pred = model(current_sequence)
            forecasts.append(next_pred.cpu().numpy().flatten())
            
            # Update sequence: remove first element and add prediction
            current_sequence = torch.cat([
                current_sequence[:, 1:, :], 
                next_pred.unsqueeze(0).unsqueeze(2)
            ], dim=1)
    
    # Inverse transform forecasts
    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts = scaler.inverse_transform(forecasts)
    
    return forecasts.flatten()

# Function to calculate technical indicators
def calculate_technical_indicators(df, use_sma=True, use_ema=True, use_rsi=False, use_macd=False):
    df = df.copy()
    
    if use_sma:
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    if use_ema:
        df['EMA_7'] = df['Close'].ewm(span=7).mean()
        df['EMA_30'] = df['Close'].ewm(span=30).mean()
    
    if use_rsi:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    if use_macd:
        exp12 = df['Close'].ewm(span=12).mean()
        exp26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    return df

# Function to download data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def download_data(ticker, start_date, end_date):
    df_raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    
    if use_technical_indicators:
        df_raw = calculate_technical_indicators(
            df_raw, 
            use_sma=use_sma, 
            use_ema=use_ema, 
            use_rsi=use_rsi, 
            use_macd=use_macd
        )
    
    return df_raw

# Function to load CSV data
def load_csv_data(uploaded_file):
    try:
        # Read CSV with proper encoding and handle the index column
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig', index_col=0)
        
        # Check if DataFrame is empty
        if df.empty:
            st.error("CSV file is empty.")
            return pd.DataFrame()
        
        # Rename columns to expected names
        df = df.rename(columns={'Date': 'ds', 'Price': 'y'})
        
        # Convert date column to datetime
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        
        # Convert price column to numeric
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Check if we have valid date and price columns
        if df['ds'].isnull().all():
            st.error("Could not find a valid date column in the CSV file.")
            return pd.DataFrame()
            
        if df['y'].isnull().all():
            st.error("Could not find a valid price column in the CSV file.")
            return pd.DataFrame()
        
        # Remove rows with NaN values
        df = df.dropna(subset=['ds', 'y'])
        
        if df.empty:
            st.error("No valid data found after cleaning.")
            return pd.DataFrame()
        
        # Sort by date
        df = df.sort_values('ds')
        
        # Add technical indicators if enabled
        if use_technical_indicators:
            temp_df = df.set_index('ds')[['y']].copy()
            temp_df.columns = ['Close']
            
            temp_df = calculate_technical_indicators(
                temp_df, 
                use_sma=use_sma, 
                use_ema=use_ema, 
                use_rsi=use_rsi, 
                use_macd=use_macd
            )
            
            # Merge indicators back to original data
            for col in temp_df.columns:
                if col != 'Close':
                    df[col] = temp_df[col].values
        
        return df
        
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return pd.DataFrame()

# Function to calculate metrics
def calculate_metrics(actual, predicted):
    metrics = {}
    
    if len(actual) > 0 and len(predicted) > 0:
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        metrics['RMSE'] = np.sqrt(mean_squared_error(actual, predicted))
        metrics['MAE'] = mean_absolute_error(actual, predicted)
        
        # MAPE (avoid division by zero)
        if np.any(actual == 0):
            metrics['MAPE'] = float('nan')
        else:
            metrics['MAPE'] = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        metrics['R2'] = r2_score(actual, predicted)
    
    return metrics

# Linear regression function
def linear_regression_forecast(df, forecast_days):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['y'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_X = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    forecast = model.predict(future_X)
    
    return forecast

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "📈 Historical Data", "🤖 Models", "🔮 Forecast", "📋 Evaluation", "📝 Conclusion"])

with tab1:
    st.header("Analysis Overview")
    
    if use_csv:  # Check if CSV is selected
        if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
            st.markdown(f"""
            You are analyzing data from a custom CSV file.
            - Forecast: {forecast_days} days
            - Active models: {"ARIMA, " if use_arima else ""}{"Holt-Winters, " if use_holtwinters else ""}{"Prophet, " if use_prophet else ""}{"LSTM, " if use_lstm else ""}{"ETS, " if use_ets else ""}{"Theta, " if use_theta else ""}{"Linear Regression" if use_linear else ""}
            - Technical indicators: {"Activated" if use_technical_indicators else "Deactivated"}
            - CPU cores: {cpu_cores}
            """)
        else:
            st.warning("Please upload a CSV file in the sidebar.")
    else:
        # Only show crypto-specific info if CSV is NOT used
        st.markdown(f"""
        You are analyzing **{crypto_name}** against **{fiat_curr}**.
        - Period: {start_date} to {end_date}
        - Forecast: {forecast_days} days
        - Active models: {"ARIMA, " if use_arima else ""}{"Holt-Winters, " if use_holtwinters else ""}{"Prophet, " if use_prophet else ""}{"LSTM, " if use_lstm else ""}{"ETS, " if use_ets else ""}{"Theta, " if use_theta else ""}{"Linear Regression" if use_linear else ""}
        - Technical indicators: {"Activated" if use_technical_indicators else "Deactivated"}
        - CPU cores: {cpu_cores}
        """)
    
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
        st.session_state.analysis_done = False

with tab2:
    st.header("Historical Data")
    
    if 'run_analysis' in st.session_state and st.session_state.run_analysis:
        with st.spinner("Loading historical data..."):
            if use_csv:
                if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
                    df = load_csv_data(st.session_state.uploaded_file)
                    if df is not None and not df.empty:
                        df_raw = df.set_index('ds')[['y']].copy()
                        df_raw.columns = ['Close']
                        ticker = "Custom CSV Data"
                    else:
                        st.error("⚠️ CSV file is empty or in incorrect format.")
                        st.stop()
                else:
                    st.error("⚠️ No CSV file uploaded.")
                    st.stop()
            else:
                df_raw = download_data(ticker, start_date, end_date)
                
                if df_raw.empty or 'Close' not in df_raw.columns:
                    st.error("⚠️ No data available for this currency pair.")
                    st.stop()
                else:
                    df = df_raw[['Close']].reset_index()
                    df.columns = ['ds', 'y']
                    df['ds'] = pd.to_datetime(df['ds'])
            
            # Display data
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Price Chart")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df['ds'], df['y'], label='Price', color='blue')
                ax.set_title(f"{ticker} - Historical Prices")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Statistics")
                if not df.empty and len(df) > 0:
                    st.metric("Starting price", f"{df['y'].iloc[0]:.4f}" if len(df) > 0 else "N/A")
                    st.metric("Ending price", f"{df['y'].iloc[-1]:.4f}" if len(df) > 0 else "N/A")
                    st.metric("Maximum price", f"{df['y'].max():.4f}" if len(df) > 0 else "N/A")
                    st.metric("Minimum price", f"{df['y'].min():.4f}" if len(df) > 0 else "N/A")
                    st.metric("Average price", f"{df['y'].mean():.4f}" if len(df) > 0 else "N/A")
                    st.metric("Standard deviation", f"{df['y'].std():.4f}" if len(df) > 0 else "N/A")
                else:
                    st.warning("No data available for statistics.")
            
            # Display technical indicators if activated
            if use_technical_indicators:
                st.subheader("Technical Indicators")
                
                indicators_to_show = []
                if use_sma:
                    indicators_to_show.extend(['SMA_7', 'SMA_30'])
                if use_ema:
                    indicators_to_show.extend(['EMA_7', 'EMA_30'])
                if use_rsi:
                    indicators_to_show.append('RSI')
                if use_macd:
                    indicators_to_show.extend(['MACD', 'MACD_Signal'])
                
                for indicator in indicators_to_show:
                    if indicator in df.columns:
                        fig, ax = plt.subplots(figsize=(10, 3))
                        ax.plot(df['ds'], df[indicator], label=indicator, color='green')
                        ax.set_title(f"{indicator} for {ticker}")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Value")
                        ax.grid(alpha=0.3)
                        st.pyplot(fig)
            
            # Display raw data
            st.subheader("Data")
            display_df = df.copy()
            display_df.columns = ['Date', 'Price'] + list(display_df.columns[2:])
            st.dataframe(display_df.style.format({"Price": "{:.4f}"}), use_container_width=True)
            
            # Save data to session state
            st.session_state.df = df
            st.session_state.df_raw = df_raw
            st.session_state.ticker = ticker
            st.session_state.data_loaded = True

with tab3:
    st.header("Model Configuration")
    
    if 'data_loaded' not in st.session_state:
        st.info("Please load data in the 'Historical Data' tab first.")
    else:
        st.success("Data successfully loaded. Models are ready for training.")
        
        # Display model information
        model_info = ""
        if use_arima:
            model_info += """
            - **ARIMA**: AutoRegressive Integrated Moving Average - good for stationary series with linear patterns
            """
        if use_holtwinters:
            model_info += """
            - **Holt-Winters**: Exponential smoothing with trend - good for series with trend
            """
        if use_prophet:
            model_info += """
            - **Prophet**: Model developed by Facebook - good for series with seasonality and trends
            """
        if use_lstm:
            model_info += """
            - **LSTM**: Long Short-Term Memory neural network (PyTorch implementation) - good for nonlinear patterns and long-term dependencies
            """
        if use_ets:
            model_info += """
            - **ETS**: Error, Trend, Seasonality model - good for data with trend and seasonality
            """
        if use_theta:
            model_info += """
            - **Theta**: Theta method - good for various types of time series
            """
        if use_linear:
            model_info += """
            - **Linear Regression**: Simple model that assumes linear trend in data
            """
        
        if model_info:
            st.info(f"Information about selected models:{model_info}")
        
        if st.button("🎯 Train Models", type="primary", use_container_width=True):
            st.session_state.train_models = True

with tab4:
    st.header("Forecast Results")
    
    if 'train_models' in st.session_state and st.session_state.train_models:
        # Function to train models
        def train_models():
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = {}
            model_performance = {}
            
            df = st.session_state.df
            df_train = df.copy()
            last_date = df_train['ds'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
            
            total_models = sum([use_arima, use_holtwinters, use_prophet, use_lstm, 
                               use_ets, use_theta, use_linear])
            if total_models == 0:
                st.error("No models selected. Please select at least one model.")
                return {}, {}, future_dates
                
            current_progress = 0
            
            # ARIMA model
            if use_arima:
                try:
                    status_text.text("Training ARIMA model...")
                    arima_model = AutoTS(
                        forecast_length=forecast_days, 
                        frequency='D', 
                        model_list=['ARIMA'], 
                        ensemble=None,
                        n_jobs=cpu_cores
                    )
                    arima_model = arima_model.fit(df_train, date_col='ds', value_col='y', id_col=None)
                    arima_forecast = arima_model.predict().forecast
                    arima_forecast.index = future_dates
                    results['ARIMA'] = arima_forecast['y'].values
                    
                    # Model evaluation
                    if len(df_train) >= forecast_days:
                        actual_values = df_train['y'].iloc[-forecast_days:].values
                        predicted_values = arima_forecast['y'].values[:len(actual_values)]
                        model_performance['ARIMA'] = calculate_metrics(actual_values, predicted_values)
                    else:
                        model_performance['ARIMA'] = {'RMSE': float('nan'), 'MAE': float('nan'), 'MAPE': float('nan'), 'R2': float('nan')}
                    
                    current_progress += 1
                    progress_bar.progress(current_progress / total_models)
                except Exception as e:
                    st.error(f"ARIMA failed: {str(e)}")
            
            # Holt-Winters model
            if use_holtwinters:
                try:
                    status_text.text("Training Holt-Winters model...")
                    hw_model = ExponentialSmoothing(df_train['y'], trend='add', seasonal=None).fit()
                    hw_forecast = hw_model.forecast(forecast_days)
                    results['Holt-Winters'] = hw_forecast.values
                    
                    # Model evaluation
                    if len(df_train) >= forecast_days:
                        actual_values = df_train['y'].iloc[-forecast_days:].values
                        predicted_values = hw_forecast.values[:len(actual_values)]
                        model_performance['Holt-Winters'] = calculate_metrics(actual_values, predicted_values)
                    else:
                        model_performance['Holt-Winters'] = {'RMSE': float('nan'), 'MAE': float('nan'), 'MAPE': float('nan'), 'R2': float('nan')}
                    
                    current_progress += 1
                    progress_bar.progress(current_progress / total_models)
                except Exception as e:
                    st.error(f"Holt-Winters failed: {str(e)}")
            
            # Prophet model
            if use_prophet:
                try:
                    status_text.text("Training Prophet model...")
                    prophet_model = Prophet()
                    prophet_model.fit(df_train)
                    future_df = pd.DataFrame({'ds': future_dates})
                    prophet_forecast = prophet_model.predict(future_df)
                    results['Prophet'] = prophet_forecast['yhat'].values
                    
                    # Model evaluation
                    if len(df_train) >= forecast_days:
                        actual_values = df_train['y'].iloc[-forecast_days:].values
                        predicted_values = prophet_forecast['yhat'].values[:len(actual_values)]
                        model_performance['Prophet'] = calculate_metrics(actual_values, predicted_values)
                    else:
                        model_performance['Prophet'] = {'RMSE': float('nan'), 'MAE': float('nan'), 'MAPE': float('nan'), 'R2': float('nan')}
                    
                    current_progress += 1
                    progress_bar.progress(current_progress / total_models)
                except Exception as e:
                    st.error(f"Prophet failed: {str(e)}")
            
            # LSTM model (PyTorch implementation)
            if use_lstm:
                try:
                    status_text.text("Training LSTM model (PyTorch)...")
                    
                    # Use PyTorch LSTM implementation
                    lstm_forecast = lstm_forecast_pytorch(
                        df_train, 
                        forecast_days, 
                        lstm_sequence_length,
                        lstm_hidden_size,
                        lstm_num_layers,
                        lstm_epochs,
                        lstm_learning_rate,
                        lstm_batch_size
                    )
                    
                    results['LSTM'] = lstm_forecast
                    
                    # Model evaluation
                    if len(df_train) >= forecast_days:
                        actual_values = df_train['y'].iloc[-forecast_days:].values
                        predicted_values = lstm_forecast[:len(actual_values)]
                        model_performance['LSTM'] = calculate_metrics(actual_values, predicted_values)
                    else:
                        model_performance['LSTM'] = {'RMSE': float('nan'), 'MAE': float('nan'), 'MAPE': float('nan'), 'R2': float('nan')}
                    
                    current_progress += 1
                    progress_bar.progress(current_progress / total_models)
                except Exception as e:
                    st.error(f"LSTM model failed: {str(e)}")
            
            # ETS model
            if use_ets:
                try:
                    status_text.text("Training ETS model...")
                    
                    ets_model = AutoTS(
                        forecast_length=forecast_days,
                        frequency='D',
                        model_list=['ETS'],
                        ensemble=None,
                        n_jobs=cpu_cores,
                    )
                    
                    ets_model = ets_model.fit(df_train, date_col='ds', value_col='y', id_col=None)
                    ets_forecast = ets_model.predict().forecast
                    ets_forecast.index = future_dates
                    results['ETS'] = ets_forecast['y'].values
                    
                    # Model evaluation
                    if len(df_train) >= forecast_days:
                        actual_values = df_train['y'].iloc[-forecast_days:].values
                        predicted_values = ets_forecast['y'].values[:len(actual_values)]
                        model_performance['ETS'] = calculate_metrics(actual_values, predicted_values)
                    else:
                        model_performance['ETS'] = {'RMSE': float('nan'), 'MAE': float('nan'), 'MAPE': float('nan'), 'R2': float('nan')}
                    
                    current_progress += 1
                    progress_bar.progress(current_progress / total_models)
                except Exception as e:
                    st.error(f"ETS model failed: {str(e)}")
            
            # Theta model
            if use_theta:
                try:
                    status_text.text("Training Theta model...")
                    
                    theta_model = AutoTS(
                        forecast_length=forecast_days,
                        frequency='D',
                        model_list=['Theta'],
                        ensemble=None,
                        n_jobs=cpu_cores,
                    )
                    
                    theta_model = theta_model.fit(df_train, date_col='ds', value_col='y', id_col=None)
                    theta_forecast = theta_model.predict().forecast
                    theta_forecast.index = future_dates
                    results['Theta'] = theta_forecast['y'].values
                    
                    # Model evaluation
                    if len(df_train) >= forecast_days:
                        actual_values = df_train['y'].iloc[-forecast_days:].values
                        predicted_values = theta_forecast['y'].values[:len(actual_values)]
                        model_performance['Theta'] = calculate_metrics(actual_values, predicted_values)
                    else:
                        model_performance['Theta'] = {'RMSE': float('nan'), 'MAE': float('nan'), 'MAPE': float('nan'), 'R2': float('nan')}
                    
                    current_progress += 1
                    progress_bar.progress(current_progress / total_models)
                except Exception as e:
                    st.error(f"Theta model failed: {str(e)}")
            
            # Linear regression model
            if use_linear:
                try:
                    status_text.text("Training Linear Regression...")
                    
                    linear_forecast = linear_regression_forecast(df_train, forecast_days)
                    results['Linear Regression'] = linear_forecast
                    
                    # Model evaluation
                    if len(df_train) >= forecast_days:
                        actual_values = df_train['y'].iloc[-forecast_days:].values
                        predicted_values = linear_forecast[:len(actual_values)]
                        model_performance['Linear Regression'] = calculate_metrics(actual_values, predicted_values)
                    else:
                        model_performance['Linear Regression'] = {'RMSE': float('nan'), 'MAE': float('nan'), 'MAPE': float('nan'), 'R2': float('nan')}
                    
                    current_progress += 1
                    progress_bar.progress(current_progress / total_models)
                except Exception as e:
                    st.error(f"Linear regression failed: {str(e)}")
            
            status_text.text("Training completed!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            return results, model_performance, future_dates
        
        # Run model training
        results, model_performance, future_dates = train_models()
        
        if results:
            # Display results by model
            st.subheader("Individual Forecasts by Model")
            
            for model_name, forecast in results.items():
                with st.expander(f"{model_name} Model", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(st.session_state.df['ds'], st.session_state.df['y'], 
                               label='History', color='blue')
                        ax.plot(future_dates, forecast, label='Forecast', color='red')
                        ax.set_title(f"{model_name} Forecast for {st.session_state.ticker}")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price")
                        ax.grid(alpha=0.3)
                        ax.legend()
                        st.pyplot(fig)
                    
                    with col2:
                        st.metric("Last known price", f"{st.session_state.df['y'].iloc[-1]:.4f}")
                        
                        # Display model performance if available
                        if model_name in model_performance:
                            perf_data = model_performance[model_name]
                            st.metric("RMSE", f"{perf_data.get('RMSE', 'N/A'):.4f}" if not np.isnan(perf_data.get('RMSE', np.nan)) else "N/A")
                            st.metric("MAE", f"{perf_data.get('MAE', 'N/A'):.4f}" if not np.isnan(perf_data.get('MAE', np.nan)) else "N/A")
                            st.metric("MAPE", f"{perf_data.get('MAPE', 'N/A'):.4f}%" if not np.isnan(perf_data.get('MAPE', np.nan)) else "N/A")
                            st.metric("R²", f"{perf_data.get('R2', 'N/A'):.4f}" if not np.isnan(perf_data.get('R2', np.nan)) else "N/A")
            
            # Combined display of all models
            st.subheader("Combined Forecast")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(st.session_state.df['ds'], st.session_state.df['y'], 
                   label='History', color='black', linewidth=2)
            
            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            for i, (model_name, forecast) in enumerate(results.items()):
                ax.plot(future_dates, forecast, label=model_name, color=colors[i % len(colors)], linewidth=1.5)
            
            ax.set_title(f"Combined forecast - {st.session_state.ticker}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            
            # Tabular display of forecast
            st.subheader("Tabular Forecast Display")
            pred_df = pd.DataFrame(results, index=future_dates)
            pred_df.index.name = 'Date'
            st.dataframe(pred_df.style.format("{:.4f}"), use_container_width=True)
            
            st.session_state.results = results
            st.session_state.model_performance = model_performance
            st.session_state.future_dates = future_dates
            st.session_state.analysis_done = True

with tab5:
    st.header("Model Evaluation")
    
    if 'analysis_done' in st.session_state and st.session_state.analysis_done:
        st.subheader("Comparative Model Performance")
        
        # Display metrics in table
        perf_df = pd.DataFrame(st.session_state.model_performance).T
        st.dataframe(perf_df.style.format("{:.4f}"), use_container_width=True)
        
        # Visual performance display
        metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
        for metric in metrics:
            if metric in perf_df.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                valid_data = perf_df[metric].dropna()
                if len(valid_data) > 0:
                    ax.bar(valid_data.index, valid_data.values)
                    ax.set_title(f"{metric} by Models")
                    ax.set_ylabel(metric)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        
        # Best model analysis
        if 'RMSE' in perf_df.columns:
            best_model_rmse = perf_df['RMSE'].idxmin()
            st.info(f"**Best model by RMSE metric:** {best_model_rmse} (RMSE = {perf_df['RMSE'].min():.4f})")
        
        if 'R2' in perf_df.columns:
            best_model_r2 = perf_df['R2'].idxmax()
            st.info(f"**Best model by R² metric:** {best_model_r2} (R² = {perf_df['R2'].max():.4f})")
        
        # Detailed analysis for each model
        st.subheader("Detailed Analysis by Model")
        for model_name, metrics in st.session_state.model_performance.items():
            with st.expander(f"{model_name} - Detailed Analysis"):
                st.write("**Metrics:**")
                for metric_name, metric_value in metrics.items():
                    st.write(f"- {metric_name}: {metric_value:.4f}")
                
                # Additional model-specific analysis
                if model_name == 'LSTM':
                    st.write("""
                    **LSTM Model Analysis (PyTorch):**
                    - Implementation uses PyTorch framework
                    - Suitable for nonlinear patterns and long-term dependencies
                    - Requires more data for good results
                    - Sensitive to hyperparameter tuning
                    - Uses sequence length: {}, hidden size: {}, layers: {}
                    """.format(lstm_sequence_length, lstm_hidden_size, lstm_num_layers))
                elif model_name == 'ARIMA':
                    st.write("""
                    **ARIMA Model Analysis:**
                    - Good for stationary time series
                    - Effective for short-term predictions
                    - Sensitive to changes in trend and seasonality
                    """)
                elif model_name == 'Prophet':
                    st.write("""
                    **Prophet Model Analysis:**
                    - Robust to outliers and missing data
                    - Automatically detects seasonality
                    - Good for data with strong seasonal patterns
                    """)
    else:
        st.info("Run the analysis to see model evaluation.")

with tab6:
    st.header("Conclusion")
    
    if 'analysis_done' in st.session_state and st.session_state.analysis_done:
        st.success("Analysis successfully completed!")
        st.markdown("""
        ### Result Interpretation
        - Different models give different forecasts because they use different approaches
        - Short-term forecasts are usually more accurate than long-term ones
        - Combining multiple models gives better insight into possible future scenarios
        - Always use multiple information sources before making investment decisions
        """)
        
        # Recommendation based on models
        st.subheader("Model Usage Recommendations")
        st.markdown("""
        - **ARIMA**: Good for short-term predictions and stable trends
        - **Holt-Winters**: Effective for data with trend and seasonality
        - **Prophet**: Robust model that works well with different time series
        - **LSTM**: Advanced model for complex patterns, requires more data
        - **ETS**: Good for data with trend and seasonality
        - **Theta**: Good for various types of time series
        - **Linear Regression**: Simple model for data with clear linear trend
        """)
        
        # Recommendations for future improvements
        st.subheader("Recommendations for Future Improvements")
        st.markdown("""
        - Adding social media data for sentiment analysis
        - Including macroeconomic indicators
        - Implementing ensemble models that combine predictions from multiple models
        - Models have their limitations and assumptions
        - Some models require more data for good results
        """)
    
    st.caption("""
    Note: This application is intended for educational purposes only and does not represent 
    financial advice. Always consult a financial advisor before making investment decisions.
    """)

# Footer
st.markdown("---")
st.markdown("https://github.com/Stojanovic94")
