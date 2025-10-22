# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AirQualityPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.sequence_length = 24  # 24 hours of historical data
        self.features = ['pm2_5', 'co2', 'co', 'no2', 'temperature', 'humidity']
        
    def create_synthetic_data(self, n_samples=1000):
        """Create synthetic air quality data for demonstration"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        
        data = {
            'datetime': dates,
            'pm2_5': np.random.normal(35, 15, n_samples).clip(0, 300),
            'co2': np.random.normal(400, 50, n_samples).clip(300, 1000),
            'co': np.random.normal(1, 0.5, n_samples).clip(0, 5),
            'no2': np.random.normal(25, 10, n_samples).clip(0, 100),
            'temperature': np.random.normal(20, 5, n_samples).clip(-10, 40),
            'humidity': np.random.normal(60, 20, n_samples).clip(10, 100)
        }
        
        # Add some seasonality and trends
        for i in range(n_samples):
            # Daily pattern
            hour = dates[i].hour
            data['pm2_5'][i] += 10 * np.sin(2 * np.pi * hour / 24)
            data['temperature'][i] += 5 * np.sin(2 * np.pi * hour / 24)
            data['humidity'][i] += 10 * np.cos(2 * np.pi * hour / 24)
            
            # Weekly pattern
            day_of_week = dates[i].dayofweek
            if day_of_week < 5:  # Weekdays
                data['co2'][i] += 50
                data['co'][i] += 0.2
                data['no2'][i] += 5
        
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for LSTM"""
        # Handle missing values
        df = df.fillna(method='ffill')
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(df[self.features])
        
        return scaled_data
    
    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(len(self.features))  # Output layer for all features
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train the LSTM model"""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=0.0001
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict_future(self, data, n_steps=24):
        """Predict future values"""
        predictions = []
        current_sequence = data[-self.sequence_length:].copy()
        
        for _ in range(n_steps):
            # Predict next step
            next_pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, len(self.features)), verbose=0)
            predictions.append(next_pred[0])
            
            # Update sequence for next prediction
            current_sequence = np.vstack([current_sequence[1:], next_pred[0]])
        
        return np.array(predictions)

def main():
    st.title("🌫️ Air Quality Prediction using LSTM Neural Network")
    st.markdown("Predict future PM2.5, CO2, CO, NO2, Temperature, and Humidity levels")
    
    # Initialize predictor
    predictor = AirQualityPredictor()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your air quality data (CSV)", 
        type=['csv'],
        help="Upload a CSV file with columns: datetime, pm2_5, co2, co, no2, temperature, humidity"
    )
    
    # Prediction parameters
    prediction_hours = st.sidebar.slider(
        "Hours to predict ahead",
        min_value=1,
        max_value=168,  # 1 week
        value=24
    )
    
    train_size = st.sidebar.slider(
        "Training data ratio",
        min_value=0.5,
        max_value=0.9,
        value=0.8,
        step=0.05
    )
    
    # Load or generate data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.info("Using synthetic data instead.")
            df = predictor.create_synthetic_data()
    else:
        st.info("No file uploaded. Using synthetic air quality data for demonstration.")
        df = predictor.create_synthetic_data()
    
    # Display data
    st.subheader("📊 Air Quality Data Overview")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.metric("Total Records", len(df))
        st.metric("Data Start", df.index.min().strftime('%Y-%m-%d'))
        st.metric("Data End", df.index.max().strftime('%Y-%m-%d'))
    
    # Data statistics
    st.subheader("📈 Data Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Time series visualization
    st.subheader("📈 Time Series Visualization")
    selected_feature = st.selectbox(
        "Select feature to visualize:",
        predictor.features
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df[selected_feature], linewidth=1)
    ax.set_title(f'{selected_feature.upper()} Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel(selected_feature)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlations")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    
    # Model training section
    st.subheader("🤖 LSTM Model Training")
    
    if st.button("Train LSTM Model"):
        with st.spinner("Training in progress... This may take a few minutes."):
            try:
                # Preprocess data
                scaled_data = predictor.preprocess_data(df)
                
                # Create sequences
                X, y = predictor.create_sequences(scaled_data)
                
                # Split data
                split_idx = int(len(X) * train_size)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                # Build and train model
                predictor.model = predictor.build_model((predictor.sequence_length, len(predictor.features)))
                
                # Display model architecture
                st.text("Model Architecture:")
                model_summary = []
                predictor.model.summary(print_fn=lambda x: model_summary.append(x))
                st.text("\n".join(model_summary))
                
                # Train model
                history = predictor.train_model(X_train, y_train, X_val, y_val, epochs=100)
                
                # Plot training history
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                ax1.plot(history.history['loss'], label='Training Loss')
                ax1.plot(history.history['val_loss'], label='Validation Loss')
                ax1.set_title('Model Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(history.history['mae'], label='Training MAE')
                ax2.plot(history.history['val_mae'], label='Validation MAE')
                ax2.set_title('Model MAE')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('MAE')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Save model and scaler
                predictor.model.save('air_quality_lstm.h5')
                joblib.dump(predictor.scaler, 'scaler.pkl')
                
                st.success("✅ Model trained and saved successfully!")
                
            except Exception as e:
                st.error(f"Error during training: {e}")
    
    # Prediction section
    st.subheader("🔮 Future Predictions")
    
    if os.path.exists('air_quality_lstm.h5') and os.path.exists('scaler.pkl'):
        # Load model and scaler
        predictor.model = load_model('air_quality_lstm.h5')
        predictor.scaler = joblib.load('scaler.pkl')
        
        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                try:
                    # Preprocess current data
                    scaled_data = predictor.preprocess_data(df)
                    
                    # Generate predictions
                    predictions_scaled = predictor.predict_future(scaled_data, prediction_hours)
                    
                    # Inverse transform predictions
                    predictions = predictor.scaler.inverse_transform(predictions_scaled)
                    
                    # Create prediction dataframe
                    last_date = df.index[-1]
                    future_dates = [last_date + timedelta(hours=i+1) for i in range(prediction_hours)]
                    
                    pred_df = pd.DataFrame(
                        predictions,
                        index=future_dates,
                        columns=[f'predicted_{col}' for col in predictor.features]
                    )
                    
                    # Display predictions
                    st.subheader("📋 Prediction Results")
                    st.dataframe(pred_df.style.format("{:.2f}"), use_container_width=True)
                    
                    # Plot predictions
                    st.subheader("📊 Prediction Visualization")
                    
                    # Select feature to plot
                    plot_feature = st.selectbox(
                        "Select feature to plot predictions:",
                        predictor.features,
                        key="pred_plot"
                    )
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical data (last 7 days)
                    historical_cutoff = last_date - timedelta(days=7)
                    historical_data = df[df.index >= historical_cutoff]
                    
                    ax.plot(historical_data.index, historical_data[plot_feature], 
                           label='Historical', linewidth=2, color='blue')
                    
                    # Plot predictions
                    ax.plot(pred_df.index, pred_df[f'predicted_{plot_feature}'], 
                           label='Predicted', linewidth=2, color='red', linestyle='--')
                    
                    ax.axvline(x=last_date, color='gray', linestyle=':', alpha=0.7, label='Prediction Start')
                    ax.set_title(f'{plot_feature.upper()} - Historical vs Predicted')
                    ax.set_xlabel('Date')
                    ax.set_ylabel(plot_feature)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    
                    st.pyplot(fig)
                    
                    # Feature importance analysis (simplified)
                    st.subheader("📊 All Features Prediction")
                    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                    axes = axes.flatten()
                    
                    for idx, feature in enumerate(predictor.features):
                        axes[idx].plot(historical_data.index, historical_data[feature], 
                                     label='Historical', linewidth=1)
                        axes[idx].plot(pred_df.index, pred_df[f'predicted_{feature}'], 
                                     label='Predicted', linewidth=1, linestyle='--')
                        axes[idx].set_title(feature.upper())
                        axes[idx].set_xlabel('Date')
                        axes[idx].set_ylabel(feature)
                        axes[idx].legend()
                        axes[idx].grid(True, alpha=0.3)
                        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Download predictions
                    csv = pred_df.to_csv()
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name=f"air_quality_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    else:
        st.info("Please train the model first to generate predictions.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Note:** This application uses LSTM neural networks for time series forecasting of air quality parameters.
        For real-world applications, ensure you have sufficient historical data and validate predictions with actual measurements.
        """
    )

if __name__ == "__main__":
    main()
