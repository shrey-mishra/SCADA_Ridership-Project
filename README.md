
# Time Series Forecasting with N-BEATS for Transaction Count Prediction

## Overview

This project leverages **N-BEATS**, a cutting-edge deep learning model, to predict future transaction counts based on historical data. Designed for time series forecasting, the model processes large datasets to generate accurate hourly forecasts. With support for dynamic lookback windows and dropout regularization, the model is robust, scalable, and capable of handling extensive datasets.

The repository provides a full implementation of data preprocessing, model training, and future forecasting, with efficient handling of large datasets.

## Features

- **Data Preprocessing**: 
  - Handles large datasets with over 20 million records.
  - Resamples data into hourly intervals for time series forecasting.
  - Cleans and processes timestamps, filtering invalid entries.
  
- **N-BEATS Model Architecture**:
  - Implements a deep learning model for time series forecasting using **PyTorch**.
  - Includes multiple hidden layers and dropout to improve generalization.
  - Supports multiple lookback windows for flexible input sequences.

- **Forecasting**:
  - Predicts future transaction counts over specified time ranges.
  - Saves the results in CSV format for further analysis.
  - Filters predictions for specific business hours (e.g., between 5 AM and 11 PM).

## Project Structure

- **`nbeats_model.pth`**: Pre-trained model weights.
- **`scaler.save`**: Saved scaler object for normalizing input data.
- **`forecast_future_transactions.py`**: Script for forecasting future transactions.
- **`data/`**: Folder for raw transaction data.
- **`results/`**: Folder for storing forecast results.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:
- **Python 3.x**
- **PyTorch**
- **Pandas**
- **NumPy**
- **Joblib**

Install dependencies with:
```bash
pip install torch pandas numpy joblib
```

### Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/nbeats-transaction-forecast.git
   cd nbeats-transaction-forecast
   ```

2. **Prepare the Data**:
   Place your transaction data CSV file in the `data/` directory or load it from an external source (e.g., S3).

3. **Run the Forecasting Script**:
   Use the provided script to load the pre-trained N-BEATS model and make predictions. Modify the script parameters as necessary:
   ```python
   # Example to generate forecasts for a specific date range
   forecast_df = forecast_future_transactions(
       model_path='nbeats_model.pth',
       scaler_path='scaler.save',
       hourly_transaction=hourly_transaction_data,
       start_date='2024-10-05 00:00:00',
       end_date='2024-10-30 23:00:00'
   )
   ```

4. **Save Forecasts**:
   The forecast results are automatically saved to the `results/` folder as a CSV file.

### Example Workflow

1. Load and preprocess your dataset:
   ```python
   df = pd.read_csv('data/your_transaction_data.csv')
   # Apply preprocessing and resampling to hourly intervals
   ```
   
2. Train or load the N-BEATS model for prediction:
   ```python
   model = NBeatsModel(input_size=LOOKBACK)
   model.load_state_dict(torch.load('nbeats_model.pth'))
   ```

3. Predict future transactions for a given time range:
   ```python
   forecast_df = forecast_future_transactions(
       model_path='nbeats_model.pth',
       scaler_path='scaler.save',
       hourly_transaction=hourly_transaction,
       start_date='2024-10-05 00:00:00',
       end_date='2024-10-30 23:00:00'
   )
   ```

4. Filter and save the results for analysis:
   ```python
   filtered_forecast = filter_by_hour_range(forecast_df, start_hour=5, end_hour=23)
   filtered_forecast.to_csv('results/future_forecasts_filtered.csv', index=False)
   ```

### Model Details

The **N-BEATS model** is designed for univariate time series forecasting. It consists of several fully connected layers, with each block focusing on the trend or seasonal component of the data. Dropout layers are included to prevent overfitting and improve generalization.

The architecture is optimized for transaction count forecasting over time, ensuring accurate predictions even with fluctuating historical trends.

## Future Enhancements

Some potential areas for further improvement:
- **Hyperparameter Tuning**: Optimize the model's learning rate, batch size, and other hyperparameters.
- **Multivariate Forecasting**: Incorporate additional features (e.g., external factors influencing transactions).
- **Model Evaluation**: Add metrics like MAE, RMSE, or MAPE for evaluating forecast accuracy.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.

---

Feel free to adjust the description and details as necessary!
