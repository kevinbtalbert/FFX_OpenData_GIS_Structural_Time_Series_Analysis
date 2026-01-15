# ###########################################################################
#
#  Forecasting API for Fairfax County Real Estate
#  Modular interface for generating and retrieving forecasts
#
# ###########################################################################

import os
import json
import pickle
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prophet import Prophet

from sts.data.fairfax_loader import (
    load_district_data,
    prepare_prophet_data,
    get_available_districts,
    add_external_regressors
)


class ForecastAPI:
    """
    Modular API for generating and retrieving real estate value forecasts.
    Supports district-level and county-level forecasting with external regressors.
    """
    
    def __init__(self, models_dir='data/models', forecasts_dir='data/forecasts'):
        """
        Initialize the Forecast API.
        
        Args:
            models_dir: Directory to save/load trained models
            forecasts_dir: Directory to save/load forecasts
        """
        self.models_dir = models_dir
        self.forecasts_dir = forecasts_dir
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(forecasts_dir, exist_ok=True)
        
        self.models = {}
        self.forecasts = {}
    
    def train_model(
        self,
        district_id: Optional[str] = None,
        periods_ahead: int = 6,
        yearly_seasonality: int = 10,
        changepoint_prior_scale: float = 0.05,
        external_regressors: Optional[Dict[str, pd.DataFrame]] = None,
        save_model: bool = True
    ) -> Prophet:
        """
        Train a Prophet model for a specific district or county-level.
        
        Args:
            district_id: District identifier (None for county-level)
            periods_ahead: Number of periods to forecast ahead
            yearly_seasonality: Fourier order for yearly seasonality
            changepoint_prior_scale: Flexibility of trend changes
            external_regressors: Dictionary of external regressor data (Phase 2)
            save_model: Whether to save the trained model
        
        Returns:
            Trained Prophet model
        """
        print(f"Training model for {'district ' + district_id if district_id else 'county-level'}...")
        
        # Load and prepare data
        df = prepare_prophet_data(district_id=district_id)
        
        # Add external regressors if provided (Phase 2)
        if external_regressors:
            df = add_external_regressors(df, external_regressors)
        
        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            interval_width=0.95
        )
        
        # Add external regressors to model (Phase 2)
        if external_regressors:
            for regressor_name in external_regressors.keys():
                model.add_regressor(regressor_name)
        
        # Fit model
        model.fit(df)
        
        # Save model
        if save_model:
            model_key = district_id if district_id else 'county'
            self.models[model_key] = model
            self._save_model(model, model_key)
        
        print(f"Model trained successfully!")
        return model
    
    def generate_forecast(
        self,
        district_id: Optional[str] = None,
        periods_ahead: int = 6,
        freq: str = 'MS',  # Month start
        model: Optional[Prophet] = None,
        external_regressors_future: Optional[Dict[str, pd.DataFrame]] = None,
        save_forecast: bool = True
    ) -> pd.DataFrame:
        """
        Generate forecast for a specific district or county-level.
        
        Args:
            district_id: District identifier (None for county-level)
            periods_ahead: Number of periods to forecast ahead
            freq: Frequency of forecast ('MS' for month start, 'AS' for year start)
            model: Pre-trained model (if None, will load from disk)
            external_regressors_future: Future values of external regressors (Phase 2)
            save_forecast: Whether to save the forecast
        
        Returns:
            DataFrame with forecast results
        """
        model_key = district_id if district_id else 'county'
        
        # Load model if not provided
        if model is None:
            model = self._load_model(model_key)
            if model is None:
                raise ValueError(f"No trained model found for {model_key}. Please train a model first.")
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods_ahead, freq=freq)
        
        # Add external regressors for future periods (Phase 2)
        if external_regressors_future:
            future = add_external_regressors(future, external_regressors_future)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Save forecast
        if save_forecast:
            self.forecasts[model_key] = forecast
            self._save_forecast(forecast, model_key)
        
        return forecast
    
    def get_forecast(
        self,
        district_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve a saved forecast for a district.
        
        Args:
            district_id: District identifier (None for county-level)
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
        
        Returns:
            DataFrame with forecast results
        """
        model_key = district_id if district_id else 'county'
        
        # Load from memory or disk
        if model_key in self.forecasts:
            forecast = self.forecasts[model_key]
        else:
            forecast = self._load_forecast(model_key)
        
        if forecast is None:
            raise ValueError(f"No forecast found for {model_key}. Please generate a forecast first.")
        
        # Filter by date range if specified
        if start_date:
            forecast = forecast[forecast['ds'] >= pd.to_datetime(start_date)]
        if end_date:
            forecast = forecast[forecast['ds'] <= pd.to_datetime(end_date)]
        
        return forecast
    
    def get_forecast_summary(
        self,
        district_id: Optional[str] = None,
        periods_ahead: int = 6
    ) -> Dict:
        """
        Get a summary of the forecast for easy consumption.
        
        Args:
            district_id: District identifier (None for county-level)
            periods_ahead: Number of future periods to include
        
        Returns:
            Dictionary with forecast summary
        """
        forecast = self.get_forecast(district_id)
        
        # Get only future predictions
        future_forecast = forecast.tail(periods_ahead)
        
        summary = {
            'district': district_id if district_id else 'county',
            'forecast_date': datetime.now().isoformat(),
            'periods_ahead': periods_ahead,
            'predictions': [
                {
                    'date': row['ds'].isoformat(),
                    'predicted_value': float(row['yhat']),
                    'lower_bound': float(row['yhat_lower']),
                    'upper_bound': float(row['yhat_upper']),
                    'confidence_interval': 0.95
                }
                for _, row in future_forecast.iterrows()
            ],
            'total_predicted_value': float(future_forecast['yhat'].sum()),
            'mean_predicted_value': float(future_forecast['yhat'].mean())
        }
        
        return summary
    
    def train_all_districts(
        self,
        top_n: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Prophet]:
        """
        Train models for all districts (or top N by property count).
        
        Args:
            top_n: Train only top N districts by property count (None for all)
            **kwargs: Additional arguments to pass to train_model
        
        Returns:
            Dictionary mapping district IDs to trained models
        """
        from sts.data.fairfax_loader import get_district_summary
        
        # Get district summary
        summary = get_district_summary()
        
        if top_n:
            summary = summary.head(top_n)
        
        models = {}
        for _, row in summary.iterrows():
            district_id = row['district']
            try:
                model = self.train_model(district_id=district_id, **kwargs)
                models[district_id] = model
                print(f"✓ Trained model for district {district_id}")
            except Exception as e:
                print(f"✗ Failed to train model for district {district_id}: {e}")
        
        return models
    
    def _save_model(self, model: Prophet, model_key: str):
        """Save model to disk."""
        filepath = os.path.join(self.models_dir, f'{model_key}_model.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filepath}")
    
    def _load_model(self, model_key: str) -> Optional[Prophet]:
        """Load model from disk."""
        filepath = os.path.join(self.models_dir, f'{model_key}_model.pkl')
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_forecast(self, forecast: pd.DataFrame, model_key: str):
        """Save forecast to disk."""
        filepath = os.path.join(self.forecasts_dir, f'{model_key}_forecast.csv')
        forecast.to_csv(filepath, index=False)
        print(f"Forecast saved to {filepath}")
    
    def _load_forecast(self, model_key: str) -> Optional[pd.DataFrame]:
        """Load forecast from disk."""
        filepath = os.path.join(self.forecasts_dir, f'{model_key}_forecast.csv')
        if os.path.exists(filepath):
            return pd.read_csv(filepath, parse_dates=['ds'])
        return None


# Convenience functions for easy API access

def get_forecast(district_id: str, **kwargs) -> pd.DataFrame:
    """
    Convenience function to get forecast for a district.
    
    Example:
        forecast = get_forecast('0804')
    """
    api = ForecastAPI()
    return api.get_forecast(district_id=district_id, **kwargs)


def get_forecast_summary(district_id: str, **kwargs) -> Dict:
    """
    Convenience function to get forecast summary for a district.
    
    Example:
        summary = get_forecast_summary('0804')
        print(f"Predicted value: ${summary['total_predicted_value']:,.0f}")
    """
    api = ForecastAPI()
    return api.get_forecast_summary(district_id=district_id, **kwargs)


if __name__ == '__main__':
    # Test the API
    print("Testing Forecast API...")
    
    api = ForecastAPI()
    
    # Get available districts
    districts = get_available_districts()
    print(f"\nFound {len(districts)} districts")
    
    # Train a model for the first district (as a test)
    if len(districts) > 0:
        test_district = districts[0]
        print(f"\nTraining model for district {test_district}...")
        
        try:
            model = api.train_model(district_id=test_district, periods_ahead=6)
            
            # Generate forecast
            print(f"\nGenerating forecast...")
            forecast = api.generate_forecast(district_id=test_district, periods_ahead=6)
            
            # Get summary
            summary = api.get_forecast_summary(district_id=test_district)
            print(f"\nForecast Summary:")
            print(json.dumps(summary, indent=2))
            
        except Exception as e:
            print(f"Error: {e}")
