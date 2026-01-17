#!/usr/bin/env python3
"""
Projection API for Fairfax County Real Estate
Simplified approach for single-year data analysis
"""

import os
import json
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

from sts.data.fairfax_loader import load_fairfax_raw_data
from sts.models.simple_projections import SimpleProjectionModel, compare_districts, generate_scenario_analysis


class ProjectionAPI:
    """
    API for generating real estate value projections and analysis.
    Works with single-year data.
    """
    
    def __init__(self, data_dir='data'):
        """Initialize the Projection API."""
        self.data_dir = data_dir
        self.models_dir = os.path.join(data_dir, 'models')
        self.forecasts_dir = os.path.join(data_dir, 'forecasts')
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.forecasts_dir, exist_ok=True)
        
        # Load base data
        self.df = load_fairfax_raw_data()
        self.models = {}
        self.projections = {}
        
    def train_model(
        self,
        district_id: Optional[str] = None,
        periods_ahead: int = 6,
        growth_rate: float = 0.03
    ):
        """
        Train a projection model for a district or county-wide.
        
        Args:
            district_id: District ID (None for county-wide)
            periods_ahead: Number of months to project
            growth_rate: Annual growth rate assumption
        """
        model_key = district_id if district_id else 'county'
        print(f"Training model for {model_key}...")
        
        # Create and fit model
        model = SimpleProjectionModel(base_growth_rate=growth_rate)
        model.fit(self.df, district_id=district_id)
        
        # Generate projections
        projections = model.project(periods=periods_ahead, frequency='M')
        
        # Store
        self.models[model_key] = model
        self.projections[model_key] = projections
        
        # Save
        self._save_projection(projections, model_key)
        
        print(f" Model trained for {model_key}")
        return model
    
    def generate_forecast(
        self,
        district_id: Optional[str] = None,
        periods_ahead: int = 6,
        growth_rate: float = 0.03
    ) -> pd.DataFrame:
        """
        Generate forecast/projection.
        
        Args:
            district_id: District ID (None for county-wide)
            periods_ahead: Number of months to project
            growth_rate: Annual growth rate
            
        Returns:
            DataFrame with projections
        """
        model_key = district_id if district_id else 'county'
        
        # Check if model exists
        if model_key not in self.models:
            # Train if not exists
            self.train_model(district_id, periods_ahead, growth_rate)
        
        return self.projections[model_key]
    
    def get_forecast(self, district_id: Optional[str] = None) -> pd.DataFrame:
        """Get existing forecast."""
        model_key = district_id if district_id else 'county'
        
        if model_key in self.projections:
            return self.projections[model_key]
        
        # Try to load from disk
        forecast = self._load_projection(model_key)
        if forecast is not None:
            self.projections[model_key] = forecast
            return forecast
        
        raise ValueError(f"No forecast found for {model_key}. Run generate_forecast() first.")
    
    def get_forecast_summary(
        self,
        district_id: Optional[str] = None,
        periods_ahead: int = 6
    ) -> Dict:
        """
        Get forecast summary with key metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        model_key = district_id if district_id else 'county'
        
        # Get or generate forecast
        try:
            forecast = self.get_forecast(district_id)
        except ValueError:
            self.generate_forecast(district_id, periods_ahead)
            forecast = self.get_forecast(district_id)
        
        # Get model
        model = self.models.get(model_key)
        if model is None:
            raise ValueError(f"Model not found for {model_key}")
        
        # Calculate summary
        predictions = []
        for _, row in forecast.iterrows():
            predictions.append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'predicted_value': float(row['yhat']),
                'lower_bound': float(row['yhat_lower']),
                'upper_bound': float(row['yhat_upper']),
                'period': int(row['period'])
            })
        
        return {
            'district_id': district_id if district_id else 'COUNTY_TOTAL',
            'base_year': model.base_year,
            'base_value': float(model.total_value),
            'property_count': int(model.property_count),
            'mean_value': float(model.mean_value),
            'median_value': float(model.median_value),
            'growth_rate': float(model.base_growth_rate),
            'periods': len(predictions),
            'total_predicted_value': float(forecast['yhat'].sum()),
            'mean_predicted_value': float(forecast['yhat'].mean()),
            'final_predicted_value': float(forecast['yhat'].iloc[-1]),
            'total_growth_pct': float(((forecast['yhat'].iloc[-1] / model.total_value) - 1) * 100),
            'predictions': predictions
        }
    
    def get_district_comparison(self, top_n: int = 15) -> pd.DataFrame:
        """Get comparison of top districts."""
        return compare_districts(self.df, top_n=top_n)
    
    def get_risk_analysis(self, district_id: Optional[str] = None) -> Dict:
        """Get risk analysis for a district."""
        model_key = district_id if district_id else 'county'
        
        if model_key not in self.models:
            raise ValueError(f"Model not trained for {model_key}")
        
        model = self.models[model_key]
        return model.analyze_risk()
    
    def get_scenario_analysis(
        self,
        district_id: Optional[str] = None,
        scenarios: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Generate scenario analysis.
        
        Args:
            district_id: District ID
            scenarios: List of scenarios with 'name' and 'growth_rate'
        """
        if scenarios is None:
            scenarios = [
                {'name': 'Conservative', 'growth_rate': 0.02},
                {'name': 'Baseline', 'growth_rate': 0.03},
                {'name': 'Optimistic', 'growth_rate': 0.045},
                {'name': 'Recession', 'growth_rate': -0.02}
            ]
        
        # Get base value
        if district_id:
            base_value = self.df[self.df['district'] == district_id]['APRTOT'].sum()
        else:
            base_value = self.df['APRTOT'].sum()
        
        return generate_scenario_analysis(base_value, scenarios)
    
    def _save_projection(self, projection: pd.DataFrame, model_key: str):
        """Save projection to disk."""
        filepath = os.path.join(self.forecasts_dir, f'{model_key}_projection.csv')
        projection.to_csv(filepath, index=False)
        print(f"Projection saved to {filepath}")
    
    def _load_projection(self, model_key: str) -> Optional[pd.DataFrame]:
        """Load projection from disk."""
        filepath = os.path.join(self.forecasts_dir, f'{model_key}_projection.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['ds'] = pd.to_datetime(df['ds'])
            return df
        return None


# Convenience function for compatibility
def get_forecast_summary(district_id: Optional[str] = None, periods_ahead: int = 6) -> Dict:
    """Get forecast summary (compatibility function)."""
    api = ProjectionAPI()
    return api.get_forecast_summary(district_id, periods_ahead)
