# ###########################################################################
#
#  External Regressors Module (Phase 2)
#  Support for adding CPI, inflation, and other economic indicators
#
# ###########################################################################

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import os


class ExternalRegressorManager:
    """
    Manager for external regressors like CPI, inflation, interest rates, etc.
    Provides a modular interface for adding economic indicators to forecasting models.
    """
    
    def __init__(self, data_dir='data/regressors'):
        """
        Initialize the external regressor manager.
        
        Args:
            data_dir: Directory to store regressor data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.regressors = {}
    
    def add_regressor(
        self,
        name: str,
        data: pd.DataFrame,
        date_column: str = 'ds',
        value_column: str = 'value'
    ):
        """
        Add an external regressor.
        
        Args:
            name: Name of the regressor (e.g., 'cpi', 'inflation_rate')
            data: DataFrame with date and value columns
            date_column: Name of the date column
            value_column: Name of the value column
        """
        # Standardize column names
        regressor_data = data[[date_column, value_column]].copy()
        regressor_data.columns = ['ds', 'value']
        regressor_data['ds'] = pd.to_datetime(regressor_data['ds'])
        regressor_data = regressor_data.sort_values('ds')
        
        self.regressors[name] = regressor_data
        
        # Save to disk
        filepath = os.path.join(self.data_dir, f'{name}.csv')
        regressor_data.to_csv(filepath, index=False)
        print(f"Added regressor '{name}' with {len(regressor_data)} data points")
    
    def load_regressor(self, name: str) -> Optional[pd.DataFrame]:
        """
        Load a regressor from disk.
        
        Args:
            name: Name of the regressor
        
        Returns:
            DataFrame with regressor data or None if not found
        """
        if name in self.regressors:
            return self.regressors[name]
        
        filepath = os.path.join(self.data_dir, f'{name}.csv')
        if os.path.exists(filepath):
            data = pd.read_csv(filepath, parse_dates=['ds'])
            self.regressors[name] = data
            return data
        
        return None
    
    def get_all_regressors(self) -> Dict[str, pd.DataFrame]:
        """
        Get all available regressors.
        
        Returns:
            Dictionary mapping regressor names to dataframes
        """
        # Load any regressors from disk that aren't in memory
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                name = filename[:-4]
                if name not in self.regressors:
                    self.load_regressor(name)
        
        return self.regressors
    
    def interpolate_regressor(
        self,
        name: str,
        target_dates: pd.DatetimeIndex,
        method: str = 'linear'
    ) -> pd.Series:
        """
        Interpolate regressor values for specific dates.
        
        Args:
            name: Name of the regressor
            target_dates: Dates to interpolate for
            method: Interpolation method ('linear', 'ffill', 'bfill')
        
        Returns:
            Series with interpolated values
        """
        regressor_data = self.load_regressor(name)
        if regressor_data is None:
            raise ValueError(f"Regressor '{name}' not found")
        
        # Create a dataframe with target dates
        target_df = pd.DataFrame({'ds': target_dates})
        
        # Merge with regressor data
        merged = target_df.merge(regressor_data, on='ds', how='left')
        
        # Interpolate missing values
        if method == 'linear':
            merged['value'] = merged['value'].interpolate(method='linear')
        elif method == 'ffill':
            merged['value'] = merged['value'].fillna(method='ffill')
        elif method == 'bfill':
            merged['value'] = merged['value'].fillna(method='bfill')
        
        return merged['value']


# Placeholder functions for loading common economic indicators
# These can be implemented in Phase 2 with actual data sources

def load_cpi_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    source: str = 'fred'  # Federal Reserve Economic Data
) -> pd.DataFrame:
    """
    Load Consumer Price Index (CPI) data.
    
    This is a placeholder function. In Phase 2, implement actual data loading
    from sources like FRED API, BLS API, etc.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        source: Data source ('fred', 'bls', etc.)
    
    Returns:
        DataFrame with columns ['ds', 'value']
    """
    # Placeholder implementation
    # TODO: Implement actual CPI data loading in Phase 2
    print("Warning: load_cpi_data is a placeholder. Implement in Phase 2.")
    
    # Generate dummy data for testing
    dates = pd.date_range(start='2020-01-01', end='2025-12-31', freq='MS')
    values = 250 + np.cumsum(np.random.randn(len(dates)) * 2)  # Simulated CPI
    
    return pd.DataFrame({'ds': dates, 'value': values})


def load_inflation_rate(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load inflation rate data.
    
    This is a placeholder function. In Phase 2, implement actual data loading.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
    
    Returns:
        DataFrame with columns ['ds', 'value']
    """
    # Placeholder implementation
    print("Warning: load_inflation_rate is a placeholder. Implement in Phase 2.")
    
    # Generate dummy data for testing
    dates = pd.date_range(start='2020-01-01', end='2025-12-31', freq='MS')
    values = 2.0 + np.random.randn(len(dates)) * 0.5  # Simulated inflation rate (%)
    
    return pd.DataFrame({'ds': dates, 'value': values})


def load_interest_rate(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    rate_type: str = 'federal_funds'
) -> pd.DataFrame:
    """
    Load interest rate data.
    
    This is a placeholder function. In Phase 2, implement actual data loading.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        rate_type: Type of interest rate ('federal_funds', 'mortgage_30yr', etc.)
    
    Returns:
        DataFrame with columns ['ds', 'value']
    """
    # Placeholder implementation
    print("Warning: load_interest_rate is a placeholder. Implement in Phase 2.")
    
    # Generate dummy data for testing
    dates = pd.date_range(start='2020-01-01', end='2025-12-31', freq='MS')
    values = 3.0 + np.random.randn(len(dates)) * 0.8  # Simulated interest rate (%)
    
    return pd.DataFrame({'ds': dates, 'value': values})


def load_unemployment_rate(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    region: str = 'national'
) -> pd.DataFrame:
    """
    Load unemployment rate data.
    
    This is a placeholder function. In Phase 2, implement actual data loading.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        region: Geographic region ('national', 'state', 'county')
    
    Returns:
        DataFrame with columns ['ds', 'value']
    """
    # Placeholder implementation
    print("Warning: load_unemployment_rate is a placeholder. Implement in Phase 2.")
    
    # Generate dummy data for testing
    dates = pd.date_range(start='2020-01-01', end='2025-12-31', freq='MS')
    values = 5.0 + np.random.randn(len(dates)) * 1.0  # Simulated unemployment rate (%)
    
    return pd.DataFrame({'ds': dates, 'value': values})


# Example usage and testing
if __name__ == '__main__':
    print("Testing External Regressor Manager...")
    
    # Initialize manager
    manager = ExternalRegressorManager()
    
    # Load placeholder data
    print("\nLoading placeholder economic indicators...")
    cpi_data = load_cpi_data()
    inflation_data = load_inflation_rate()
    interest_data = load_interest_rate()
    
    # Add regressors
    print("\nAdding regressors to manager...")
    manager.add_regressor('cpi', cpi_data)
    manager.add_regressor('inflation_rate', inflation_data)
    manager.add_regressor('interest_rate', interest_data)
    
    # Get all regressors
    print("\nAvailable regressors:")
    all_regressors = manager.get_all_regressors()
    for name, data in all_regressors.items():
        print(f"  - {name}: {len(data)} data points")
    
    # Test interpolation
    print("\nTesting interpolation...")
    target_dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='MS')
    interpolated_cpi = manager.interpolate_regressor('cpi', target_dates)
    print(f"Interpolated CPI for {len(target_dates)} dates")
    print(interpolated_cpi.head())
    
    print("\n✓ External regressor module ready for Phase 2 implementation!")
