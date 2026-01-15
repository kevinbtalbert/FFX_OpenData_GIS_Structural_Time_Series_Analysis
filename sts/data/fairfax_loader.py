# ###########################################################################
#
#  Fairfax County Real Estate Data Loader
#  Modified from Cloudera AMP Structural Time Series
#
# ###########################################################################

import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


def load_fairfax_assessed_values(
    assessed_values_path='csvs/Tax_Administration_s_Real_Estate_-_Assessed_Values.csv',
    parcel_data_path='csvs/Tax_Administration_s_Real_Estate_-_Parcel_Data.csv',
    district_column='district',
    aggregate_by='district',
    min_year=None,
    max_year=None
) -> pd.DataFrame:
    """
    Load and prepare Fairfax County real estate assessed values for time series forecasting.
    
    Args:
        assessed_values_path: Path to assessed values CSV
        parcel_data_path: Path to parcel data CSV (for additional metadata)
        district_column: Column name to use for district identification
        aggregate_by: How to aggregate data ('district', 'quadname', or 'county')
        min_year: Minimum tax year to include (optional)
        max_year: Maximum tax year to include (optional)
    
    Returns:
        DataFrame with columns ['ds', 'y', 'district'] where:
            - ds: datetime (tax year as date)
            - y: aggregate assessed value
            - district: district identifier
    """
    
    # Load assessed values
    print(f"Loading assessed values from {assessed_values_path}...")
    df_assessed = pd.read_csv(assessed_values_path)
    
    # Extract district from PARID (first 4 digits)
    df_assessed['district'] = df_assessed['PARID'].str[:4]
    
    # Filter by year if specified
    if min_year:
        df_assessed = df_assessed[df_assessed['TAXYR'] >= min_year]
    if max_year:
        df_assessed = df_assessed[df_assessed['TAXYR'] <= max_year]
    
    # Convert tax year to datetime (use January 1st of each tax year)
    df_assessed['ds'] = pd.to_datetime(df_assessed['TAXYR'].astype(str) + '-01-01')
    
    # Aggregate by district and tax year
    if aggregate_by == 'district':
        grouped = df_assessed.groupby(['ds', 'district'])['APRTOT'].agg(['sum', 'mean', 'count']).reset_index()
        grouped.columns = ['ds', 'district', 'y', 'mean_value', 'property_count']
    elif aggregate_by == 'county':
        grouped = df_assessed.groupby('ds')['APRTOT'].agg(['sum', 'mean', 'count']).reset_index()
        grouped.columns = ['ds', 'y', 'mean_value', 'property_count']
        grouped['district'] = 'COUNTY_TOTAL'
    else:
        raise ValueError(f"aggregate_by must be 'district' or 'county', got {aggregate_by}")
    
    print(f"Loaded {len(grouped)} district-year combinations")
    print(f"Tax years: {df_assessed['TAXYR'].min()} to {df_assessed['TAXYR'].max()}")
    print(f"Districts: {sorted(df_assessed['district'].unique())[:10]}... ({df_assessed['district'].nunique()} total)")
    
    return grouped


def load_district_data(district_id: str, **kwargs) -> pd.DataFrame:
    """
    Load data for a specific district.
    
    Args:
        district_id: District identifier (e.g., '0804', '0601')
        **kwargs: Additional arguments to pass to load_fairfax_assessed_values
    
    Returns:
        DataFrame with columns ['ds', 'y'] for Prophet modeling
    """
    df = load_fairfax_assessed_values(**kwargs)
    df_district = df[df['district'] == district_id][['ds', 'y']].copy()
    
    if len(df_district) == 0:
        raise ValueError(f"No data found for district {district_id}")
    
    return df_district.sort_values('ds').reset_index(drop=True)


def get_available_districts(
    assessed_values_path='csvs/Tax_Administration_s_Real_Estate_-_Assessed_Values.csv'
) -> List[str]:
    """
    Get list of available districts in the data.
    
    Returns:
        List of district identifiers
    """
    df = pd.read_csv(assessed_values_path, usecols=['PARID'], nrows=100000)
    districts = df['PARID'].str[:4].unique()
    return sorted(districts.tolist())


def get_district_summary(
    assessed_values_path='csvs/Tax_Administration_s_Real_Estate_-_Assessed_Values.csv'
) -> pd.DataFrame:
    """
    Get summary statistics for each district.
    
    Returns:
        DataFrame with district-level summary statistics
    """
    df = pd.read_csv(assessed_values_path)
    df['district'] = df['PARID'].str[:4]
    
    summary = df.groupby('district').agg({
        'APRTOT': ['sum', 'mean', 'count'],
        'TAXYR': ['min', 'max']
    }).reset_index()
    
    summary.columns = ['district', 'total_value', 'mean_value', 'property_count', 'min_year', 'max_year']
    
    return summary.sort_values('total_value', ascending=False)


def prepare_prophet_data(
    district_id: Optional[str] = None,
    aggregate_by: str = 'district',
    **kwargs
) -> pd.DataFrame:
    """
    Prepare data in Prophet format (ds, y columns).
    
    Args:
        district_id: Specific district to prepare (None for county-level)
        aggregate_by: Aggregation level
        **kwargs: Additional arguments
    
    Returns:
        DataFrame ready for Prophet modeling with ['ds', 'y'] columns
    """
    if district_id:
        df = load_district_data(district_id, aggregate_by='district', **kwargs)
    else:
        df = load_fairfax_assessed_values(aggregate_by='county', **kwargs)
        df = df[['ds', 'y']].copy()
    
    return df


def add_external_regressors(
    df: pd.DataFrame,
    regressor_data: Optional[Dict[str, pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Add external regressors to the dataframe for Phase 2.
    
    This is a placeholder function for adding external regressors like CPI, inflation, etc.
    
    Args:
        df: Base dataframe with 'ds' column
        regressor_data: Dictionary mapping regressor names to dataframes with ['ds', 'value'] columns
    
    Returns:
        DataFrame with additional regressor columns
    """
    if regressor_data is None:
        return df
    
    df_with_regressors = df.copy()
    
    for regressor_name, regressor_df in regressor_data.items():
        # Merge regressor data on date
        regressor_df = regressor_df.rename(columns={'value': regressor_name})
        df_with_regressors = df_with_regressors.merge(
            regressor_df[['ds', regressor_name]],
            on='ds',
            how='left'
        )
        
        # Forward fill missing values
        df_with_regressors[regressor_name] = df_with_regressors[regressor_name].fillna(method='ffill')
    
    return df_with_regressors


if __name__ == '__main__':
    # Test the loader
    print("Testing Fairfax data loader...")
    
    # Get available districts
    print("\nAvailable districts:")
    districts = get_available_districts()
    print(f"Found {len(districts)} districts")
    print(f"Sample districts: {districts[:10]}")
    
    # Get district summary
    print("\nDistrict summary:")
    summary = get_district_summary()
    print(summary.head(10))
    
    # Load data for a specific district
    if len(districts) > 0:
        test_district = districts[0]
        print(f"\nLoading data for district {test_district}:")
        df = load_district_data(test_district)
        print(df.head())
        print(f"\nShape: {df.shape}")
