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


def load_fairfax_raw_data(
    assessed_values_path='csvs/Tax_Administration_s_Real_Estate_-_Assessed_Values.csv'
) -> pd.DataFrame:
    """
    Load raw Fairfax County real estate data.
    
    Args:
        assessed_values_path: Path to assessed values CSV
        
    Returns:
        DataFrame with all property records including district
    """
    print(f"Loading data from {assessed_values_path}...")
    df = pd.read_csv(assessed_values_path)
    
    # Extract district from PARID (first 4 digits)
    df['district'] = df['PARID'].astype(str).str[:4]
    
    print(f" Loaded {len(df):,} properties")
    print(f"  Tax Year: {df['TAXYR'].unique()}")
    print(f"  Districts: {df['district'].nunique()}")
    print(f"  Total Assessed Value: ${df['APRTOT'].sum():,.0f}")
    
    return df


def load_fairfax_assessed_values(
    assessed_values_path='csvs/Tax_Administration_s_Real_Estate_-_Assessed_Values.csv',
    parcel_data_path='csvs/Tax_Administration_s_Real_Estate_-_Parcel_Data.csv',
    district_column='district',
    aggregate_by='district',
    min_year=None,
    max_year=None,
    generate_historical=False
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
        generate_historical: If True and only one year exists, generate synthetic historical data
    
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
    
    # Check if we only have one year of data
    unique_years = df_assessed['TAXYR'].nunique()
    print(f"Found {unique_years} unique tax year(s): {sorted(df_assessed['TAXYR'].unique())}")
    
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
    
    # Generate synthetic historical data if needed
    if generate_historical and unique_years == 1:
        print("  Only one year of data detected. Generating synthetic historical data for demonstration...")
        grouped = _generate_historical_data(grouped, aggregate_by)
    
    print(f"Loaded {len(grouped)} district-year combinations")
    if len(grouped) > 0:
        print(f"Date range: {grouped['ds'].min()} to {grouped['ds'].max()}")
        if 'district' in grouped.columns:
            print(f"Districts: {sorted(grouped['district'].unique())[:10]}... ({grouped['district'].nunique()} total)")
    
    return grouped


def _generate_historical_data(df: pd.DataFrame, aggregate_by: str, years_back: int = 10) -> pd.DataFrame:
    """
    Generate synthetic historical data based on current year data.
    Applies realistic growth rates (2-5% annually) with some variance.
    
    Args:
        df: DataFrame with current year data
        aggregate_by: 'district' or 'county'
        years_back: Number of historical years to generate
    
    Returns:
        DataFrame with synthetic historical data appended
    """
    if len(df) == 0:
        return df
    
    # Get the current year
    current_year = df['ds'].max()
    current_year_num = current_year.year
    
    # Generate historical years
    historical_dfs = []
    
    # Annual growth rate: 3% average with some variance
    np.random.seed(42)  # For reproducibility
    
    for year_offset in range(1, years_back + 1):
        # Calculate year
        hist_year = current_year_num - year_offset
        hist_date = pd.Timestamp(f'{hist_year}-01-01')
        
        # Create copy of current data
        hist_df = df.copy()
        hist_df['ds'] = hist_date
        
        # Apply compound growth rate backwards (deflate values)
        # Use 3% average growth with slight variance per district
        if aggregate_by == 'district':
            for district in hist_df['district'].unique():
                mask = hist_df['district'] == district
                # Random growth rate between 2.5% and 4.5% per year
                growth_rate = np.random.uniform(0.025, 0.045)
                deflation_factor = (1 - growth_rate) ** year_offset
                hist_df.loc[mask, 'y'] = hist_df.loc[mask, 'y'] * deflation_factor
                if 'mean_value' in hist_df.columns:
                    hist_df.loc[mask, 'mean_value'] = hist_df.loc[mask, 'mean_value'] * deflation_factor
        else:
            growth_rate = 0.035  # 3.5% for county total
            deflation_factor = (1 - growth_rate) ** year_offset
            hist_df['y'] = hist_df['y'] * deflation_factor
            if 'mean_value' in hist_df.columns:
                hist_df['mean_value'] = hist_df['mean_value'] * deflation_factor
        
        historical_dfs.append(hist_df)
    
    # Combine all data
    all_data = pd.concat([df] + historical_dfs, ignore_index=True)
    all_data = all_data.sort_values(['district', 'ds'] if 'district' in all_data.columns else 'ds')
    
    print(f" Generated {years_back} years of synthetic historical data ({current_year_num - years_back} to {current_year_num})")
    print(f"  Using realistic growth rates (2.5-4.5% annually)")
    
    return all_data.reset_index(drop=True)


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
    
    # Validate data
    print(f"Prepared data shape: {df.shape}")
    print(f"Data preview:\n{df.head()}")
    print(f"Non-null rows: {df.dropna().shape[0]}")
    
    if len(df) < 2:
        raise ValueError(f"Insufficient data: only {len(df)} rows found. Need at least 2 rows for forecasting.")
    
    if df['y'].isna().all():
        raise ValueError("All values are NaN. Check data quality.")
    
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
