#!/usr/bin/env python3
"""
Simple Projection Model for Fairfax County Real Estate
Works with single-year data to provide growth projections and risk analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


class SimpleProjectionModel:
    """
    Simple projection model for real estate values.
    Uses growth rate assumptions and district-level analysis.
    """
    
    def __init__(self, base_growth_rate: float = 0.03):
        """
        Initialize the projection model.
        
        Args:
            base_growth_rate: Annual growth rate assumption (default 3%)
        """
        self.base_growth_rate = base_growth_rate
        self.base_data = None
        self.projections = None
        
    def fit(self, df: pd.DataFrame, district_id: Optional[str] = None):
        """
        Fit the model with base year data.
        
        Args:
            df: DataFrame with columns ['PARID', 'TAXYR', 'APRTOT', 'district']
            district_id: Optional district to filter (None for county-wide)
        """
        if district_id:
            self.base_data = df[df['district'] == district_id].copy()
        else:
            self.base_data = df.copy()
            
        # Calculate statistics
        self.base_year = self.base_data['TAXYR'].max()
        self.total_value = self.base_data['APRTOT'].sum()
        self.property_count = len(self.base_data)
        self.mean_value = self.base_data['APRTOT'].mean()
        self.median_value = self.base_data['APRTOT'].median()
        
        return self
    
    def project(
        self,
        periods: int = 6,
        frequency: str = 'M',
        confidence_interval_pct: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate projections for future periods.
        
        Args:
            periods: Number of periods to project
            frequency: 'M' for monthly, 'Y' for yearly
            
        Returns:
            DataFrame with projections
        """
        if self.base_data is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create date range
        base_date = datetime(self.base_year, 1, 1)
        
        if frequency == 'M':
            dates = [base_date + relativedelta(months=i) for i in range(1, periods + 1)]
            # Monthly growth rate
            monthly_rate = (1 + self.base_growth_rate) ** (1/12) - 1
            growth_factors = [(1 + monthly_rate) ** i for i in range(1, periods + 1)]
        else:  # Yearly
            dates = [base_date + relativedelta(years=i) for i in range(1, periods + 1)]
            growth_factors = [(1 + self.base_growth_rate) ** i for i in range(1, periods + 1)]
        
        # Generate projections (deterministic, no random variance)
        projections = []
        for date, factor in zip(dates, growth_factors):
            projected_value = self.total_value * factor
           
            projections.append({
                'ds': date,
                'yhat': projected_value,
                'yhat_lower': projected_value * (1 - confidence_interval_pct),
                'yhat_upper': projected_value * (1 + confidence_interval_pct),
                'growth_rate': self.base_growth_rate,
                'confidence_interval': confidence_interval_pct,
                'period': len(projections) + 1
            })
        
        self.projections = pd.DataFrame(projections)
        return self.projections
    
    def get_summary(self) -> Dict:
        """Get summary statistics and projections."""
        if self.projections is None:
            raise ValueError("No projections generated. Call project() first.")
        
        return {
            'base_year': self.base_year,
            'base_total_value': self.total_value,
            'base_mean_value': self.mean_value,
            'base_median_value': self.median_value,
            'property_count': self.property_count,
            'growth_rate': self.base_growth_rate,
            'projected_total_value': self.projections['yhat'].iloc[-1],
            'total_growth': ((self.projections['yhat'].iloc[-1] / self.total_value) - 1) * 100,
            'projections': self.projections.to_dict('records')
        }
    
    def analyze_risk(self) -> Dict:
        """
        Analyze revenue risk based on projections.
        
        Returns:
            Dictionary with risk metrics
        """
        if self.projections is None:
            raise ValueError("No projections generated. Call project() first.")
        
        # Calculate volatility (standard deviation of growth)
        volatility = self.projections['yhat'].pct_change().std()
        
        # Risk categories
        if volatility < 0.02:
            risk_level = "LOW"
            risk_color = "green"
        elif volatility < 0.05:
            risk_level = "MODERATE"
            risk_color = "yellow"
        else:
            risk_level = "HIGH"
            risk_color = "red"
        
        # Calculate potential revenue impact
        projected_final = self.projections['yhat'].iloc[-1]
        worst_case = self.projections['yhat_lower'].iloc[-1]
        best_case = self.projections['yhat_upper'].iloc[-1]
        
        return {
            'risk_level': risk_level,
            'risk_color': risk_color,
            'volatility': volatility,
            'projected_value': projected_final,
            'worst_case_value': worst_case,
            'best_case_value': best_case,
            'downside_risk': ((worst_case - projected_final) / projected_final) * 100,
            'upside_potential': ((best_case - projected_final) / projected_final) * 100
        }


def compare_districts(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Compare districts by total assessed value and growth potential.
    
    Args:
        df: DataFrame with property data
        top_n: Number of top districts to return
        
    Returns:
        DataFrame with district comparisons
    """
    # Group by district
    district_stats = df.groupby('district').agg({
        'APRTOT': ['sum', 'mean', 'median', 'count'],
        'APRLAND': 'sum',
        'APRBLDG': 'sum'
    }).reset_index()
    
    district_stats.columns = ['district', 'total_value', 'mean_value', 'median_value', 
                              'property_count', 'total_land_value', 'total_building_value']
    
    # Calculate percentages
    total_county_value = district_stats['total_value'].sum()
    district_stats['pct_of_county'] = (district_stats['total_value'] / total_county_value) * 100
    
    # Sort by total value
    district_stats = district_stats.sort_values('total_value', ascending=False)
    
    return district_stats.head(top_n)


def generate_scenario_analysis(base_value: float, scenarios: List[Dict]) -> pd.DataFrame:
    """
    Generate scenario analysis for budget planning.
    
    Args:
        base_value: Base assessed value
        scenarios: List of scenario dicts with 'name' and 'growth_rate'
        
    Returns:
        DataFrame with scenario projections
    """
    results = []
    
    for scenario in scenarios:
        name = scenario['name']
        rate = scenario['growth_rate']
        
        # Project 5 years
        for year in range(1, 6):
            projected = base_value * ((1 + rate) ** year)
            results.append({
                'scenario': name,
                'year': year,
                'growth_rate': rate,
                'projected_value': projected,
                'change_from_base': ((projected - base_value) / base_value) * 100
            })
    
    return pd.DataFrame(results)
