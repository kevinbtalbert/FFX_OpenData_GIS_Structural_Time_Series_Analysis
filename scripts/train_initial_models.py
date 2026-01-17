#!/usr/bin/env python3
"""
Train Initial Models Script
Trains projection models for top districts.
"""

import sys
from sts.models.projection_api import ProjectionAPI
from sts.data.fairfax_loader import get_district_summary

def main():
    print("Training initial forecasting models...")
    print("=" * 60)
    
    try:
        # Initialize API
        api = ProjectionAPI()
        
        # Get top districts
        summary = get_district_summary()
        top_districts = summary.head(5)['district'].tolist()
        
        print(f"Training models for top 5 districts: {top_districts}")
        print()
        
        # Train models
        for i, district_id in enumerate(top_districts, 1):
            print(f"[{i}/5] Training model for district {district_id}...")
            try:
                model = api.train_model(
                    district_id=district_id,
                    periods_ahead=6,
                    growth_rate=0.03
                )
                
                # Generate initial forecast
                forecast = api.generate_forecast(
                    district_id=district_id,
                    periods_ahead=6
                )
                
                print(f"     District {district_id} complete")
            except Exception as e:
                print(f"     District {district_id} failed: {e}")
        
        # Train county-level model
        print()
        print("Training county-level model...")
        try:
            model = api.train_model(district_id=None, periods_ahead=6)
            forecast = api.generate_forecast(district_id=None, periods_ahead=6)
            print("     County-level model complete")
        except Exception as e:
            print(f"     County-level model failed: {e}")
        
        print()
        print("=" * 60)
        print("Initial model training complete!")
        print("Models saved to: data/models/")
        print("Forecasts saved to: data/forecasts/")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during model training: {e}")
        print("Application will still launch but models need to be trained manually.")
        sys.exit(0)  # Don't fail deployment, just warn

if __name__ == '__main__':
    main()
