#!/usr/bin/env python3
# ###########################################################################
#
#  Basic Usage Examples
#  Demonstrates how to use the Fairfax Real Estate Forecast API
#
# ###########################################################################

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sts.models.projection_api import ProjectionAPI, get_forecast_summary
from sts.data.fairfax_loader import get_available_districts, get_district_summary
from sts.ai.chatbot import create_chatbot

def example_1_list_districts():
    """Example 1: List available districts."""
    print("\n" + "="*70)
    print("Example 1: List Available Districts")
    print("="*70 + "\n")
    
    # Get all districts
    districts = get_available_districts()
    print(f"Found {len(districts)} districts")
    print(f"Sample districts: {districts[:10]}")
    
    # Get district summary with statistics
    summary = get_district_summary()
    print("\nTop 5 districts by total value:")
    print(summary.head(5)[['district', 'total_value', 'property_count']])

def example_2_train_and_forecast():
    """Example 2: Train a model and generate forecast."""
    print("\n" + "="*70)
    print("Example 2: Train Model and Generate Forecast")
    print("="*70 + "\n")
    
    # Initialize API
    api = ProjectionAPI()
    
    # Get a district to work with
    districts = get_available_districts()
    if len(districts) == 0:
        print("No districts available. Please check CSV files.")
        return
    
    test_district = districts[0]
    print(f"Using district: {test_district}")
    
    # Train model
    print("\nTraining model...")
    try:
        model = api.train_model(
            district_id=test_district,
            periods_ahead=6,
            growth_rate=0.03
        )
        print("✓ Model trained successfully")
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return
    
    # Generate forecast
    print("\nGenerating forecast...")
    try:
        forecast = api.generate_forecast(
            district_id=test_district,
            periods_ahead=6
        )
        print("✓ Forecast generated successfully")
        print(f"\nForecast shape: {forecast.shape}")
        print(f"Columns: {forecast.columns.tolist()}")
    except Exception as e:
        print(f"✗ Error generating forecast: {e}")
        return

def example_3_get_forecast_summary():
    """Example 3: Get forecast summary."""
    print("\n" + "="*70)
    print("Example 3: Get Forecast Summary")
    print("="*70 + "\n")
    
    # Get districts
    districts = get_available_districts()
    if len(districts) == 0:
        print("No districts available.")
        return
    
    test_district = districts[0]
    
    try:
        # Get forecast summary
        summary = get_forecast_summary(test_district, periods_ahead=6)
        
        print(f"District: {summary['district']}")
        print(f"Forecast Date: {summary['forecast_date']}")
        print(f"Periods Ahead: {summary['periods_ahead']}")
        print(f"\nTotal Predicted Value: ${summary['total_predicted_value']:,.0f}")
        print(f"Mean Predicted Value: ${summary['mean_predicted_value']:,.0f}")
        
        print("\nDetailed Predictions:")
        for pred in summary['predictions']:
            print(f"  {pred['date']}: ${pred['predicted_value']:,.0f} "
                  f"[${pred['lower_bound']:,.0f} - ${pred['upper_bound']:,.0f}]")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've trained a model first (run example_2)")

def example_4_use_chatbot():
    """Example 4: Use the AI chatbot."""
    print("\n" + "="*70)
    print("Example 4: AI Chatbot Interaction")
    print("="*70 + "\n")
    
    # Create chatbot (will use mock if Azure OpenAI not configured)
    chatbot = create_chatbot(use_mock=True)
    
    # Ask some questions
    questions = [
        "What is the forecast for property values?",
        "What are the main revenue risks?",
        "How confident are these predictions?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        response = chatbot.chat(question)
        print(f"A: {response}")

def example_5_train_multiple_districts():
    """Example 5: Train models for multiple districts."""
    print("\n" + "="*70)
    print("Example 5: Train Multiple District Models")
    print("="*70 + "\n")
    
    api = ProjectionAPI()
    
    # Train models for top 3 districts
    print("Training models for top 3 districts...")
    try:
        models = api.train_all_districts(top_n=3, periods_ahead=6)
        print(f"\n✓ Successfully trained {len(models)} models")
        print(f"Districts: {list(models.keys())}")
    except Exception as e:
        print(f"✗ Error: {e}")

def example_6_county_level_forecast():
    """Example 6: County-level (aggregate) forecast."""
    print("\n" + "="*70)
    print("Example 6: County-Level Forecast")
    print("="*70 + "\n")
    
    api = ProjectionAPI()
    
    # Train county-level model (district_id=None)
    print("Training county-level model...")
    try:
        model = api.train_model(district_id=None, periods_ahead=6)
        print("✓ Model trained")
        
        # Generate forecast
        forecast = api.generate_forecast(district_id=None, periods_ahead=6)
        print("✓ Forecast generated")
        
        # Get summary
        summary = api.get_forecast_summary(district_id=None, periods_ahead=6)
        print(f"\nCounty-wide total predicted value: ${summary['total_predicted_value']:,.0f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  Fairfax County Real Estate Forecast - API Examples")
    print("="*70)
    
    examples = [
        ("List Districts", example_1_list_districts),
        ("Train and Forecast", example_2_train_and_forecast),
        ("Get Forecast Summary", example_3_get_forecast_summary),
        ("Use Chatbot", example_4_use_chatbot),
        ("Train Multiple Districts", example_5_train_multiple_districts),
        ("County-Level Forecast", example_6_county_level_forecast)
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n✗ Example '{name}' failed: {e}")
    
    print("\n" + "="*70)
    print("  Examples Complete!")
    print("="*70 + "\n")
    
    print("For more information, see:")
    print("  - README.md: Full documentation")
    print("  - QUICKSTART.md: Quick start guide")
    print("  - apps/fairfax_forecast.py: Full application")

if __name__ == '__main__':
    main()
