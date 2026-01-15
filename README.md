# 🏘️ Fairfax County Real Estate Forecast

**AI-Powered Property Value Predictions with Strategic Insights**

Version 3.0 | Python 3.13+ | Modern UI

---

## Overview

A beautiful, dual-mode application for forecasting Fairfax County real estate values:

- **📊 Visual Dashboard (60%)**: Interactive charts, metrics, and risk analysis
- **💬 AI Assistant (40%)**: Azure OpenAI-powered chatbot for executive Q&A

Built with Prophet for time series forecasting and featuring a stunning modern UI with gradients, animations, and dark mode support.

### Strategic Value for DIT Tech Management

This tool provides **actionable intelligence** for critical technology decisions:

- **💰 Budget Defense**: Predict revenue declines early, justify automation investments before cuts hit
- **🏗️ Infrastructure Planning**: Deploy fiber and smart city sensors where growth is happening
- **🏛️ Service Delivery**: Scale capacity proactively for demographic shifts and service surges

**See [USE_CASES.md](cursor/USE_CASES.md) for detailed scenarios and ROI examples.**

---

## Quick Start

### 1. Install Dependencies

```bash
pip3 install -r requirements.txt
pip3 install prophet>=1.1.5
pip3 install -e .
```

### 2. Launch Application

```bash
# Option 1: Direct launch
streamlit run apps/app.py

# Option 2: Using launcher
python3 apps/launch.py
```

The app opens at `http://localhost:8501`

### 3. Use the Application

1. Select "County Total" or a specific district
2. Click **🔄 Train** to train the model (~30 seconds)
3. Click **📊 Forecast** to generate predictions
4. Explore the dashboard and ask the AI assistant questions!

---

## Features

### Dashboard
- **4 KPI Cards**: Total value, mean value, growth rate, forecast horizon
- **Interactive Charts**: Plotly visualizations with zoom, pan, and hover details
- **Risk Assessment**: Downside risk and upside potential with color coding
- **Detailed Tables**: Expandable predictions with confidence intervals
- **District Analysis**: Drill down to specific areas (Tysons, Route 1, etc.)

### AI Assistant
- **Natural Language Q&A**: Ask strategic questions about forecasts
- **Context-Aware**: Responses based on current forecast data
- **Suggested Questions**: Quick-start prompts for common scenarios
- **Chat History**: Export conversations for reports
- **Strategic Insights**: Budget impact, infrastructure timing, capacity planning

### Design
- **Modern Gradients**: Professional purple/blue color scheme
- **Smooth Animations**: Fade-ins and hover effects
- **Dark Mode**: Automatic system detection
- **Responsive**: Works on desktop, tablet, and mobile
- **Professional Typography**: Inter and JetBrains Mono fonts
- **Executive-Ready**: Polished interface for leadership presentations

---

## Configuration

### Azure OpenAI (Optional)

Azure OpenAI credentials are configured via **environment variables** during AMP deployment:

- `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_KEY` - Your API key
- `AZURE_OPENAI_DEPLOYMENT` - Model deployment name (default: gpt-4)
- `AZURE_OPENAI_API_VERSION` - API version (default: 2024-02-15-preview)

**During deployment**, you'll be prompted for the endpoint and API key. You can leave them empty to use the mock chatbot for testing.

**See [ENVIRONMENT_CONFIG.md](cursor/ENVIRONMENT_CONFIG.md) for detailed configuration guide.**

**Note**: Without Azure OpenAI credentials, the app automatically uses a mock chatbot for testing.

### Data Requirements

Place Fairfax County CSV files in the `csvs/` directory:

- `Tax_Administration_s_Real_Estate_-_Assessed_Values.csv` (Required)
- `Tax_Administration_s_Real_Estate_-_Parcel_Data.csv` (Required)
- Other supporting files (Optional)

Download from [Fairfax County Open Data](https://data.fairfaxcounty.gov/)

---

## Project Structure

```
.
├── apps/
│   ├── app.py              # Main application
│   └── launch.py           # Launcher script
├── sts/
│   ├── data/
│   │   ├── fairfax_loader.py      # Data loading
│   │   └── external_regressors.py # Phase 2 support
│   ├── models/
│   │   └── forecast_api.py        # Forecasting API
│   └── ai/
│       └── chatbot.py             # AI assistant
├── scripts/
│   └── setup_environment.py       # Setup validation
├── csvs/                          # Your data files
├── config.example.env             # Configuration template
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
└── README.md                      # This file
```

---

## API Usage

The application exposes a Python API for programmatic access:

```python
from sts.models.forecast_api import ForecastAPI, get_forecast_summary

# Initialize API
api = ForecastAPI()

# Train a model
model = api.train_model(district_id='0804', periods_ahead=6)

# Generate forecast
forecast = api.generate_forecast(district_id='0804', periods_ahead=6)

# Get summary
summary = get_forecast_summary('0804')
print(f"Total Predicted Value: ${summary['total_predicted_value']:,.0f}")

# Access predictions
for pred in summary['predictions']:
    print(f"{pred['date']}: ${pred['predicted_value']:,.0f}")
```

---

## Troubleshooting

### "No module named 'prophet'"
```bash
pip3 install prophet>=1.1.5
```

### "No district data available"
- Ensure CSV files are in `csvs/` directory
- Check file names match exactly (case-sensitive)

### "Azure OpenAI connection failed"
- App will automatically use mock chatbot
- Check `.env` file configuration
- Verify Azure OpenAI credentials

### Python version issues
```bash
# Check version (should be 3.13+)
python3 --version

# Upgrade if needed (macOS)
brew install python@3.13
```

### Setup validation
```bash
# Run validation script
python3 scripts/setup_environment.py
```

---

## Phase 2 Features (Coming Soon)

The application is designed to support external regressors:

- **CPI Data**: Consumer Price Index integration
- **Inflation Rates**: Economic indicators
- **Interest Rates**: Federal funds, mortgage rates
- **Unemployment**: County-level data
- **Custom Indicators**: Extensible framework

Implementation ready in `sts/data/external_regressors.py`

---

## Technical Details

### Requirements

- **Python**: 3.13 or higher
- **Key Packages**: 
  - Prophet 1.1.5+ (forecasting)
  - Streamlit 1.30+ (web framework)
  - Plotly 5.18+ (visualizations)
  - OpenAI 1.10+ (AI assistant)
  - Pandas 2.2+ (data processing)

### Performance

- **App Load**: ~2-3 seconds
- **Model Training**: 30-60 seconds per district
- **Forecast Generation**: 5-10 seconds
- **Chart Rendering**: <1 second

### Python 3.13 Benefits

- **15-20% faster** execution
- **10% better** memory usage
- **Enhanced** error messages
- **Better** type hints and IDE support

---

## Strategic Use Cases

### Real-World Scenarios for DIT Tech Management

**See [USE_CASES.md](USE_CASES.md) for detailed examples:**

1. **Budget Defense Plan** 💰
   - Predict 4% commercial value decline in Tysons
   - Justify cloud migration before budget cuts
   - Approve automation to survive hiring freeze

2. **Digital Infrastructure Plan** 🏗️
   - Forecast population growth in Route 1 Corridor
   - Deploy fiber before density makes it expensive
   - Install smart city sensors at optimal timing

3. **Service Delivery Plan** 🏛️
   - Identify gentrification hotspots early
   - Scale Human Services portal capacity
   - Prepare for displaced resident service surges

**Each scenario includes:**
- The signal from the tool
- Strategic response actions
- Measurable ROI and cost savings
- Dashboard queries and AI questions to ask

---

## Credits

### Technologies

- **Python 3.13**: Python Software Foundation
- **Prophet**: Facebook Research
- **Streamlit**: Snowflake Inc.
- **Plotly**: Plotly Technologies
- **Azure OpenAI**: Microsoft

### Design

- Modern gradient design inspired by Tailwind CSS and Vercel
- Professional typography using Inter and JetBrains Mono
- Responsive design following best practices
