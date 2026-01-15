# 🚀 Quick Start Guide

## Install & Run (2 Minutes)

```bash
# 1. Install dependencies
pip3 install -r requirements.txt
pip3 install prophet>=1.1.5
pip3 install -e .

# 2. Launch application
streamlit run apps/app.py
```

## First Time Use

1. **Select Location**: Choose "County Total" or a specific district (Tysons, Route 1, etc.)
2. **Train Model**: Click 🔄 Train button (~30 seconds)
3. **Generate Forecast**: Click 📊 Forecast button
4. **Explore**: View dashboard and ask AI assistant strategic questions!

## Strategic Use Cases 🎯

**For DIT Tech Management - See [USE_CASES.md](cursor/USE_CASES.md):**

- **💰 Budget Defense**: Predict revenue declines, justify automation investments
- **🏗️ Infrastructure Planning**: Deploy fiber where growth is happening
- **🏛️ Service Delivery**: Scale capacity for demographic shifts

**Example Questions to Ask:**
- "What's the commercial outlook for Tysons Corner?"
- "Where should we prioritize infrastructure investment?"
- "Which areas will need increased service capacity?"

## Configure Azure OpenAI (Optional)

Azure OpenAI is configured via environment variables during AMP deployment. The application will automatically use mock chatbot if credentials are not provided.

To enable real AI assistant, set these during deployment:
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`

