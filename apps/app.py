# ###########################################################################
#
#  Fairfax County Real Estate Forecast - Enhanced UI
#  Python 3.13+ | Modern Design | Dark Mode | Advanced Visualizations
#
# ###########################################################################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sts.models.projection_api import ProjectionAPI, get_forecast_summary
from sts.data.fairfax_loader import get_available_districts, get_district_summary
from sts.ai.chatbot import create_chatbot

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Fairfax County Real Estate Forecast",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/fairfaxcounty/real-estate-forecast',
        'Report a bug': 'https://github.com/fairfaxcounty/real-estate-forecast/issues',
        'About': '# Fairfax County Real Estate Forecast\n\nAI-powered property value forecasting with strategic insights.'
    }
)

# ==============================================================================
# CUSTOM CSS - MODERN, VISUALLY APPEALING DESIGN
# ==============================================================================

st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }
    
    /* Ensure all text is readable */
    p, h1, h2, h3, h4, h5, h6, span, div, label {
        color: #1e293b !important;
    }
    
    /* Streamlit specific overrides */
    .stMarkdown, .stText {
        color: #1e293b !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        color: white;
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.95;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid rgba(255, 255, 255, 0.1);
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        color: white !important;
    }
    
    .metric-card * {
        color: white !important;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff !important;
        margin: 0.5rem 0;
        font-family: 'JetBrains Mono', monospace;
        word-break: break-word;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #e2e8f0 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        line-height: 1.3;
    }

    .metric-delta {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.5rem;
        color: #ffffff !important;
    }

    .metric-delta.positive {
        color: #10b981 !important;
    }

    .metric-delta.negative {
        color: #ef4444 !important;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Chat interface */
    .chat-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        max-height: 500px;
        min-height: 500px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        max-width: 85%;
        animation: fadeIn 0.3s ease-in;
        word-wrap: break-word;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        margin-left: auto;
        text-align: right;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }

    .user-message strong {
        color: white !important;
    }
    
    .user-message p {
        color: white !important;
    }

    .assistant-message {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: #1e293b !important;
        margin-right: auto;
        border: 1px solid #cbd5e1;
    }

    .assistant-message strong {
        color: #1e293b !important;
    }

    .chat-message strong {
        display: block;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        font-weight: 700;
    }

    .chat-message p {
        margin: 0;
        line-height: 1.5;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .info-box * {
        color: #1e293b !important;
    }

    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .warning-box * {
        color: #1e293b !important;
    }

    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .success-box * {
        color: #1e293b !important;
    }
    
    /* Suggested questions */
    .suggested-question {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.95rem;
    }
    
    .suggested-question:hover {
        border-color: #667eea;
        background: #f8fafc;
        transform: translateX(4px);
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main-header {
            background: linear-gradient(135deg, #4c1d95 0%, #5b21b6 100%);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: #f1f5f9;
        }
        
        .metric-value {
            color: #a78bfa;
        }
        
        .metric-label {
            color: #94a3b8;
        }
        
        .section-header {
            color: #f1f5f9;
            border-bottom-color: #a78bfa;
        }
        
        .chat-container {
            background: #1e293b;
            border-color: #334155;
        }
        
        .assistant-message {
            background: linear-gradient(135deg, #334155 0%, #475569 100%);
            color: #f1f5f9;
            border-color: #475569;
        }
    }
    
    /* Loading animation */
    .loading-spinner {
        border: 4px solid #f3f4f6;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
        
        .chat-message {
            max-width: 95%;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

# Initialize chatbot - ALWAYS recreate to ensure fresh state
try:
    # Get credentials from environment
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', 'https://ktalbert.openai.azure.com/')
    azure_key = os.getenv('AZURE_OPENAI_API_KEY')
    
    # Ensure endpoint ends with /
    if azure_endpoint and not azure_endpoint.endswith('/'):
        azure_endpoint = azure_endpoint + '/'
    
    if azure_endpoint and azure_key:
        # Use Azure OpenAI SDK
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
        )
        
        # Simple wrapper matching Azure's sample code
        class DirectChatbot:
            def __init__(self, client):
                self.client = client
                self.conversation_history = []
            
            def chat(self, user_message, forecast_data=None, context=None, **kwargs):
                try:
                    messages = [
                        {"role": "system", "content": "You are an AI assistant for Fairfax County real estate forecasting. Provide concise, executive-focused insights about property values, trends, and revenue risks."}
                    ]
                    
                    if forecast_data:
                        context_msg = f"Current forecast data: Total value ${forecast_data.get('base_value', 0):,.0f}, Growth {forecast_data.get('total_growth_pct', 0):.2f}%"
                        messages.append({"role": "system", "content": context_msg})
                    
                    messages.append({"role": "user", "content": user_message})
                    
                    # Use EXACT parameters from Azure sample
                    completion = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=800
                    )
                    
                    return completion.choices[0].message.content
                    
                except Exception as e:
                    return f"Error: {str(e)}"
            
            def reset_conversation(self):
                self.conversation_history = []
            
            def get_suggested_questions(self, forecast_data=None):
                return [
                    "What is the predicted property value for the next 6 months?",
                    "Which districts show the highest growth potential?",
                    "What is the revenue risk if values decline?",
                    "How confident are these predictions?",
                    "What factors could impact these forecasts?"
                ]
        
        st.session_state.chatbot = DirectChatbot(client)
        st.session_state.using_mock = False
    else:
        # Use mock
        st.warning("DEBUG: No credentials, using mock chatbot")
        st.session_state.chatbot = create_chatbot(use_mock=True)
        st.session_state.using_mock = True
except Exception as e:
    st.error(f"DEBUG: Error creating chatbot: {e}")
    import traceback
    st.error(traceback.format_exc())
    st.session_state.chatbot = create_chatbot(use_mock=True)
    st.session_state.using_mock = True

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'forecast_api' not in st.session_state:
    st.session_state.forecast_api = ProjectionAPI()

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# ==============================================================================
# HEADER
# ==============================================================================

st.markdown("""
<div class="main-header">
    <h1>🏘️ Fairfax County Real Estate Forecast</h1>
    <p>AI-Powered Property Value Predictions & Strategic Revenue Insights</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================

with st.sidebar:
    # Logo/Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">🏘️ Fairfax County</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Real Estate Forecast</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Configuration")
    
    # District selection
    st.markdown("#### 📍 Location")
    
    @st.cache_data
    def load_districts():
        try:
            return get_available_districts()
        except:
            return []
    
    @st.cache_data
    def load_district_summary():
        try:
            return get_district_summary()
        except:
            return pd.DataFrame()
    
    districts = load_districts()
    district_summary = load_district_summary()
    
    if len(districts) > 0:
        view_level = st.radio(
            "Analysis Level",
            ["🏙️ County Total", "📍 Specific District"],
            help="Choose between county-wide or district-level analysis"
        )
        
        if view_level == "📍 Specific District":
            if not district_summary.empty:
                st.markdown("**Top Districts by Value:**")
                top_districts = district_summary.head(15)['district'].tolist()
                
                selected_district = st.selectbox(
                    "Select District",
                    options=top_districts,
                    format_func=lambda x: f"District {x}",
                    help="Select a district to analyze"
                )
            else:
                selected_district = st.selectbox(
                    "Select District",
                    options=districts[:20],
                    format_func=lambda x: f"District {x}"
                )
        else:
            selected_district = None
    else:
        st.warning("⚠️ No district data available")
        selected_district = None
    
    st.markdown("---")
    
    # Forecast parameters
    st.markdown("#### 📊 Forecast Settings")
    
    periods_ahead = st.slider(
        "Forecast Periods",
        min_value=3,
        max_value=24,
        value=6,
        help="Number of months to forecast ahead"
    )
    
    confidence_level = st.select_slider(
        "Confidence Level",
        options=[80, 90, 95, 99],
        value=95,
        format_func=lambda x: f"{x}%"
    )
    
    st.markdown("---")
    
    # Model management
    st.markdown("#### 🤖 Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Train", width='stretch', help="Train new model", key="train_btn"):
            with st.spinner("Training model... This may take 30-60 seconds."):
                try:
                    st.session_state.forecast_api.train_model(
                        district_id=selected_district,
                        periods_ahead=periods_ahead,
                        growth_rate=0.03
                    )
                    st.success("✓ Model trained successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col2:
        if st.button("📊 Forecast", width='stretch', help="Generate forecast", key="forecast_btn"):
            with st.spinner("Generating forecast..."):
                try:
                    st.session_state.forecast_api.generate_forecast(
                        district_id=selected_district,
                        periods_ahead=periods_ahead
                    )
                    st.success("✓ Forecast generated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Forecast failed: {str(e)}")
                    st.info("💡 Tip: Make sure to train the model first using the 🔄 Train button.")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # Info
    st.markdown("#### ℹ️ About")
    st.markdown("""
    <div style='font-size: 0.85rem; color: #64748b;'>
    <strong>Version:</strong> 3.0.0<br>
    <strong>Python:</strong> 3.13+<br>
    <strong>Model:</strong> Growth Projections<br>
    <strong>AI:</strong> Azure OpenAI
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# MAIN CONTENT - DASHBOARD (60%) & CHATBOT (40%)
# ==============================================================================

# Create two columns with 60/40 split
col_dashboard, col_chat = st.columns([6, 4], gap="large")

# ==============================================================================
# DASHBOARD SECTION (60%)
# ==============================================================================

with col_dashboard:
    st.markdown('<div class="section-header">📈 Forecast Dashboard</div>', unsafe_allow_html=True)
    
    try:
        # Load forecast data
        forecast = st.session_state.forecast_api.get_forecast(district_id=selected_district)
        forecast_summary = st.session_state.forecast_api.get_forecast_summary(
            district_id=selected_district,
            periods_ahead=periods_ahead
        )
        
        # KEY METRICS ROW
        st.markdown("#### Key Performance Indicators")
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            total_value = forecast_summary['total_predicted_value']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Predicted Value</div>
                <div class="metric-value">${total_value/1e9:.2f}B</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[1]:
            mean_value = forecast_summary['mean_predicted_value']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Value/Period</div>
                <div class="metric-value">${mean_value/1e6:.1f}M</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[2]:
            if len(forecast_summary['predictions']) > 1:
                first_val = forecast_summary['predictions'][0]['predicted_value']
                last_val = forecast_summary['predictions'][-1]['predicted_value']
                growth_rate = ((last_val - first_val) / first_val) * 100
                delta_class = "positive" if growth_rate > 0 else "negative"
                arrow = "↑" if growth_rate > 0 else "↓"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Projected Growth</div>
                    <div class="metric-value">{growth_rate:.1f}%</div>
                    <div class="metric-delta {delta_class}">{arrow} {abs(growth_rate):.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        with metric_cols[3]:
            num_periods = len(forecast_summary['predictions'])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Forecast Horizon</div>
                <div class="metric-value">{num_periods}</div>
                <div style="font-size: 0.9rem; color: #e2e8f0; margin-top: 0.5rem;">months ahead</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # MAIN FORECAST CHART
        st.markdown("#### Interactive Forecast Visualization")
        
        forecast_plot = forecast.copy()
        forecast_plot['ds'] = pd.to_datetime(forecast_plot['ds'])
        
        # Create advanced Plotly chart
        fig = go.Figure()
        
        # Add current value as starting point
        base_date = datetime(forecast_summary['base_year'], 1, 1)
        base_value = forecast_summary['base_value']
        
        # Current value point
        fig.add_trace(go.Scatter(
            x=[base_date],
            y=[base_value],
            mode='markers',
            name='Current Value (2025)',
            marker=dict(size=12, color='#667eea', symbol='circle'),
            hovertemplate='<b>Current Value</b><br><b>Date:</b> Jan 2025<br><b>Value:</b> $%{y:,.0f}<extra></extra>'
        ))
        
        # Connecting line from current to first projection
        fig.add_trace(go.Scatter(
            x=[base_date, forecast_plot['ds'].iloc[0]],
            y=[base_value, forecast_plot['yhat'].iloc[0]],
            mode='lines',
            name='Transition',
            line=dict(color='#94a3b8', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Projected values
        fig.add_trace(go.Scatter(
            x=forecast_plot['ds'],
            y=forecast_plot['yhat'],
            mode='lines+markers',
            name='Projected Growth',
            line=dict(color='#10b981', width=3),
            marker=dict(size=8, color='#10b981'),
            hovertemplate='<b>Projection</b><br><b>Date:</b> %{x|%b %Y}<br><b>Value:</b> $%{y:,.0f}<extra></extra>'
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_plot['ds'],
            y=forecast_plot['yhat_upper'],
            mode='lines',
            name=f'Upper Bound ({confidence_level}%)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_plot['ds'],
            y=forecast_plot['yhat_lower'],
            mode='lines',
            name=f'Confidence Interval ({confidence_level}%)',
            line=dict(width=0),
            fillcolor='rgba(16, 185, 129, 0.15)',
            fill='tonexty',
            hovertemplate='<b>Range:</b> $%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"Property Value Forecast - {selected_district if selected_district else 'County Total'}",
                font=dict(size=20, family='Inter', weight=700)
            ),
            xaxis_title="Date",
            yaxis_title="Assessed Value ($)",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(family='Inter')
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # AI EXPLANATION SECTION
        st.markdown("#### 🤖 AI Analysis")
        with st.spinner("Generating AI explanation..."):
            try:
                # Generate AI explanation of the forecast
                explanation_prompt = f"""Provide a brief executive summary (2-3 sentences) of this forecast:
                
- Current Value: ${forecast_summary['base_value']:,.0f}
- Projected Value (6 months): ${forecast_summary['final_predicted_value']:,.0f}
- Growth: {forecast_summary['total_growth_pct']:.2f}%
- Properties: {forecast_summary['property_count']:,}
- District: {selected_district if selected_district else 'County Total'}

Focus on: What does this mean for revenue planning? Any concerns or opportunities?"""
                
                ai_explanation = st.session_state.chatbot.chat(
                    explanation_prompt,
                    forecast_data=forecast_summary
                )
                
                # Strip HTML tags from AI explanation
                import re
                from html.parser import HTMLParser
                
                class MLStripper(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.reset()
                        self.strict = False
                        self.convert_charrefs= True
                        self.text = []
                    def handle_data(self, d):
                        self.text.append(d)
                    def get_data(self):
                        return ''.join(self.text)
                
                def strip_tags(html):
                    s = MLStripper()
                    s.feed(html)
                    return s.get_data()
                
                ai_explanation = strip_tags(ai_explanation)
                
                st.markdown(f"""
                <div class="info-box">
                    <p style="margin: 0; line-height: 1.6;">{ai_explanation}</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.info("💡 **Quick Summary:** " + 
                       f"The forecast projects {'growth' if forecast_summary['total_growth_pct'] > 0 else 'decline'} " +
                       f"of {abs(forecast_summary['total_growth_pct']):.1f}% over the next 6 months, " +
                       f"from ${forecast_summary['base_value']/1e9:.2f}B to ${forecast_summary['final_predicted_value']/1e9:.2f}B. " +
                       f"This represents a {'positive' if forecast_summary['total_growth_pct'] > 0 else 'negative'} trend for revenue planning.")
        
        # RISK ASSESSMENT SECTION
        st.markdown("#### Revenue Risk Assessment")
        
        risk_cols = st.columns(2)
        
        with risk_cols[0]:
            lower_total = sum([p['lower_bound'] for p in forecast_summary['predictions']])
            expected_total = forecast_summary['total_predicted_value']
            downside_risk = expected_total - lower_total
            downside_pct = (downside_risk/expected_total)*100
            
            st.markdown(f"""
            <div class="warning-box">
                <h4 style="margin: 0 0 0.5rem 0;">⚠️ Downside Risk</h4>
                <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">
                    ${downside_risk/1e6:.1f}M
                </div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                    Potential revenue loss at {confidence_level}% confidence (-{downside_pct:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with risk_cols[1]:
            upper_total = sum([p['upper_bound'] for p in forecast_summary['predictions']])
            upside_potential = upper_total - expected_total
            upside_pct = (upside_potential/expected_total)*100
            
            st.markdown(f"""
            <div class="success-box">
                <h4 style="margin: 0 0 0.5rem 0;">📈 Upside Potential</h4>
                <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">
                    ${upside_potential/1e6:.1f}M
                </div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                    Potential revenue gain at {confidence_level}% confidence (+{upside_pct:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # DETAILED PREDICTIONS TABLE
        with st.expander("📋 Detailed Predictions Table", expanded=False):
            predictions_df = pd.DataFrame(forecast_summary['predictions'])
            predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.strftime('%Y-%m-%d')
            predictions_df['predicted_value'] = predictions_df['predicted_value'].apply(lambda x: f"${x:,.0f}")
            predictions_df['lower_bound'] = predictions_df['lower_bound'].apply(lambda x: f"${x:,.0f}")
            predictions_df['upper_bound'] = predictions_df['upper_bound'].apply(lambda x: f"${x:,.0f}")
            predictions_df['confidence_interval'] = predictions_df['confidence_interval'].apply(lambda x: f"{x*100:.0f}%")
            
            st.dataframe(
                predictions_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "date": "Date",
                    "predicted_value": "Predicted Value",
                    "lower_bound": "Lower Bound",
                    "upper_bound": "Upper Bound",
                    "confidence_interval": "Confidence"
                }
            )
        
    except Exception as e:
        st.markdown("""
        <div class="info-box">
            <h4 style="margin: 0 0 0.5rem 0;">ℹ️ Getting Started</h4>
            <p style="margin: 0;">No forecast data available yet. Follow these steps:</p>
            <ol style="margin: 0.5rem 0 0 1rem;">
                <li>Select a district or view county total</li>
                <li>Click <strong>🔄 Train</strong> to train the model</li>
                <li>Click <strong>📊 Forecast</strong> to generate predictions</li>
                <li>Explore the dashboard and ask the AI assistant questions!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# AI ASSISTANT SECTION (40%)
# ==============================================================================

with col_chat:
    st.markdown('<div class="section-header">💬 AI Assistant</div>', unsafe_allow_html=True)
    
    if st.session_state.using_mock:
        st.markdown("""
        <div class="info-box" style="font-size: 0.85rem;">
            ℹ️ Using mock chatbot. Configure Azure OpenAI for full AI capabilities.
        </div>
        """, unsafe_allow_html=True)
    
    # Chat container with scrollable history - build as single HTML block
    import re
    from html.parser import HTMLParser
    
    class MLStripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.reset()
            self.strict = False
            self.convert_charrefs= True
            self.text = []
        def handle_data(self, d):
            self.text.append(d)
        def get_data(self):
            return ''.join(self.text)
    
    def strip_html_tags(html):
        """Completely remove all HTML tags and return only text content"""
        s = MLStripper()
        s.feed(html)
        return s.get_data()
    
    chat_html = '<div class="chat-container">'
    
    # Display chat history - strip ALL HTML tags from content
    for message in st.session_state.chat_history:
        # Completely strip all HTML tags
        clean_content = strip_html_tags(message['content'])
        
        if message['role'] == 'user':
            chat_html += f"""
            <div class="chat-message user-message">
                <strong style="color: white !important;">👤 You</strong><br>
                <span style="color: white !important;">{clean_content}</span>
            </div>
            """
        else:
            chat_html += f"""
            <div class="chat-message assistant-message">
                <strong>🤖 AI Assistant</strong><br>
                <span>{clean_content}</span>
            </div>
            """
    
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)
    
    # Suggested questions
    with st.expander("💡 Suggested Questions", expanded=True):
        suggested = st.session_state.chatbot.get_suggested_questions()
        for i, question in enumerate(suggested[:5]):
            if st.button(question, key=f"suggested_{i}", width='stretch'):
                st.session_state.pending_question = question
                st.rerun()
    
    # User input with form for Enter key support
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question:",
            placeholder="e.g., What's the revenue outlook for next quarter?",
            key="user_input_field",
            label_visibility="collapsed"
        )
        
        col_send, col_clear = st.columns([3, 1])
        
        with col_send:
            send_button = st.form_submit_button("📤 Send", type="primary", use_container_width=True)
        
        with col_clear:
            clear_button = st.form_submit_button("🗑️ Clear", use_container_width=True)
    
    # Handle clear button
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.chatbot.reset_conversation()
        st.rerun()
    
    # Handle pending question from suggested questions
    if hasattr(st.session_state, 'pending_question'):
        user_input = st.session_state.pending_question
        del st.session_state.pending_question
        send_button = True
    
    if send_button and user_input:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Get forecast context
        try:
            forecast_summary = st.session_state.forecast_api.get_forecast_summary(
                district_id=selected_district,
                periods_ahead=periods_ahead
            )
            context = {
                'district': selected_district if selected_district else 'County Total',
                'date_range': f"Next {periods_ahead} months"
            }
        except:
            forecast_summary = None
            context = None
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.chatbot.chat(
                    user_input,
                    forecast_data=forecast_summary,
                    context=context
                )
                
                # Strip any HTML tags from the response
                import re
                from html.parser import HTMLParser
                
                class MLStripper(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.reset()
                        self.strict = False
                        self.convert_charrefs= True
                        self.text = []
                    def handle_data(self, d):
                        self.text.append(d)
                    def get_data(self):
                        return ''.join(self.text)
                
                def strip_tags(html):
                    s = MLStripper()
                    s.feed(html)
                    return s.get_data()
                
                # Clean the response
                response = strip_tags(response)
                
            except Exception as e:
                response = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
        
        # Add assistant message (now cleaned of HTML)
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response
        })
        
        st.rerun()

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.9rem; padding: 1rem 0;'>
    <strong>Fairfax County Real Estate Forecast</strong> | Version 3.0.0 | Python 3.13+ | 
    Powered by Growth Projections & Azure OpenAI | 
    <a href='https://github.com/fairfaxcounty/real-estate-forecast' style='color: #667eea;'>GitHub</a>
</div>
""", unsafe_allow_html=True)
