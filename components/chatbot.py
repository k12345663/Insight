"""
InsightGenie AI - Chatbot Component
Interactive chat interface for data exploration
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.agent import InsightAgent
from engine.story_mode import render_story_mode
from config.theme import COLORS


def render_chatbot(df: pd.DataFrame, schema: Optional[Dict] = None):
    """
    Render the interactive chatbot interface.
    Handles natural language commands and updates visualizations.
    """
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="background: linear-gradient(90deg, #FF6B35, #9B59B6);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            💬 AI Data Assistant
        </h2>
        <p style="color: #8B949E;">Ask questions, request filters, or modify visualizations with natural language</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat components
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    if 'filtered_df' not in st.session_state:
        st.session_state['filtered_df'] = df
    
    if 'chart_settings' not in st.session_state:
        st.session_state['chart_settings'] = {
            'color': COLORS['orange'],
            'chart_type': 'bar'
        }
    
    # Get API key from environment or sidebar
    api_key = os.environ.get('GEMINI_API_KEY')
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("**🔑 LLM Configuration**")
        user_api_key = st.text_input(
            "Gemini API Key (optional)",
            type="password",
            help="Enter your Google Gemini API key for enhanced AI capabilities"
        )
        if user_api_key:
            api_key = user_api_key
    
    # Create agent
    agent = InsightAgent(df, schema, api_key)
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for msg in st.session_state['chat_history']:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 0.5rem;">
                        <div style="background: linear-gradient(135deg, #FF6B35, #F39C12);
                                    color: white; padding: 0.75rem 1rem;
                                    border-radius: 1rem 1rem 0.25rem 1rem;
                                    max-width: 80%;">
                            {msg['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 0.5rem;">
                        <div style="background: #1E2329;
                                    color: white; padding: 0.75rem 1rem;
                                    border-radius: 1rem 1rem 1rem 0.25rem;
                                    max-width: 80%; border: 1px solid #30363D;">
                            🤖 {msg['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Ask me anything about your data...")
        
        if user_input:
            # Add user message to history
            st.session_state['chat_history'].append({
                'role': 'user',
                'content': user_input
            })
            
            # Process command
            response = agent.process_command(user_input)
            
            # Handle the response
            if response['action'] == 'filter':
                if response.get('filter_condition'):
                    try:
                        # Apply filter
                        filtered = eval(response['filter_condition'])
                        if isinstance(filtered, pd.Series):
                            st.session_state['filtered_df'] = df[filtered]
                        response['message'] += f" ({len(st.session_state['filtered_df'])} records match)"
                    except Exception as e:
                        response['message'] += f" (Filter couldn't be applied: {str(e)})"
            
            elif response['action'] == 'chart_update':
                if response.get('chart_type'):
                    st.session_state['chart_settings']['chart_type'] = response['chart_type']
            
            elif response['action'] == 'color_change':
                if response.get('color'):
                    st.session_state['chart_settings']['color'] = response['color']
            
            # Add assistant response to history
            st.session_state['chat_history'].append({
                'role': 'assistant',
                'content': response.get('message', 'I processed your request.')
            })
            
            # Add suggestions if available
            if response.get('suggestions'):
                st.session_state['chat_history'].append({
                    'role': 'assistant',
                    'content': "**Suggestions:** " + ", ".join(response['suggestions'])
                })
            
            st.rerun()
    
    with col2:
        st.markdown("### 💡 Quick Actions")
        
        # Quick action buttons
        suggestions = agent.get_suggestions()
        
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggest_{suggestion[:20]}", use_container_width=True):
                st.session_state['chat_history'].append({
                    'role': 'user',
                    'content': suggestion
                })
                response = agent.process_command(suggestion)
                st.session_state['chat_history'].append({
                    'role': 'assistant',
                    'content': response.get('message', 'Processing...')
                })
                st.rerun()
        
        st.markdown("---")
        
        # Current state
        st.markdown("### 📊 Current State")
        st.metric("Active Records", len(st.session_state['filtered_df']))
        st.metric("Chart Type", st.session_state['chart_settings']['chart_type'].title())
        
        # Color preview
        st.markdown("**Current Color:**")
        st.markdown(f"""
        <div style="background: {st.session_state['chart_settings']['color']}; 
                    width: 100%; height: 30px; border-radius: 4px;"></div>
        """, unsafe_allow_html=True)
        
        # Reset button
        if st.button("🔄 Reset Filters", use_container_width=True):
            st.session_state['filtered_df'] = df
            st.session_state['chat_history'].append({
                'role': 'assistant', 
                'content': "Filters have been reset to show all data."
            })
            st.rerun()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state['chat_history'] = []
            st.rerun()
    
    # Show filtered data preview
    st.markdown("---")
    st.markdown("### 📋 Current Data View")
    
    display_df = st.session_state['filtered_df']
    st.dataframe(display_df.head(20), use_container_width=True, height=300)
    
    # Show a dynamic chart based on settings
    st.markdown("### 📊 Dynamic Visualization")
    
    chart_type = st.session_state['chart_settings']['chart_type']
    color = st.session_state['chart_settings']['color']
    
    _render_dynamic_chart(display_df, chart_type, color, schema)
    
    # Story mode toggle
    with st.expander("📖 View Story Mode Insights"):
        render_story_mode(display_df, schema)


def _render_dynamic_chart(df: pd.DataFrame, chart_type: str, color: str, schema: Optional[Dict] = None):
    """Render a chart based on current settings."""
    import plotly.express as px
    
    mapping = schema.get('mapping', {}) if schema else {}
    
    # Find suitable columns
    categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() <= 20]
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not categorical_cols or not numeric_cols:
        st.info("📊 Select data with both categorical and numeric columns for visualization.")
        return
    
    # Default selections
    cat_col = categorical_cols[0]
    num_col = numeric_cols[0]
    
    # Try to use mapped columns
    for col, role in mapping.items():
        if role in ['party', 'state', 'category'] and col in categorical_cols:
            cat_col = col
        elif role in ['votes', 'margin', 'value'] and col in numeric_cols:
            num_col = col
    
    try:
        # Aggregate data
        agg_df = df.groupby(cat_col)[num_col].sum().reset_index().head(15)
        agg_df = agg_df.sort_values(num_col, ascending=False)
        
        if chart_type == 'bar':
            fig = px.bar(
                agg_df, x=cat_col, y=num_col,
                color_discrete_sequence=[color],
                template='plotly_dark'
            )
        elif chart_type == 'pie':
            fig = px.pie(
                agg_df, names=cat_col, values=num_col,
                color_discrete_sequence=px.colors.sequential.RdBu,
                template='plotly_dark'
            )
        elif chart_type == 'scatter':
            if len(numeric_cols) >= 2:
                fig = px.scatter(
                    df.head(500), x=numeric_cols[0], y=numeric_cols[1],
                    color=cat_col if len(df[cat_col].unique()) <= 10 else None,
                    template='plotly_dark'
                )
            else:
                fig = px.scatter(
                    agg_df, x=cat_col, y=num_col,
                    color_discrete_sequence=[color],
                    template='plotly_dark'
                )
        elif chart_type == 'line':
            fig = px.line(
                agg_df, x=cat_col, y=num_col,
                color_discrete_sequence=[color],
                template='plotly_dark',
                markers=True
            )
        else:  # Default to bar
            fig = px.bar(
                agg_df, x=cat_col, y=num_col,
                color_discrete_sequence=[color],
                template='plotly_dark'
            )
        
        fig.update_layout(
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering chart: {str(e)}")
