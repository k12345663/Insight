"""
InsightGenie AI - Quadrant Analysis Component
Scatter plot mapping two metrics to identify Swing vs Safe zones
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.theme import COLORS, PLOTLY_TEMPLATE


def render_quadrant(df: pd.DataFrame, schema: Optional[Dict] = None):
    """
    Render quadrant analysis scatter plot.
    Auto-calculates medians to create 4 zones.
    """
    
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h4 style="color: #8B949E; font-weight: 400;">
            🎯 Identify patterns by mapping two key metrics against each other
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    mapping = schema.get('mapping', {}) if schema else {}
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and 3 <= df[c].nunique() <= 30]
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for quadrant analysis.")
        return
    
    # UI controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Default to turnout if available
        default_x = None
        for col, role in mapping.items():
            if role in ['turnout', 'percentage']:
                default_x = col
                break
        
        x_col = st.selectbox(
            "X-Axis",
            numeric_cols,
            index=numeric_cols.index(default_x) if default_x and default_x in numeric_cols else 0,
            key="quad_x"
        )
    
    with col2:
        # Default to margin
        default_y = None
        for col, role in mapping.items():
            if role == 'margin':
                default_y = col
                break
        
        remaining_cols = [c for c in numeric_cols if c != x_col]
        y_col = st.selectbox(
            "Y-Axis",
            remaining_cols,
            index=remaining_cols.index(default_y) if default_y and default_y in remaining_cols else 0,
            key="quad_y"
        ) if remaining_cols else None
    
    with col3:
        # Color by category
        color_col = st.selectbox(
            "Color By",
            ["None"] + categorical_cols,
            key="quad_color"
        )
    
    if not x_col or not y_col:
        st.warning("⚠️ Please select both X and Y axes.")
        return
    
    # Calculate medians for quadrant lines
    x_median = df[x_col].median()
    y_median = df[y_col].median()
    
    try:
        # Create scatter plot
        if color_col != "None":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                color_discrete_sequence=COLORS["chart_gradient"],
                hover_data=df.columns[:5].tolist(),
                template="plotly_dark"
            )
        else:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color_discrete_sequence=[COLORS["orange"]],
                hover_data=df.columns[:5].tolist(),
                template="plotly_dark"
            )
        
        # Add quadrant lines
        fig.add_hline(
            y=y_median,
            line_dash="dash",
            line_color=COLORS["text_secondary"],
            annotation_text=f"Median: {y_median:.2f}",
            annotation_position="right"
        )
        
        fig.add_vline(
            x=x_median,
            line_dash="dash",
            line_color=COLORS["text_secondary"],
            annotation_text=f"Median: {x_median:.2f}",
            annotation_position="top"
        )
        
        # Add quadrant labels
        x_range = df[x_col].max() - df[x_col].min()
        y_range = df[y_col].max() - df[y_col].min()
        
        quadrant_labels = [
            # Top-right: High X, High Y
            {"x": x_median + x_range * 0.25, "y": y_median + y_range * 0.35, "text": "🔥 High Engagement"},
            # Top-left: Low X, High Y
            {"x": x_median - x_range * 0.25, "y": y_median + y_range * 0.35, "text": "⚡ Swing Zone"},
            # Bottom-right: High X, Low Y
            {"x": x_median + x_range * 0.25, "y": y_median - y_range * 0.35, "text": "🏆 Safe Zone"},
            # Bottom-left: Low X, Low Y
            {"x": x_median - x_range * 0.25, "y": y_median - y_range * 0.35, "text": "😴 Low Priority"},
        ]
        
        for label in quadrant_labels:
            fig.add_annotation(
                x=label["x"],
                y=label["y"],
                text=label["text"],
                showarrow=False,
                font=dict(size=12, color=COLORS["text_muted"]),
                opacity=0.7
            )
        
        fig.update_layout(
            title=f"Quadrant Analysis: {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}",
            **PLOTLY_TEMPLATE["layout"],
            height=550
        )
        
        fig.update_traces(marker=dict(size=10, opacity=0.7))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quadrant statistics
        with st.expander("📊 Quadrant Statistics"):
            # Calculate counts per quadrant
            high_x_high_y = len(df[(df[x_col] >= x_median) & (df[y_col] >= y_median)])
            low_x_high_y = len(df[(df[x_col] < x_median) & (df[y_col] >= y_median)])
            high_x_low_y = len(df[(df[x_col] >= x_median) & (df[y_col] < y_median)])
            low_x_low_y = len(df[(df[x_col] < x_median) & (df[y_col] < y_median)])
            
            stat_cols = st.columns(4)
            with stat_cols[0]:
                st.metric("🔥 High Engagement", high_x_high_y)
            with stat_cols[1]:
                st.metric("⚡ Swing Zone", low_x_high_y)
            with stat_cols[2]:
                st.metric("🏆 Safe Zone", high_x_low_y)
            with stat_cols[3]:
                st.metric("😴 Low Priority", low_x_low_y)
            
            st.markdown(f"""
            **Interpretation:**
            - **High Engagement**: High {x_col}, high {y_col} - competitive areas with strong participation
            - **Swing Zone**: Lower {x_col}, high {y_col} - potential for change with engagement
            - **Safe Zone**: High {x_col}, lower {y_col} - stable with good participation
            - **Low Priority**: Lower in both metrics - requires attention
            """)
        
    except Exception as e:
        st.error(f"❌ Error generating quadrant plot: {str(e)}")
