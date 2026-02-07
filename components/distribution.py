"""
InsightGenie AI - Distribution Analysis Component
Violin/Density plots for margin distribution analysis
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


def render_distribution(df: pd.DataFrame, schema: Optional[Dict] = None):
    """
    Render distribution analysis with violin/density plots.
    Highlights thresholds for razor-thin vs landslide victories.
    """
    
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h4 style="color: #8B949E; font-weight: 400;">
            📈 Understand the spread and concentration of your key metrics
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    mapping = schema.get('mapping', {}) if schema else {}
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() <= 15]
    
    if not numeric_cols:
        st.warning("⚠️ No numeric columns found for distribution analysis.")
        return
    
    # UI controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Default to margin if available
        default_val = None
        for col, role in mapping.items():
            if role in ['margin', 'percentage']:
                default_val = col
                break
        
        val_col = st.selectbox(
            "Value Column",
            numeric_cols,
            index=numeric_cols.index(default_val) if default_val and default_val in numeric_cols else 0,
            key="dist_val"
        )
    
    with col2:
        # Group by category
        group_col = st.selectbox(
            "Group By (Optional)",
            ["None"] + categorical_cols,
            key="dist_group"
        )
    
    with col3:
        # Chart type
        chart_type = st.selectbox(
            "Chart Type",
            ["Violin", "Histogram", "Box", "Density"],
            key="dist_type"
        )
    
    try:
        if chart_type == "Violin":
            if group_col != "None":
                fig = px.violin(
                    df,
                    y=val_col,
                    x=group_col,
                    color=group_col,
                    box=True,
                    points="outliers",
                    color_discrete_sequence=COLORS["chart_gradient"],
                    template="plotly_dark"
                )
            else:
                fig = px.violin(
                    df,
                    y=val_col,
                    box=True,
                    points="outliers",
                    color_discrete_sequence=[COLORS["orange"]],
                    template="plotly_dark"
                )
        
        elif chart_type == "Histogram":
            if group_col != "None":
                fig = px.histogram(
                    df,
                    x=val_col,
                    color=group_col,
                    marginal="box",
                    color_discrete_sequence=COLORS["chart_gradient"],
                    template="plotly_dark",
                    barmode="overlay",
                    opacity=0.7
                )
            else:
                fig = px.histogram(
                    df,
                    x=val_col,
                    marginal="box",
                    color_discrete_sequence=[COLORS["orange"]],
                    template="plotly_dark"
                )
        
        elif chart_type == "Box":
            if group_col != "None":
                fig = px.box(
                    df,
                    y=val_col,
                    x=group_col,
                    color=group_col,
                    color_discrete_sequence=COLORS["chart_gradient"],
                    template="plotly_dark"
                )
            else:
                fig = px.box(
                    df,
                    y=val_col,
                    color_discrete_sequence=[COLORS["orange"]],
                    template="plotly_dark"
                )
        
        else:  # Density
            fig = go.Figure()
            
            if group_col != "None":
                for i, group in enumerate(df[group_col].unique()[:8]):
                    subset = df[df[group_col] == group][val_col].dropna()
                    fig.add_trace(go.Violin(
                        y=subset,
                        name=str(group),
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=COLORS["chart_gradient"][i % len(COLORS["chart_gradient"])],
                        opacity=0.6,
                        line_color=COLORS["text_primary"]
                    ))
            else:
                fig.add_trace(go.Violin(
                    y=df[val_col].dropna(),
                    name=val_col,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=COLORS["orange"],
                    opacity=0.6,
                    line_color=COLORS["text_primary"]
                ))
        
        # Add threshold lines if it looks like percentage data
        if df[val_col].max() <= 100 or 'percent' in val_col.lower() or 'margin' in val_col.lower():
            # Razor's edge threshold (< 2%)
            fig.add_hline(
                y=2,
                line_dash="dot",
                line_color="#E74C3C",
                annotation_text="Razor's Edge (<2%)",
                annotation_position="right"
            )
            
            # Landslide threshold (> 20%)
            fig.add_hline(
                y=20,
                line_dash="dot",
                line_color="#2ECC71",
                annotation_text="Landslide (>20%)",
                annotation_position="right"
            )
        
        fig.update_layout(
            title=f"Distribution of {val_col.replace('_', ' ').title()}",
            **PLOTLY_TEMPLATE["layout"],
            height=500,
            showlegend=group_col != "None"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution statistics
        with st.expander("📊 Distribution Statistics"):
            stats_cols = st.columns(5)
            
            data = df[val_col].dropna()
            
            with stats_cols[0]:
                st.metric("Mean", f"{data.mean():.2f}")
            with stats_cols[1]:
                st.metric("Median", f"{data.median():.2f}")
            with stats_cols[2]:
                st.metric("Std Dev", f"{data.std():.2f}")
            with stats_cols[3]:
                st.metric("Min", f"{data.min():.2f}")
            with stats_cols[4]:
                st.metric("Max", f"{data.max():.2f}")
            
            # Threshold analysis if percentage-like
            if df[val_col].max() <= 100 or 'percent' in val_col.lower() or 'margin' in val_col.lower():
                st.markdown("---")
                threshold_cols = st.columns(3)
                
                razor_count = len(df[df[val_col] < 2])
                moderate_count = len(df[(df[val_col] >= 2) & (df[val_col] < 20)])
                landslide_count = len(df[df[val_col] >= 20])
                
                with threshold_cols[0]:
                    st.metric("🔪 Razor's Edge (<2%)", razor_count, 
                              f"{(razor_count/len(df)*100):.1f}%")
                with threshold_cols[1]:
                    st.metric("⚔️ Moderate (2-20%)", moderate_count,
                              f"{(moderate_count/len(df)*100):.1f}%")
                with threshold_cols[2]:
                    st.metric("🏆 Landslide (>20%)", landslide_count,
                              f"{(landslide_count/len(df)*100):.1f}%")
        
    except Exception as e:
        st.error(f"❌ Error generating distribution plot: {str(e)}")
