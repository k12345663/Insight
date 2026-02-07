"""
InsightGenie AI - Heatmap Component
Competitiveness heatmap with state-wise intensity visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.theme import COLORS, PLOTLY_TEMPLATE


def render_heatmap(df: pd.DataFrame, schema: Optional[Dict] = None):
    """
    Render a competitiveness heatmap based on available data.
    Auto-detects categorical and numeric columns for the heatmap.
    """
    
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h4 style="color: #8B949E; font-weight: 400;">
            🔥 Regional intensity map showing competitiveness across categories
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    mapping = schema.get('mapping', {}) if schema else {}
    
    # Find suitable columns
    categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() <= 30]
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # UI controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Default to state column if detected
        default_x = None
        for col, role in mapping.items():
            if role in ['state', 'category']:
                default_x = col
                break
        
        x_col = st.selectbox(
            "X-Axis (Categories)",
            categorical_cols,
            index=categorical_cols.index(default_x) if default_x and default_x in categorical_cols else 0,
            key="heatmap_x"
        ) if categorical_cols else None
    
    with col2:
        # Secondary category for heatmap
        y_options = [c for c in categorical_cols if c != x_col]
        y_col = st.selectbox(
            "Y-Axis (Groups)",
            y_options if y_options else ["None"],
            key="heatmap_y"
        ) if y_options else None
    
    with col3:
        # Value column
        default_val = None
        for col, role in mapping.items():
            if role in ['margin', 'votes', 'percentage', 'value']:
                default_val = col
                break
        
        val_col = st.selectbox(
            "Value (Intensity)",
            numeric_cols,
            index=numeric_cols.index(default_val) if default_val and default_val in numeric_cols else 0,
            key="heatmap_val"
        ) if numeric_cols else None
    
    if not x_col or not val_col:
        st.warning("⚠️ Not enough columns detected for heatmap visualization.")
        return
    
    # Generate heatmap
    try:
        if y_col and y_col != "None":
            # Create pivot table for 2D heatmap
            pivot_df = df.pivot_table(
                values=val_col,
                index=y_col,
                columns=x_col,
                aggfunc='mean'
            ).fillna(0)
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_df.values,
                x=pivot_df.columns.tolist(),
                y=pivot_df.index.tolist(),
                colorscale=[
                    [0, COLORS["background"]],
                    [0.25, "#16213E"],
                    [0.5, "#0F3460"],
                    [0.75, "#E94560"],
                    [1, COLORS["orange"]]
                ],
                hovertemplate="<b>%{x}</b><br>%{y}<br>Value: %{z:.2f}<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"{val_col.replace('_', ' ').title()} by {x_col.title()} and {y_col.title()}",
                **PLOTLY_TEMPLATE["layout"],
                height=500
            )
        else:
            # Single category bar-style heatmap
            agg_df = df.groupby(x_col)[val_col].mean().reset_index()
            agg_df = agg_df.sort_values(val_col, ascending=False).head(20)
            
            fig = px.bar(
                agg_df,
                x=x_col,
                y=val_col,
                color=val_col,
                color_continuous_scale=[
                    COLORS["background"],
                    "#0F3460",
                    "#E94560",
                    COLORS["orange"]
                ],
                template="plotly_dark"
            )
            
            fig.update_layout(
                title=f"Average {val_col.replace('_', ' ').title()} by {x_col.title()}",
                **PLOTLY_TEMPLATE["layout"],
                height=450,
                showlegend=False
            )
            
            fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights below the chart
        with st.expander("📊 Heatmap Insights"):
            if y_col and y_col != "None":
                st.markdown(f"""
                **Analysis:**
                - Showing average **{val_col}** across **{x_col}** and **{y_col}**
                - Darker/warmer colors indicate higher intensity
                - Identify regional hotspots for targeted analysis
                """)
            else:
                top_3 = agg_df.head(3)[x_col].tolist()
                st.markdown(f"""
                **Top 3 by {val_col}:** {', '.join(str(x) for x in top_3)}
                """)
        
    except Exception as e:
        st.error(f"❌ Error generating heatmap: {str(e)}")
        st.info("Try selecting different columns for the visualization.")
