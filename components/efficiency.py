"""
InsightGenie AI - Efficiency Analysis Component
Vote Share vs Seat Share comparison with Winner's Bonus calculation
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


def render_efficiency(df: pd.DataFrame, schema: Optional[Dict] = None):
    """
    Render efficiency analysis comparing input vs output metrics.
    Calculates Winner's Bonus and conversion efficiency.
    """
    
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h4 style="color: #8B949E; font-weight: 400;">
            ⚡ Compare input metrics against outcomes to identify efficiency patterns
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    mapping = schema.get('mapping', {}) if schema else {}
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() <= 20]
    
    if not categorical_cols or not numeric_cols:
        st.warning("⚠️ Need categorical and numeric columns for efficiency analysis.")
        return
    
    # UI controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Group by (party, state, etc.)
        default_group = None
        for col, role in mapping.items():
            if role in ['party', 'category']:
                default_group = col
                break
        
        group_col = st.selectbox(
            "Group By",
            categorical_cols,
            index=categorical_cols.index(default_group) if default_group and default_group in categorical_cols else 0,
            key="eff_group"
        )
    
    with col2:
        # Primary metric (e.g., votes)
        default_metric = None
        for col, role in mapping.items():
            if role in ['votes', 'value']:
                default_metric = col
                break
        
        metric_col = st.selectbox(
            "Primary Metric",
            numeric_cols,
            index=numeric_cols.index(default_metric) if default_metric and default_metric in numeric_cols else 0,
            key="eff_metric"
        )
    
    with col3:
        # Chart type
        chart_mode = st.selectbox(
            "Analysis Type",
            ["Vote Share vs Count", "Efficiency Comparison", "Strike Rate"],
            key="eff_mode"
        )
    
    try:
        if chart_mode == "Vote Share vs Count":
            # Grouped bar showing total metric vs count per group
            agg_df = df.groupby(group_col).agg({
                metric_col: ['sum', 'count', 'mean']
            }).reset_index()
            agg_df.columns = [group_col, 'Total', 'Count', 'Average']
            
            # Calculate shares
            agg_df['Metric Share (%)'] = (agg_df['Total'] / agg_df['Total'].sum()) * 100
            agg_df['Count Share (%)'] = (agg_df['Count'] / agg_df['Count'].sum()) * 100
            agg_df['Winner\'s Bonus'] = agg_df['Count Share (%)'] - agg_df['Metric Share (%)']
            
            # Sort by total
            agg_df = agg_df.sort_values('Total', ascending=False).head(15)
            
            # Create grouped bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Metric Share (%)',
                x=agg_df[group_col],
                y=agg_df['Metric Share (%)'],
                marker_color=COLORS["blue"],
                text=agg_df['Metric Share (%)'].round(1),
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                name='Count Share (%)',
                x=agg_df[group_col],
                y=agg_df['Count Share (%)'],
                marker_color=COLORS["orange"],
                text=agg_df['Count Share (%)'].round(1),
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"{metric_col.replace('_', ' ').title()} Share vs Record Count Share by {group_col.title()}",
                barmode='group',
                **PLOTLY_TEMPLATE["layout"],
                height=500
            )
            
        elif chart_mode == "Efficiency Comparison":
            # Calculate efficiency metrics per group
            agg_df = df.groupby(group_col).agg({
                metric_col: ['sum', 'count']
            }).reset_index()
            agg_df.columns = [group_col, 'Total', 'Count']
            
            # Efficiency = Average per record
            agg_df['Efficiency'] = agg_df['Total'] / agg_df['Count']
            agg_df['Share (%)'] = (agg_df['Total'] / agg_df['Total'].sum()) * 100
            
            # Sort by efficiency
            agg_df = agg_df.sort_values('Efficiency', ascending=True)
            
            fig = px.bar(
                agg_df.tail(15),
                y=group_col,
                x='Efficiency',
                color='Share (%)',
                color_continuous_scale=[COLORS["background"], COLORS["orange"]],
                orientation='h',
                template="plotly_dark",
                text='Efficiency'
            )
            
            fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            
            fig.update_layout(
                title=f"{metric_col.replace('_', ' ').title()} Efficiency by {group_col.title()}",
                **PLOTLY_TEMPLATE["layout"],
                height=500
            )
            
        else:  # Strike Rate
            # Strike rate = success count / total count
            agg_df = df.groupby(group_col).size().reset_index(name='Total Records')
            
            # If we have a "winner" or success indicator, use it
            winner_col = None
            for col, role in mapping.items():
                if role in ['winner', 'party']:
                    winner_col = col
                    break
            
            if winner_col and winner_col != group_col:
                # Count wins per group
                wins = df[df[winner_col] == df[group_col]].groupby(group_col).size().reset_index(name='Wins')
                agg_df = agg_df.merge(wins, on=group_col, how='left').fillna(0)
                agg_df['Strike Rate (%)'] = (agg_df['Wins'] / agg_df['Total Records']) * 100
            else:
                # Use count as proxy
                agg_df['Strike Rate (%)'] = (agg_df['Total Records'] / agg_df['Total Records'].sum()) * 100
            
            agg_df = agg_df.sort_values('Strike Rate (%)', ascending=False).head(15)
            
            fig = px.bar(
                agg_df,
                x=group_col,
                y='Strike Rate (%)',
                color='Strike Rate (%)',
                color_continuous_scale=[COLORS["background"], COLORS["green"], COLORS["orange"]],
                template="plotly_dark",
                text='Strike Rate (%)'
            )
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            
            fig.update_layout(
                title=f"Strike Rate by {group_col.title()}",
                **PLOTLY_TEMPLATE["layout"],
                height=500,
                showlegend=False
            )
            
            fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency insights
        with st.expander("📊 Efficiency Analysis"):
            if chart_mode == "Vote Share vs Count":
                st.markdown("**Winner's Bonus Analysis:**")
                st.markdown("""
                The difference between Count Share and Metric Share reveals:
                - **Positive bonus**: Getting more seats/records than metric share would suggest
                - **Negative bonus**: Underperforming relative to metric share
                """)
                
                # Show top performers
                top_bonus = agg_df.nlargest(3, 'Winner\'s Bonus')
                st.dataframe(
                    top_bonus[[group_col, 'Metric Share (%)', 'Count Share (%)', 'Winner\'s Bonus']].round(2),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                # General stats
                st.markdown("**Key Metrics:**")
                total_records = len(df)
                unique_groups = df[group_col].nunique()
                st.markdown(f"- Total Records: **{total_records:,}**")
                st.markdown(f"- Unique {group_col}s: **{unique_groups}**")
                st.markdown(f"- Average per group: **{total_records/unique_groups:.1f}**")
        
    except Exception as e:
        st.error(f"❌ Error generating efficiency analysis: {str(e)}")
