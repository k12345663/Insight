"""
InsightGenie AI - KPI Cards Component
Dynamic metric cards with auto-calculation and styling
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.theme import COLORS, KPI_STYLE


def render_kpi_cards(df: pd.DataFrame, schema: Optional[Dict] = None):
    """
    Render dynamic KPI cards based on the dataset.
    Auto-detects important metrics and displays them beautifully.
    """
    
    # Inject KPI styling
    st.markdown(KPI_STYLE, unsafe_allow_html=True)
    
    # Calculate KPIs based on available data
    kpis = calculate_kpis(df, schema)
    
    # Display in columns
    cols = st.columns(len(kpis))
    
    for i, kpi in enumerate(kpis):
        with cols[i]:
            render_kpi_card(
                title=kpi['title'],
                value=kpi['value'],
                delta=kpi.get('delta'),
                delta_type=kpi.get('delta_type', 'normal'),
                icon=kpi.get('icon', '📊')
            )


def calculate_kpis(df: pd.DataFrame, schema: Optional[Dict] = None) -> List[Dict]:
    """
    Calculate KPIs based on the data and schema mapping.
    Returns list of KPI configurations.
    """
    kpis = []
    mapping = schema.get('mapping', {}) if schema else {}
    
    # Total records
    kpis.append({
        'title': 'Total Records',
        'value': format_number(len(df)),
        'icon': '📋'
    })
    
    # Find votes column
    votes_col = None
    for col, role in mapping.items():
        if role in ['votes', 'value']:
            votes_col = col
            break
    
    # If no mapping, try to find numeric columns
    if not votes_col:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            # Find the one with "vote" in name, or use largest values
            for col in numeric_cols:
                if 'vote' in col.lower():
                    votes_col = col
                    break
            if not votes_col:
                votes_col = numeric_cols[0]
    
    if votes_col:
        total_votes = df[votes_col].sum()
        kpis.append({
            'title': f'Total {votes_col.replace("_", " ").title()}',
            'value': format_number(total_votes),
            'icon': '🗳️'
        })
    
    # Find turnout/percentage column
    turnout_col = None
    for col, role in mapping.items():
        if role in ['turnout', 'percentage']:
            turnout_col = col
            break
    
    if not turnout_col:
        for col in df.columns:
            if 'turnout' in col.lower() or 'percent' in col.lower():
                turnout_col = col
                break
    
    if turnout_col and pd.api.types.is_numeric_dtype(df[turnout_col]):
        avg_turnout = df[turnout_col].mean()
        kpis.append({
            'title': f'Avg {turnout_col.replace("_", " ").title()}',
            'value': f'{avg_turnout:.1f}%',
            'icon': '📈'
        })
    
    # Find margin column
    margin_col = None
    for col, role in mapping.items():
        if role == 'margin':
            margin_col = col
            break
    
    if not margin_col:
        for col in df.columns:
            if 'margin' in col.lower():
                margin_col = col
                break
    
    if margin_col and pd.api.types.is_numeric_dtype(df[margin_col]):
        avg_margin = df[margin_col].mean()
        # If it looks like a percentage
        if avg_margin <= 100:
            kpis.append({
                'title': 'Avg Margin',
                'value': f'{avg_margin:.2f}%',
                'icon': '📊'
            })
        else:
            kpis.append({
                'title': 'Avg Margin',
                'value': format_number(avg_margin),
                'icon': '📊'
            })
    
    # Find party/category column for unique count
    party_col = None
    for col, role in mapping.items():
        if role in ['party', 'category']:
            party_col = col
            break
    
    if party_col:
        unique_parties = df[party_col].nunique()
        kpis.append({
            'title': 'Unique Parties',
            'value': str(unique_parties),
            'icon': '🏛️'
        })
    
    # Find state/region column
    state_col = None
    for col, role in mapping.items():
        if role in ['state', 'constituency']:
            state_col = col
            break
    
    if state_col:
        unique_regions = df[state_col].nunique()
        kpis.append({
            'title': f'Total {state_col.replace("_", " ").title()}s',
            'value': str(unique_regions),
            'icon': '🗺️'
        })
    
    # Limit to 4-5 KPIs for clean display
    return kpis[:5]


def render_kpi_card(title: str, value: str, delta: Optional[str] = None, 
                    delta_type: str = 'normal', icon: str = '📊'):
    """Render a single styled KPI card."""
    
    delta_html = ""
    if delta:
        delta_class = 'positive' if delta_type == 'positive' else 'negative' if delta_type == 'negative' else ''
        delta_symbol = '↑' if delta_type == 'positive' else '↓' if delta_type == 'negative' else ''
        delta_html = f'<div class="kpi-delta {delta_class}">{delta_symbol} {delta}</div>'
    
    st.markdown(f"""
    <div class="kpi-card">
        <div style="font-size: 1.5rem;">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{title}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def format_number(num: float) -> str:
    """Format large numbers with appropriate suffixes."""
    if num >= 1_00_00_000:  # Crore
        return f'{num / 1_00_00_000:.2f} Cr'
    elif num >= 1_00_000:  # Lakh
        return f'{num / 1_00_000:.2f} L'
    elif num >= 1_000:
        return f'{num / 1_000:.1f}K'
    else:
        return f'{num:,.0f}'


def calculate_custom_kpi(df: pd.DataFrame, column: str, aggregation: str = 'sum') -> float:
    """Calculate a custom KPI for a specific column."""
    if column not in df.columns:
        return 0
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df[column].nunique() if aggregation == 'count' else 0
    
    if aggregation == 'sum':
        return df[column].sum()
    elif aggregation == 'mean':
        return df[column].mean()
    elif aggregation == 'median':
        return df[column].median()
    elif aggregation == 'min':
        return df[column].min()
    elif aggregation == 'max':
        return df[column].max()
    elif aggregation == 'count':
        return len(df)
    else:
        return df[column].sum()
