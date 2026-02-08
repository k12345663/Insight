"""
InsightGenie AI - Premium Analytics Dashboard
Beautiful ML-powered visualizations with chart explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Deep Insights Engine
try:
    from engine.deep_insights import generate_deep_insights, format_parameter_insights, generate_parameter_report
    DEEP_INSIGHTS_AVAILABLE = True
except ImportError:
    DEEP_INSIGHTS_AVAILABLE = False
    def generate_deep_insights(*args, **kwargs): return []
    def format_parameter_insights(*args, **kwargs): return []
    def generate_parameter_report(*args, **kwargs): return {}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['GEMINI_API_KEY'] = 'AIzaSyD_EKoU7PvqlUClpmwPYNenaaS6rXymhRI'

# Premium Colors
COLORS = {'primary': '#6366f1', 'secondary': '#8b5cf6', 'accent': '#ec4899',
          'success': '#10b981', 'warning': '#f59e0b', 'danger': '#ef4444', 'info': '#0ea5e9'}
PALETTE = ['#6366f1', '#8b5cf6', '#ec4899', '#f43f5e', '#f97316', '#eab308', '#22c55e', '#14b8a6']

DARK = {'bg': '#0a0a0a', 'surface': '#141414', 'card': '#1f1f1f', 'border': '#2a2a2a', 'text': '#fff', 'muted': '#a0a0a0'}
LIGHT = {'bg': '#f8fafc', 'surface': '#fff', 'card': '#fff', 'border': '#e2e8f0', 'text': '#0f172a', 'muted': '#64748b'}


def get_layout(is_dark=True):
    t = DARK if is_dark else LIGHT
    return {
        'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': t['text'], 'family': 'Inter, sans-serif', 'size': 12},
        'hoverlabel': {'bgcolor': t['surface'], 'font': {'color': t['text']}},
        'legend': {'bgcolor': 'rgba(0,0,0,0)', 'font': {'size': 11}},
        'margin': {'l': 60, 'r': 30, 't': 50, 'b': 50}
    }


def chart_explanation(points, is_dark=True):
    """Display explanation points below a chart."""
    t = DARK if is_dark else LIGHT
    html = f'<div style="background:{t["card"]}; border-radius:8px; padding:0.75rem 1rem; margin-top:0.5rem; border-left:3px solid {COLORS["info"]};">'
    html += '<div style="font-size:0.75rem; font-weight:600; color:' + COLORS["info"] + '; margin-bottom:0.5rem;">💡 INSIGHTS</div>'
    for p in points:
        html += f'<div style="font-size:0.8rem; color:{t["muted"]}; margin-bottom:0.25rem;">• {p}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def get_css(is_dark=True):
    t = DARK if is_dark else LIGHT
    text_color = t['text']
    muted_color = t['muted']
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background: {t['bg']}; color: {text_color}; }}
        section[data-testid="stSidebar"] {{ background: {t['surface']}; border-right: 1px solid {t['border']}; }}
        section[data-testid="stSidebar"] * {{ color: {text_color}; }}
        h1, h2, h3, h4, h5, h6 {{ color: {text_color} !important; font-weight: 600 !important; }}
        h1 {{ font-size: 1.75rem !important; }} h2 {{ font-size: 1.375rem !important; }}
        p, span, label, div {{ color: {text_color}; }}
        .stMarkdown {{ color: {text_color}; }}
        .stMetric {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 12px; padding: 1rem; }}
        [data-testid="stMetricValue"] {{ color: {COLORS['primary']} !important; font-weight: 700 !important; }}
        [data-testid="stMetricLabel"] {{ color: {muted_color} !important; font-size: 0.75rem !important; text-transform: uppercase; }}
        .stButton > button {{ background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']}); color: white !important; border: none; border-radius: 8px; font-weight: 500; }}
        .stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(99,102,241,0.4); }}
        .stTabs [data-baseweb="tab-list"] {{ background: {t['surface']}; border-radius: 10px; padding: 4px; }}
        .stTabs [data-baseweb="tab"] {{ color: {text_color} !important; }}
        .stTabs [aria-selected="true"] {{ background: {COLORS['primary']} !important; color: white !important; }}
        .insight-card {{ background: {t['card']}; border-radius: 12px; padding: 1.25rem; border-left: 4px solid; margin-bottom: 1rem; color: {text_color}; }}
        .stSelectbox label, .stMultiSelect label, .stSlider label {{ color: {text_color} !important; }}
        .stSelectbox > div > div {{ background: {t['surface']}; color: {text_color}; border-color: {t['border']}; }}
        .stTextInput > div > div > input {{ background: {t['surface']}; color: {text_color}; border-color: {t['border']}; }}
        .stDataFrame {{ color: {text_color}; }}
        [data-testid="stExpanderToggleIcon"] {{ color: {text_color}; }}
        .stExpander {{ background: {t['card']}; border: 1px solid {t['border']}; }}
        .stCaption {{ color: {muted_color} !important; }}
        .stChatMessage {{ background: {t['card']}; border: 1px solid {t['border']}; }}
        #MainMenu, footer, header {{ visibility: hidden; }}
    </style>
    """


def init():
    for k, v in {'data': None, 'messages': [], 'theme': 'dark', 'filename': None}.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.set_page_config(page_title="InsightGenie AI", page_icon="Insight.png", layout="wide")
    init()
    
    is_dark = True
    st.markdown(get_css(is_dark), unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("Insight.png", width=150)
        st.markdown("<h3 style='font-family: \"Inter\", sans-serif; font-weight: 700; font-size: 1.5rem; margin-top: -10px;'>InsightGenie AI</h3>", unsafe_allow_html=True)
        st.markdown("---")
        
        if st.session_state['data'] is not None:
            view = st.radio("", ["🎯 Insights", "📊 Dashboard", "🔬 Analysis", "🤖 Chat", "📋 Data"], label_visibility="collapsed")
            st.markdown("---")
            st.caption(f"📁 {st.session_state.get('filename', 'Data')}")
            st.caption(f"📋 {len(st.session_state['data']):,} rows")
            if st.button("🔄 New Data", use_container_width=True):
                st.session_state['data'] = None
                st.rerun()
        else:
            view = "Upload"
    
    if view == "Upload": render_upload()
    elif view == "🎯 Insights": render_insights(st.session_state['data'], is_dark)
    elif view == "📊 Dashboard": render_dashboard(st.session_state['data'], is_dark)
    elif view == "🔬 Analysis": render_analysis(st.session_state['data'], is_dark)
    elif view == "🤖 Chat": render_chat(st.session_state['data'])
    elif view == "📋 Data": render_data(st.session_state['data'])


def clean_data(df):
    """Clean and preprocess the dataframe."""
    start_rows = len(df)
    df = df.drop_duplicates()
    dupes = start_rows - len(df)
    
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    converted = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
                converted += 1
            except:
                pass
                
    cat_cols = df.select_dtypes(include=['object']).columns
    filled = df[cat_cols].isnull().sum().sum()
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    
    report = {'duplicates': dupes, 'converted': converted, 'filled': filled, 'rows': len(df)}
    return df, report


def render_upload():
    st.markdown("## 🚀 Get Started")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
        if uploaded:
            try:
                df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
                df, report = clean_data(df)
                st.session_state['data'] = df
                st.session_state['cleaning_report'] = report
                st.session_state['filename'] = uploaded.name
                st.toast(f"✅ Cleaned: -{report['duplicates']} dupes, fixed {report['converted']} types")
                st.rerun()
            except Exception as e:
                st.error(str(e))
        
        if st.button("🎲 Load Sample Data", use_container_width=True):
            np.random.seed(42)
            df = pd.DataFrame({
                'Region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 200),
                'Category': np.random.choice(['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E'], 200),
                'Revenue': np.random.randint(50000, 500000, 200),
                'Margin': np.round(np.random.uniform(5, 40, 200), 1),
                'Growth': np.round(np.random.uniform(-10, 50, 200), 1),
                'Score': np.round(np.random.uniform(60, 98, 200), 0)
            })
            df, report = clean_data(df)
            st.session_state['data'] = df
            st.session_state['cleaning_report'] = report
            st.session_state['filename'] = "sample_data_cleaned.csv"
            st.toast("✅ Sample data generated & cleaned!")
            st.rerun()


def render_insights(df, is_dark):
    layout = get_layout(is_dark)
    
    st.markdown("## 🎯 Auto-Generated Insights")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() <= 20]
    
    # KPIs
    cols = st.columns(5)
    with cols[0]: st.metric("Records", f"{len(df):,}")
    with cols[1]: st.metric("Features", len(df.columns))
    with cols[2]: st.metric("Quality", f"{100 - df.isnull().sum().sum()/df.size*100:.0f}%")
    if num_cols:
        with cols[3]: st.metric(f"Avg {num_cols[0][:8]}", f"{df[num_cols[0]].mean():,.0f}")
    if cat_cols:
        with cols[4]: st.metric(f"Unique {cat_cols[0][:8]}", df[cat_cols[0]].nunique())
    
    st.markdown("---")
    
    # Insights
    st.markdown("### 💡 Key Discoveries")
    insights = []
    insights.append({'icon': '📊', 'title': 'Dataset', 'text': f"{len(df):,} records, {len(num_cols)} numeric, {len(cat_cols)} categorical", 'color': COLORS['primary']})
    
    if cat_cols and num_cols:
        top = df.groupby(cat_cols[0])[num_cols[0]].sum().idxmax()
        val = df.groupby(cat_cols[0])[num_cols[0]].sum().max()
        pct = val / df[num_cols[0]].sum() * 100
        insights.append({'icon': '🏆', 'title': f'Top {cat_cols[0]}', 'text': f"{top} leads with {val:,.0f} ({pct:.1f}% share)", 'color': COLORS['success']})
    
    if len(num_cols) >= 2:
        corr = df[num_cols[0]].corr(df[num_cols[1]])
        if abs(corr) > 0.3:
            insights.append({'icon': '🔗', 'title': 'Correlation', 'text': f"{num_cols[0]} & {num_cols[1]} correlated (r={corr:.2f})", 'color': COLORS['info']})
    
    for i in insights:
        st.markdown(f"""<div class="insight-card" style="border-color: {i['color']};">
            <div style="font-weight: 600;">{i['icon']} {i['title']}</div>
            <div style="opacity: 0.8; font-size: 0.9rem;">{i['text']}</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Key Visualizations")
    
    c1, c2 = st.columns(2)
    
    with c1:
        if cat_cols and num_cols:
            agg = df.groupby(cat_cols[0])[num_cols[0]].sum().nlargest(6).reset_index()
            fig = go.Figure(go.Bar(
                x=agg[num_cols[0]], y=agg[cat_cols[0]], orientation='h',
                marker=dict(color=agg[num_cols[0]], colorscale=[[0, '#6366f1'], [1, '#ec4899']])
            ))
            fig.update_layout(height=380, **layout)
            fig.update_yaxes(categoryorder='total ascending')
            fig.update_layout(title=dict(text=f'📊 Top {cat_cols[0]}'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Deep Insights
            if DEEP_INSIGHTS_AVAILABLE:
                insights = generate_deep_insights(df, 'bar', category=cat_cols[0], value=num_cols[0])
                chart_explanation(insights, is_dark)
            else:
                top_val = agg[num_cols[0]].max()
                top_name = agg[agg[num_cols[0]] == top_val][cat_cols[0]].values[0]
                diff = ((agg[num_cols[0]].max() - agg[num_cols[0]].min()) / agg[num_cols[0]].min() * 100)
                chart_explanation([
                    f"Horizontal bar ranking showing {cat_cols[0]} by total {num_cols[0]}",
                    f"**{top_name}** leads with {top_val:,.0f} - {diff:.0f}% higher than lowest performer",
                    f"Color gradient (purple→pink) indicates relative performance strength"
                ], is_dark)
    
    with c2:
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale=[[0, '#4c1d95'], [0.5, '#f8fafc'], [1, '#6366f1']], zmin=-1, zmax=1
            ))
            fig.update_layout(height=380, **layout)
            fig.update_layout(title=dict(text='🔗 Correlations'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Deep Correlation Insights
            if DEEP_INSIGHTS_AVAILABLE:
                insights = generate_deep_insights(df, 'correlation', columns=num_cols)
                chart_explanation(insights, is_dark)
            else:
                mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
                max_corr = corr.where(mask).abs().max().max()
                chart_explanation([
                    f"Correlation matrix showing relationships between {len(num_cols)} numeric variables",
                    f"Blue = positive correlation, White = no correlation, Purple = negative",
                    f"Strongest relationship: r={max_corr:.2f} - use for predictive analysis"
                ], is_dark)
    
    # === PARAMETER-WISE ANALYSIS SECTION ===
    st.markdown("---")
    with st.expander("📋 **Parameter-wise Details** (click to expand)", expanded=False):
        st.markdown("### Column-by-Column Analysis")
        
        if DEEP_INSIGHTS_AVAILABLE:
            report = generate_parameter_report(df)
            
            # Select columns to view
            selected_cols = st.multiselect(
                "Select columns to analyze:",
                options=df.columns.tolist(),
                default=df.columns[:5].tolist() if len(df.columns) > 5 else df.columns.tolist()
            )
            
            if selected_cols:
                for col in selected_cols:
                    if col in report:
                        data = report[col]
                        col_display = data.get('display_name', col)
                        
                        with st.container():
                            st.markdown(f"#### {col_display}")
                            
                            if data.get('type') == 'numeric':
                                # Numeric column stats
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("Min", data.get('min', 'N/A'))
                                c2.metric("Max", data.get('max', 'N/A'))
                                c3.metric("Mean", data.get('mean', 'N/A'))
                                c4.metric("Median", data.get('median', 'N/A'))
                                
                                # Percentiles
                                pcts = data.get('percentiles', {})
                                if pcts:
                                    st.caption(f"Percentiles: 25th={pcts.get('25%')} | 50th={pcts.get('50%')} | 75th={pcts.get('75%')} | 90th={pcts.get('90%')}")
                            else:
                                # Categorical column stats
                                top_vals = data.get('top_values', [])
                                if top_vals:
                                    top_str = ', '.join([f"{v['value']} ({v['percentage']}%)" for v in top_vals[:5]])
                                    st.markdown(f"**Top values**: {top_str}")
                                st.caption(f"Unique values: {data.get('unique_values', 'N/A')} | Missing: {data.get('null_count', 0)} ({data.get('null_pct', 0)}%)")
                            
                            # Column insights
                            for insight in data.get('insights', []):
                                st.info(insight)
                            
                            st.markdown("---")
        else:
            st.info("Parameter-wise analysis requires the deep insights module.")

def render_dashboard(df, is_dark):
    layout = get_layout(is_dark)
    
    st.markdown("## 📊 Analytics Dashboard")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() <= 20]
    
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Distribution", "Comparison", "Heatmap"])
    
    with tab1:
        # KPIs - Smart formatting based on column type
        cols = st.columns(4)
        for i, col in enumerate(num_cols[:4]):
            with cols[i]:
                val = df[col].sum()
                col_lower = col.lower()
                # Don't use K for year-like columns or small values
                is_year_like = ('year' in col_lower or 'date' in col_lower or 
                               'id' in col_lower or (1900 <= val/len(df) <= 2100))
                is_aggregatable = any(x in col_lower for x in ['revenue', 'sales', 'amount', 'price', 'cost', 'profit', 'total'])
                
                if is_year_like:
                    # Show average for years
                    avg_val = df[col].mean()
                    st.metric(col, f"{avg_val:.0f}")
                elif is_aggregatable and val >= 10000:
                    st.metric(col, f"{val/1000:,.1f}K" if val < 1_000_000 else f"{val/1_000_000:,.1f}M")
                else:
                    st.metric(col, f"{val:,.0f}")
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        with c1:
            if cat_cols and num_cols:
                agg = df.groupby(cat_cols[0])[num_cols[0]].sum().reset_index()
                fig = go.Figure(go.Pie(
                    labels=agg[cat_cols[0]], values=agg[num_cols[0]], hole=0.6,
                    marker=dict(colors=PALETTE), textposition='outside'
                ))
                fig.update_layout(height=400, showlegend=False, **layout)
                fig.update_layout(title=dict(text=f'📊 {num_cols[0]} by {cat_cols[0]}'))
                fig.add_annotation(text=f"<b>{df[num_cols[0]].sum():,.0f}</b>", x=0.5, y=0.5, showarrow=False, font_size=16)
                st.plotly_chart(fig, use_container_width=True)
                
                # Deep Pie Chart Insights
                if DEEP_INSIGHTS_AVAILABLE:
                    insights = generate_deep_insights(df, 'pie', category=cat_cols[0], value=num_cols[0])
                    chart_explanation(insights, is_dark)
                else:
                    top = agg.nlargest(1, num_cols[0])[cat_cols[0]].values[0]
                    top_pct = agg.nlargest(1, num_cols[0])[num_cols[0]].values[0] / agg[num_cols[0]].sum() * 100
                    chart_explanation([
                        f"Donut chart showing proportional share of {num_cols[0]} across {cat_cols[0]}",
                        f"**{top}** holds the largest share at {top_pct:.1f}% of total",
                        f"Central value shows aggregate: {df[num_cols[0]].sum():,.0f}"
                    ], is_dark)
        
        with c2:
            if cat_cols and num_cols:
                agg = df.groupby(cat_cols[0])[num_cols[0]].mean().reset_index()
                fig = go.Figure(go.Scatter(
                    x=agg[cat_cols[0]], y=agg[num_cols[0]], fill='tozeroy', mode='lines+markers',
                    line=dict(color=COLORS['primary'], width=3), fillcolor='rgba(99,102,241,0.2)',
                    marker=dict(size=10)
                ))
                fig.update_layout(height=400, **layout)
                fig.update_layout(title=dict(text=f'📈 Avg {num_cols[0]} Trend'))
                st.plotly_chart(fig, use_container_width=True)
                
                max_avg = agg[num_cols[0]].max()
                min_avg = agg[num_cols[0]].min()
                chart_explanation([
                    f"Area chart showing average {num_cols[0]} performance by {cat_cols[0]}",
                    f"Range spans from {min_avg:,.0f} to {max_avg:,.0f} ({((max_avg-min_avg)/min_avg*100):.0f}% variance)",
                    f"Shaded area emphasizes magnitude differences between categories"
                ], is_dark)
        
        c1, c2 = st.columns(2)
        with c1:
            if num_cols:
                fig = go.Figure(go.Histogram(x=df[num_cols[0]], nbinsx=25, marker=dict(color=COLORS['primary'])))
                fig.update_layout(height=350, **layout)
                fig.update_layout(title=dict(text=f'📊 {num_cols[0]} Distribution'))
                st.plotly_chart(fig, use_container_width=True)
                
                # Deep Histogram Insights
                if DEEP_INSIGHTS_AVAILABLE:
                    insights = generate_deep_insights(df, 'histogram', column=num_cols[0])
                    chart_explanation(insights, is_dark)
                else:
                    skew = df[num_cols[0]].skew()
                    chart_explanation([
                        f"Histogram showing frequency distribution of {num_cols[0]}",
                        f"Data is {'right-skewed' if skew > 0.5 else 'left-skewed' if skew < -0.5 else 'normally distributed'} (skewness: {skew:.2f})",
                        f"Mean: {df[num_cols[0]].mean():,.0f} | Median: {df[num_cols[0]].median():,.0f}"
                    ], is_dark)
        
        with c2:
            if len(num_cols) >= 2:
                fig = go.Figure(go.Scatter(
                    x=df[num_cols[0]], y=df[num_cols[1]], mode='markers',
                    marker=dict(size=10, color=df[num_cols[0]], colorscale=[[0, '#6366f1'], [1, '#ec4899']], opacity=0.8)
                ))
                x_med, y_med = df[num_cols[0]].median(), df[num_cols[1]].median()
                fig.add_hline(y=y_med, line_dash="dot", line_color="gray")
                fig.add_vline(x=x_med, line_dash="dot", line_color="gray")
                fig.update_layout(height=350, **layout)
                fig.update_layout(title=dict(text=f'🎯 {num_cols[0]} vs {num_cols[1]}'))
                st.plotly_chart(fig, use_container_width=True)
                
                # Deep Scatter Insights
                if DEEP_INSIGHTS_AVAILABLE:
                    insights = generate_deep_insights(df, 'scatter', x=num_cols[0], y=num_cols[1])
                    chart_explanation(insights, is_dark)
                else:
                    q1 = len(df[(df[num_cols[0]] >= x_med) & (df[num_cols[1]] >= y_med)])
                    corr = df[num_cols[0]].corr(df[num_cols[1]])
                    chart_explanation([
                        f"Scatter plot revealing relationship between {num_cols[0]} and {num_cols[1]}",
                        f"Correlation coefficient: r={corr:.2f} ({'strong' if abs(corr)>0.7 else 'moderate' if abs(corr)>0.4 else 'weak'} relationship)",
                        f"**{q1}** points ({q1/len(df)*100:.0f}%) are high performers in both metrics (top-right quadrant)"
                    ], is_dark)
    
    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1: metric = st.selectbox("Metric", num_cols, key="d_m")
        with c2: group = st.selectbox("Group", ["None"] + cat_cols, key="d_g")
        with c3: chart = st.selectbox("Chart", ["Violin", "Box", "Histogram"], key="d_c")
        
        if chart == "Violin":
            fig = px.violin(df, y=metric, x=group if group != "None" else None, color=group if group != "None" else None,
                           color_discrete_sequence=PALETTE, box=True)
        elif chart == "Box":
            fig = px.box(df, y=metric, x=group if group != "None" else None, color=group if group != "None" else None,
                        color_discrete_sequence=PALETTE)
        else:
            fig = go.Figure(go.Histogram(x=df[metric], nbinsx=30, marker=dict(color=COLORS['primary'])))
        
        fig.update_layout(height=500, **layout)
        fig.update_layout(title=dict(text=f'📈 {metric} Distribution'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Deep Distribution Insights
        if DEEP_INSIGHTS_AVAILABLE:
            insights = generate_deep_insights(df, 'histogram', column=metric)
            chart_explanation(insights, is_dark)
        else:
            q1, q3 = df[metric].quantile(0.25), df[metric].quantile(0.75)
            iqr = q3 - q1
            outliers = len(df[(df[metric] < q1-1.5*iqr) | (df[metric] > q3+1.5*iqr)])
            chart_explanation([
                f"{'Violin' if chart=='Violin' else 'Box' if chart=='Box' else 'Histogram'} plot showing {metric} distribution{' by ' + group if group != 'None' else ''}",
                f"IQR: {q1:,.0f} to {q3:,.0f} (middle 50% of data) | {outliers} potential outliers detected",
                f"Median: {df[metric].median():,.1f} | Std Dev: {df[metric].std():,.1f}"
            ], is_dark)
    
    with tab3:
        if cat_cols and num_cols:
            c1, c2 = st.columns(2)
            with c1: group = st.selectbox("Category", cat_cols, key="comp_g")
            with c2: metric = st.selectbox("Metric", num_cols, key="comp_m")
            
            agg = df.groupby(group)[metric].agg(['sum', 'mean']).reset_index()
            agg = agg.sort_values('sum', ascending=False).head(10)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=agg[group], y=agg['sum'], name='Total', marker=dict(color=COLORS['primary'])), secondary_y=False)
            fig.add_trace(go.Scatter(x=agg[group], y=agg['mean'], name='Avg', mode='lines+markers',
                                    line=dict(color=COLORS['accent'], width=3), marker=dict(size=10)), secondary_y=True)
            fig.update_layout(height=500, **layout)
            fig.update_layout(title=dict(text=f'📊 {metric} by {group}'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Deep Comparison Insights
            if DEEP_INSIGHTS_AVAILABLE:
                insights = generate_deep_insights(df, 'bar', category=group, value=metric)
                chart_explanation(insights, is_dark)
            else:
                top = agg.iloc[0][group]
                top_total = agg.iloc[0]['sum']
                top_avg = agg.iloc[0]['mean']
                chart_explanation([
                    f"Dual-axis chart comparing total (bars) vs average (line) {metric} by {group}",
                    f"**{top}** leads with total {top_total:,.0f} and avg {top_avg:,.1f}",
                    f"Divergence between bar height and line indicates volume vs efficiency trade-offs"
                ], is_dark)
    
    with tab4:
        if len(cat_cols) >= 2 and num_cols:
            c1, c2, c3 = st.columns(3)
            with c1: row = st.selectbox("Rows", cat_cols, key="hm_r")
            with c2: col = st.selectbox("Columns", [c for c in cat_cols if c != row], key="hm_c")
            with c3: val = st.selectbox("Values", num_cols, key="hm_v")
            
            pivot = df.pivot_table(values=val, index=row, columns=col, aggfunc='mean').fillna(0)
            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale=[[0, '#1e1b4b'], [0.5, '#7c3aed'], [1, '#ddd6fe']]
            ))
            fig.update_layout(height=500, **layout)
            fig.update_layout(title=dict(text=f'🔥 {val} Heatmap'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Deep Heatmap Insights
            if DEEP_INSIGHTS_AVAILABLE:
                insights = generate_deep_insights(df, 'heatmap', row=row, col=col, value=val)
                chart_explanation(insights, is_dark)
            else:
                max_val = pivot.values.max()
                min_val = pivot.values.min()
                chart_explanation([
                    f"Heatmap showing average {val} across {row} (rows) and {col} (columns)",
                    f"Color intensity: Dark purple = low ({min_val:,.0f}), Light purple = high ({max_val:,.0f})",
                    f"Identifies hotspots: which {row}/{col} combinations perform best or need attention"
                ], is_dark)
            
        elif len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                                       colorscale='RdBu_r', zmin=-1, zmax=1))
            fig.update_layout(height=500, **layout)
            fig.update_layout(title=dict(text='🔗 Correlation Heatmap'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Deep Correlation Insights for Heatmap
            if DEEP_INSIGHTS_AVAILABLE:
                insights = generate_deep_insights(df, 'correlation', columns=num_cols)
                chart_explanation(insights, is_dark)
            else:
                chart_explanation([
                    f"Correlation matrix for all {len(num_cols)} numeric features",
                    f"Red = positive correlation, Blue = negative correlation",
                    f"Use to identify multicollinearity or predictive relationships"
                ], is_dark)


def render_analysis(df, is_dark):
    layout = get_layout(is_dark)
    st.markdown("## 🔬 Deep Analysis")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    tab1, tab2 = st.tabs(["🔍 Anomaly Detection", "🎯 Clustering"])
    
    with tab1:
        if len(num_cols) >= 2:
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler
                
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1: x = st.selectbox("X", num_cols, key="an_x")
                with c2: y = st.selectbox("Y", [c for c in num_cols if c != x], key="an_y")
                with c3: sens = st.slider("Sensitivity", 0.01, 0.20, 0.10)
                
                X = df[[x, y]].dropna()
                preds = IsolationForest(contamination=sens, random_state=42).fit_predict(StandardScaler().fit_transform(X))
                
                df_plot = X.copy()
                df_plot['Type'] = ['Anomaly' if p == -1 else 'Normal' for p in preds]
                
                fig = go.Figure()
                for t, c in [('Normal', COLORS['primary']), ('Anomaly', COLORS['danger'])]:
                    m = df_plot['Type'] == t
                    fig.add_trace(go.Scatter(x=df_plot[m][x], y=df_plot[m][y], mode='markers', name=t,
                                            marker=dict(size=10, color=c)))
                fig.update_layout(height=500, **layout)
                fig.update_layout(title=dict(text='🔍 Anomaly Detection'))
                st.plotly_chart(fig, use_container_width=True)
                
                anomaly_count = sum(preds==-1)
                chart_explanation([
                    f"Isolation Forest ML algorithm detecting unusual patterns in {x} vs {y}",
                    f"**{anomaly_count} anomalies** ({anomaly_count/len(preds)*100:.1f}%) flagged as significantly different from normal patterns",
                    f"Red points warrant investigation - may indicate errors, fraud, or exceptional cases"
                ], is_dark)
                
                st.metric("Anomalies", f"{anomaly_count} ({anomaly_count/len(preds)*100:.1f}%)")
            except ImportError:
                st.warning("Install scikit-learn")
    
    with tab2:
        if len(num_cols) >= 2:
            try:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1: x = st.selectbox("X", num_cols, key="cl_x")
                with c2: y = st.selectbox("Y", [c for c in num_cols if c != x], key="cl_y")
                with c3: n = st.slider("Clusters", 2, 6, 3)
                
                X = df[[x, y]].dropna()
                km = KMeans(n_clusters=n, random_state=42, n_init=10)
                clusters = km.fit_predict(StandardScaler().fit_transform(X))
                
                fig = go.Figure()
                for i in range(n):
                    m = clusters == i
                    fig.add_trace(go.Scatter(x=X[m][x], y=X[m][y], mode='markers', name=f'Cluster {i+1}',
                                            marker=dict(size=10, color=PALETTE[i])))
                fig.update_layout(height=500, **layout)
                fig.update_layout(title=dict(text='🎯 Clustering'))
                st.plotly_chart(fig, use_container_width=True)
                
                sizes = [sum(clusters==i) for i in range(n)]
                largest = max(sizes)
                chart_explanation([
                    f"K-Means clustering grouping data into {n} natural segments based on {x} and {y}",
                    f"Largest cluster contains **{largest}** records ({largest/len(X)*100:.0f}%) - your dominant customer/pattern type",
                    f"Each color represents a distinct segment with similar characteristics for targeted strategies"
                ], is_dark)
            except ImportError:
                st.warning("Install scikit-learn")


def render_chat(df):
    st.markdown("## 🤖 AI Chat")
    
    API_KEY = "AIzaSyAuUHyAuMkL6YQq1ass3jIlgNykLD65L-Q"
    
    model = None
    ai = False
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        ai = True
        st.success("✅ AI Connected (Gemini)")
    except ImportError:
        st.error("❌ Install google-generativeai: `pip install google-generativeai`")
    except Exception as e:
        ai = False
        st.warning("⚠️ AI Unavailable - Using Local Analysis Mode")
    
    st.markdown("---")
    
    # Quick actions
    cols = st.columns(3)
    actions = ["📊 Summarize data", "🔍 Find patterns", "💡 Suggest analysis"]
    for i, action in enumerate(actions):
        with cols[i]:
            if st.button(action, key=f"q_{i}", use_container_width=True):
                query = action.split(" ", 1)[1]
                st.session_state['messages'].append({'role': 'user', 'content': query})
                process_query(df, query, model if ai else None)
                st.rerun()
    
    st.markdown("---")
    
    # Chat history
    for msg in st.session_state['messages'][-8:]:
        with st.chat_message("user" if msg['role'] == 'user' else "assistant"):
            st.write(msg['content'])
    
    # Chat input
    if query := st.chat_input("Ask ANYTHING about your data..."):
        st.session_state['messages'].append({'role': 'user', 'content': query})
        process_query(df, query, model if ai else None)
        st.rerun()


def process_query(df, query, model):
    """Process reasoning and generate response"""
    if model:
        try:
            # Build Rich Context
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in df.columns if df[c].dtype == 'object']
            
            context = f"""
Dataset Overview:
- Rows: {len(df)}
- Columns: {len(df.columns)} ({', '.join(df.columns)})
- Missing Values: {df.isnull().sum().to_dict()}

Numeric Stats:
{df.describe().to_string()}

Categorical Samples:
"""
            for c in cat_cols[:3]:
                try:
                    context += f"- {c}: {df[c].dropna().unique()[:5].tolist()}... (Top: {df[c].mode()[0] if not df[c].mode().empty else 'N/A'})\n"
                except: pass
            
            if len(num_cols) > 1:
                try: context += f"\nCorrelations:\n{df[num_cols].corr().to_string()}"
                except: pass

            prompt = f"""You are an elite Data Scientist. User Query: "{query}"

DATA CONTEXT:
{context}

INSTRUCTIONS:
1. Answer the user's question explicitly based accurately on the data provided.
2. If asking for specific values (highest, lowest, average), use the stats provided.
3. If asking for relationships, refer to the confusion matrix/correlations.
4. BE CONCISE but COMPREHENSIVE. Do not use markdown for tables, use lists or text.
5. If the answer is not in the data, explain why based on available columns.
6. YOU CAN ANSWER ANYTHING based on this context. Be confident."""

            resp = model.generate_content(prompt).text
        except Exception as e:
            resp = get_smart_fallback(df, query) + f"\n\n(AI Error: {str(e)})"
    else:
        resp = get_smart_fallback(df, query)
    
    st.session_state['messages'].append({'role': 'ai', 'content': resp})


def get_smart_fallback(df, query):
    """Smart fallback when AI is unavailable."""
    q = query.lower()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    
    if 'summar' in q:
        return f"""📊 **Dataset Summary**
- **Records**: {len(df):,} rows
- **Features**: {len(df.columns)} columns ({len(num_cols)} numeric, {len(cat_cols)} categorical)
- **Top Metrics**:
  - {num_cols[0]}: Avg {df[num_cols[0]].mean():,.1f}
  - {num_cols[1] if len(num_cols)>1 else ''}: Avg {df[num_cols[1]].mean():,.1f} if len(num_cols)>1 else ''
- **Missing**: {df.isnull().sum().sum()} values"""
    
    elif 'pattern' in q or 'trend' in q:
        insights = []
        if len(num_cols) >= 2:
            corr = df[num_cols[0]].corr(df[num_cols[1]])
            insights.append(f"- **Correlation**: {num_cols[0]} vs {num_cols[1]} (r={corr:.2f})")
        if cat_cols and num_cols:
            top = df.groupby(cat_cols[0])[num_cols[0]].sum().idxmax()
            insights.append(f"- **Top Performer**: {top} leads in {num_cols[0]}")
        return "🔍 **Patterns Detected**\n" + "\n".join(insights) if insights else "No strong patterns found automatically."
    
    elif 'highest' in q or 'max' in q:
        if num_cols:
            max_val = df[num_cols[0]].max()
            return f"📈 **Highest {num_cols[0]}**: {max_val:,.0f}"
    
    elif 'lowest' in q or 'min' in q:
        if num_cols:
            min_val = df[num_cols[0]].min()
            return f"📉 **Lowest {num_cols[0]}**: {min_val:,.0f}"

    return """💡 **Analysis Capabilities**
I can help you analyze this data. Try asking:
- "Summarize the dataset"
- "Find patterns"
- "What is the highest value?"
- "Show correlations"
- "Suggest analysis" """


def render_data(df):
    st.markdown("## 📋 Data")
    
    if st.session_state.get('cleaning_report'):
        r = st.session_state['cleaning_report']
        with st.expander("🧹 Data Cleaning Report (Applied)", expanded=True):
            cols = st.columns(4)
            cols[0].metric("Duplicates Removed", r['duplicates'])
            cols[1].metric("Type Corrections", r['converted'])
            cols[2].metric("Missing Filled", r['filled'])
            cols[3].metric("Total Rows", r['rows'])
            
    cols = st.multiselect("Columns", df.columns.tolist(), default=df.columns.tolist()[:8])
    st.dataframe(df[cols] if cols else df, use_container_width=True, height=500)
    
    with st.expander("📊 Stats"):
        st.dataframe(df.describe().round(2), use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1: st.download_button("⬇️ Cleaned Data (CSV)", df.to_csv(index=False), "cleaned_data.csv", use_container_width=True)
    with c2:
        try:
            import io
            buf = io.BytesIO()
            df.to_excel(buf, index=False, engine='openpyxl')
            st.download_button("⬇️ Cleaned Data (Excel)", buf.getvalue(), "cleaned_data.xlsx", use_container_width=True)
        except: pass


if __name__ == "__main__":
    main()
