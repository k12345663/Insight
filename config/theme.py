"""
InsightGenie AI - Theme Configuration
Professional, polished theme with Light/Dark mode support
"""

# Theme Definitions
THEMES = {
    "dark": {
        "name": "Dark",
        "background": "#0f0f0f",
        "surface": "#1a1a1a",
        "card": "#252525",
        "card_hover": "#2d2d2d",
        "border": "#333333",
        "text_primary": "#ffffff",
        "text_secondary": "#a0a0a0",
        "text_muted": "#666666",
        "accent": "#3b82f6",
        "accent_hover": "#2563eb",
        "success": "#22c55e",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "chart_bg": "#1a1a1a",
        "plotly_template": "plotly_dark"
    },
    "light": {
        "name": "Light",
        "background": "#f8fafc",
        "surface": "#ffffff",
        "card": "#ffffff",
        "card_hover": "#f1f5f9",
        "border": "#e2e8f0",
        "text_primary": "#0f172a",
        "text_secondary": "#475569",
        "text_muted": "#94a3b8",
        "accent": "#3b82f6",
        "accent_hover": "#2563eb",
        "success": "#16a34a",
        "warning": "#d97706",
        "danger": "#dc2626",
        "chart_bg": "#ffffff",
        "plotly_template": "plotly_white"
    }
}

# Chart color palettes (works on both themes)
CHART_COLORS = ["#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b", "#22c55e", "#06b6d4"]


def get_theme(theme_name: str = "dark") -> dict:
    """Get theme configuration by name."""
    return THEMES.get(theme_name, THEMES["dark"])


def get_plotly_layout(theme: dict) -> dict:
    """Get Plotly layout configuration for the current theme."""
    return {
        "paper_bgcolor": theme["chart_bg"],
        "plot_bgcolor": theme["chart_bg"],
        "font": {
            "color": theme["text_primary"],
            "family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            "size": 12
        },
        "xaxis": {
            "gridcolor": theme["border"],
            "linecolor": theme["border"],
            "tickfont": {"color": theme["text_secondary"]}
        },
        "yaxis": {
            "gridcolor": theme["border"],
            "linecolor": theme["border"],
            "tickfont": {"color": theme["text_secondary"]}
        },
        "colorway": CHART_COLORS,
        "hoverlabel": {
            "bgcolor": theme["surface"],
            "font": {"color": theme["text_primary"]},
            "bordercolor": theme["border"]
        },
        "margin": {"l": 60, "r": 30, "t": 50, "b": 60}
    }


def get_app_css(theme: dict) -> str:
    """Generate dynamic CSS based on current theme."""
    return f"""
    <style>
        /* === Professional Typography === */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        
        /* === App Container === */
        .stApp {{
            background-color: {theme["background"]};
        }}
        
        /* === Sidebar === */
        section[data-testid="stSidebar"] {{
            background-color: {theme["surface"]};
            border-right: 1px solid {theme["border"]};
        }}
        
        section[data-testid="stSidebar"] .stMarkdown {{
            color: {theme["text_primary"]};
        }}
        
        /* === Headers === */
        h1, h2, h3, h4, h5, h6 {{
            color: {theme["text_primary"]} !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }}
        
        h1 {{ font-size: 2rem !important; }}
        h2 {{ font-size: 1.5rem !important; }}
        h3 {{ font-size: 1.25rem !important; }}
        
        /* === Text === */
        p, span, label, .stMarkdown {{
            color: {theme["text_primary"]};
        }}
        
        /* === Cards === */
        .metric-card {{
            background: {theme["card"]};
            border: 1px solid {theme["border"]};
            border-radius: 12px;
            padding: 1.25rem;
            transition: all 0.2s ease;
        }}
        
        .metric-card:hover {{
            background: {theme["card_hover"]};
            border-color: {theme["accent"]};
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .metric-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: {theme["accent"]};
            margin: 0.25rem 0;
        }}
        
        .metric-label {{
            font-size: 0.75rem;
            font-weight: 500;
            color: {theme["text_secondary"]};
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        /* === Buttons === */
        .stButton > button {{
            background: {theme["accent"]};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            font-size: 0.875rem;
            transition: all 0.2s ease;
        }}
        
        .stButton > button:hover {{
            background: {theme["accent_hover"]};
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        }}
        
        /* === Inputs === */
        .stSelectbox > div > div,
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {{
            background-color: {theme["surface"]};
            border: 1px solid {theme["border"]};
            border-radius: 8px;
            color: {theme["text_primary"]};
        }}
        
        .stSelectbox > div > div:focus-within,
        .stTextInput > div > div > input:focus {{
            border-color: {theme["accent"]};
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
        }}
        
        /* === File Uploader === */
        .stFileUploader {{
            border: 2px dashed {theme["border"]};
            border-radius: 12px;
            padding: 2rem;
            background: {theme["surface"]};
            transition: all 0.2s ease;
        }}
        
        .stFileUploader:hover {{
            border-color: {theme["accent"]};
            background: {theme["card_hover"]};
        }}
        
        /* === Tabs === */
        .stTabs [data-baseweb="tab-list"] {{
            background: {theme["surface"]};
            border-radius: 10px;
            padding: 4px;
            gap: 4px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border-radius: 8px;
            color: {theme["text_secondary"]};
            font-weight: 500;
            padding: 0.5rem 1rem;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: {theme["accent"]};
            color: white;
        }}
        
        /* === Expander === */
        .streamlit-expanderHeader {{
            background: {theme["surface"]};
            border: 1px solid {theme["border"]};
            border-radius: 8px;
            color: {theme["text_primary"]};
            font-weight: 500;
        }}
        
        /* === Data Table === */
        .stDataFrame {{
            border: 1px solid {theme["border"]};
            border-radius: 8px;
            overflow: hidden;
        }}
        
        /* === Metrics (native) === */
        [data-testid="metric-container"] {{
            background: {theme["card"]};
            border: 1px solid {theme["border"]};
            border-radius: 12px;
            padding: 1rem;
        }}
        
        [data-testid="metric-container"] label {{
            color: {theme["text_secondary"]} !important;
        }}
        
        [data-testid="metric-container"] [data-testid="stMetricValue"] {{
            color: {theme["accent"]} !important;
        }}
        
        /* === Chat Messages === */
        .chat-user {{
            background: {theme["accent"]};
            color: white;
            padding: 0.75rem 1rem;
            border-radius: 12px 12px 4px 12px;
            max-width: 80%;
            margin-left: auto;
            margin-bottom: 0.5rem;
        }}
        
        .chat-assistant {{
            background: {theme["surface"]};
            border: 1px solid {theme["border"]};
            color: {theme["text_primary"]};
            padding: 0.75rem 1rem;
            border-radius: 12px 12px 12px 4px;
            max-width: 80%;
            margin-bottom: 0.5rem;
        }}
        
        /* === Status badges === */
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        
        .badge-success {{
            background: rgba(34, 197, 94, 0.1);
            color: {theme["success"]};
        }}
        
        .badge-warning {{
            background: rgba(245, 158, 11, 0.1);
            color: {theme["warning"]};
        }}
        
        /* === Hide Streamlit branding === */
        #MainMenu, footer, header {{
            visibility: hidden;
        }}
        
        /* === Scrollbar === */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {theme["background"]};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {theme["border"]};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {theme["text_muted"]};
        }}
    </style>
    """


# Page configuration
PAGE_CONFIG = {
    "page_title": "InsightGenie AI",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}


# For backwards compatibility
COLORS = THEMES["dark"]
PLOTLY_TEMPLATE = {"layout": get_plotly_layout(THEMES["dark"])}
KPI_STYLE = ""
APP_CSS = get_app_css(THEMES["dark"])
