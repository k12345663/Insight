"""
InsightGenie AI - Helper Utilities
Common utility functions
"""

import pandas as pd
from typing import List, Dict, Any, Optional


def format_large_number(num: float, indian_format: bool = True) -> str:
    """
    Format large numbers with appropriate suffixes.
    Uses Indian number system (Lakh, Crore) by default.
    """
    if pd.isna(num):
        return "N/A"
    
    if indian_format:
        if num >= 1_00_00_000:  # Crore
            return f"{num / 1_00_00_000:.2f} Cr"
        elif num >= 1_00_000:  # Lakh
            return f"{num / 1_00_000:.2f} L"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}K"
        else:
            return f"{num:,.0f}"
    else:
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}K"
        else:
            return f"{num:,.0f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def get_percentile_category(value: float, series: pd.Series) -> str:
    """Get the percentile category of a value within a series."""
    percentile = (series < value).sum() / len(series) * 100
    
    if percentile < 25:
        return "Low"
    elif percentile < 50:
        return "Below Average"
    elif percentile < 75:
        return "Above Average"
    else:
        return "High"


def detect_outliers(series: pd.Series, method: str = "iqr") -> pd.Series:
    """
    Detect outliers in a numeric series.
    Returns boolean series indicating outliers.
    """
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    elif method == "zscore":
        from scipy import stats
        z_scores = stats.zscore(series.dropna())
        return abs(z_scores) > 3
    else:
        return pd.Series([False] * len(series))


def calculate_moving_average(series: pd.Series, window: int = 3) -> pd.Series:
    """Calculate moving average of a series."""
    return series.rolling(window=window, min_periods=1).mean()


def normalize_series(series: pd.Series, method: str = "minmax") -> pd.Series:
    """Normalize a numeric series."""
    if method == "minmax":
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            return series
        return (series - min_val) / (max_val - min_val)
    elif method == "zscore":
        return (series - series.mean()) / series.std()
    else:
        return series


def create_bins(series: pd.Series, n_bins: int = 5, labels: Optional[List[str]] = None) -> pd.Series:
    """Create categorical bins from a numeric series."""
    return pd.cut(series, bins=n_bins, labels=labels)


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Get correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    return numeric_df.corr()


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive summary of a DataFrame."""
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "numeric_stats": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
        "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
    }
