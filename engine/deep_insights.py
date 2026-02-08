"""
InsightGenie AI - Deep Insights Engine
Generates accurate, dataset-specific insights based on actual data values
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DatasetProfiler:
    """Smart column type detection and value formatting."""
    
    # Columns that should show exact values (no K/M abbreviation)
    EXACT_PATTERNS = ['year', 'date', 'id', 'code', 'index', 'rank', 'age', 'count', 'number']
    # Columns that can use abbreviations for large values
    AGGREGATE_PATTERNS = ['revenue', 'sales', 'amount', 'price', 'cost', 'profit', 'income', 'budget', 'spend']
    
    @staticmethod
    def clean_name(col: str) -> str:
        """Convert column name to readable format."""
        return col.replace('_', ' ').replace('-', ' ').title()
    
    @staticmethod
    def is_year_column(col_name: str, values: pd.Series) -> bool:
        """Check if column contains year data."""
        col_lower = col_name.lower()
        if 'year' in col_lower:
            return True
        # Check if values are in year range
        if values.dtype in ['int64', 'float64']:
            avg = values.mean()
            if 1900 <= avg <= 2100:
                return True
        return False
    
    @staticmethod
    def format_number(value: float, col_name: str = "") -> str:
        """Smart number formatting based on column type."""
        if pd.isna(value):
            return "N/A"
        
        col_lower = col_name.lower()
        
        # Always exact for year-like, ID, count columns
        for pattern in DatasetProfiler.EXACT_PATTERNS:
            if pattern in col_lower:
                return f"{value:,.0f}" if value == int(value) else f"{value:,.1f}"
        
        # Year-like values
        if 1900 <= value <= 2100:
            return f"{int(value)}"
        
        # Large aggregatable values can use abbreviation
        if any(p in col_lower for p in DatasetProfiler.AGGREGATE_PATTERNS):
            if abs(value) >= 1_000_000:
                return f"{value/1_000_000:,.1f}M"
            elif abs(value) >= 10_000:
                return f"{value/1000:,.1f}K"
        
        # Default: exact with commas
        if value == int(value):
            return f"{int(value):,}"
        return f"{value:,.1f}"


class InsightGenerator:
    """Generate accurate insights based on actual dataset values."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def bar_chart_insights(self, category_col: str, value_col: str) -> List[str]:
        """
        Bar Chart: Show ranking, distribution, and concentration.
        Uses actual column names and values from the dataset.
        """
        try:
            # Get actual data
            cat_display = DatasetProfiler.clean_name(category_col)
            val_display = DatasetProfiler.clean_name(value_col)
            
            grouped = self.df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
            total = grouped.sum()
            n_categories = len(grouped)
            
            if total == 0 or n_categories == 0:
                return [f"No {val_display} data available for {cat_display}"]
            
            insights = []
            
            # 1. Leader insight with actual values
            leader_name = grouped.index[0]
            leader_value = grouped.iloc[0]
            leader_pct = (leader_value / total) * 100
            leader_formatted = DatasetProfiler.format_number(leader_value, value_col)
            
            if n_categories > 1:
                second_name = grouped.index[1]
                second_value = grouped.iloc[1]
                second_pct = (second_value / total) * 100
                
                insights.append(
                    f"**{leader_name}** has the highest {val_display} ({leader_formatted}, {leader_pct:.1f}% of total), "
                    f"followed by **{second_name}** ({second_pct:.1f}%)"
                )
            else:
                insights.append(f"**{leader_name}** represents all {val_display} ({leader_formatted})")
            
            # 2. Distribution insight
            if n_categories >= 3:
                top3_value = grouped.head(3).sum()
                top3_pct = (top3_value / total) * 100
                bottom_value = grouped.tail(n_categories - 3).sum()
                bottom_pct = (bottom_value / total) * 100
                
                if top3_pct > 70:
                    insights.append(f"Top 3 {cat_display} account for {top3_pct:.0f}% of {val_display} - highly concentrated")
                elif top3_pct > 50:
                    insights.append(f"Top 3 {cat_display} hold {top3_pct:.0f}% - moderately concentrated")
                else:
                    insights.append(f"{val_display} is evenly distributed across {n_categories} {cat_display} categories")
            
            # 3. Lowest performer
            if n_categories > 2:
                lowest_name = grouped.index[-1]
                lowest_value = grouped.iloc[-1]
                lowest_pct = (lowest_value / total) * 100
                insights.append(f"**{lowest_name}** has the lowest {val_display} ({lowest_pct:.1f}% of total)")
            
            return insights
            
        except Exception as e:
            return [f"Unable to analyze: {str(e)}"]
    
    def pie_chart_insights(self, category_col: str, value_col: str) -> List[str]:
        """
        Pie/Donut Chart: Show share distribution and dominance.
        """
        try:
            cat_display = DatasetProfiler.clean_name(category_col)
            val_display = DatasetProfiler.clean_name(value_col)
            
            grouped = self.df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
            total = grouped.sum()
            n_categories = len(grouped)
            
            if total == 0:
                return [f"No {val_display} data for breakdown"]
            
            insights = []
            
            # 1. Share breakdown
            shares = []
            for i, (name, val) in enumerate(grouped.head(4).items()):
                pct = (val / total) * 100
                shares.append(f"**{name}**: {pct:.1f}%")
            
            insights.append(f"Share distribution: {', '.join(shares)}")
            
            # 2. Dominance analysis
            top_pct = (grouped.iloc[0] / total) * 100
            if top_pct > 50:
                insights.append(f"**{grouped.index[0]}** holds majority ({top_pct:.0f}%) of all {val_display}")
            elif top_pct > 30 and n_categories > 3:
                insights.append(f"**{grouped.index[0]}** is the single largest {cat_display} at {top_pct:.0f}%")
            
            # 3. Small segments
            if n_categories > 4:
                others_value = grouped.iloc[4:].sum()
                others_pct = (others_value / total) * 100
                insights.append(f"Remaining {n_categories - 4} categories together account for {others_pct:.1f}%")
            
            return insights
            
        except Exception as e:
            return [f"Unable to analyze: {str(e)}"]
    
    def histogram_insights(self, column: str) -> List[str]:
        """
        Histogram: Analyze distribution shape, central tendency, and spread.
        """
        try:
            col_display = DatasetProfiler.clean_name(column)
            values = self.df[column].dropna()
            
            if len(values) == 0:
                return [f"No {col_display} data available"]
            
            insights = []
            
            # Calculate statistics
            mean_val = values.mean()
            median_val = values.median()
            std_val = values.std()
            min_val = values.min()
            max_val = values.max()
            
            # 1. Range insight
            mean_fmt = DatasetProfiler.format_number(mean_val, column)
            min_fmt = DatasetProfiler.format_number(min_val, column)
            max_fmt = DatasetProfiler.format_number(max_val, column)
            
            insights.append(f"**{col_display}** ranges from {min_fmt} to {max_fmt} (average: {mean_fmt})")
            
            # 2. Distribution shape
            skewness = values.skew()
            if abs(skewness) < 0.5:
                insights.append("Distribution is approximately symmetric around the mean")
            elif skewness > 0.5:
                insights.append(f"Distribution is right-skewed — most values are below {mean_fmt}, with some high outliers")
            else:
                insights.append(f"Distribution is left-skewed — most values are above {mean_fmt}, with some low outliers")
            
            # 3. Variability
            cv = (std_val / mean_val * 100) if mean_val != 0 else 0
            if cv < 20:
                insights.append(f"Values are consistent (CV: {cv:.0f}%) — low variation across records")
            elif cv < 50:
                insights.append(f"Moderate variation (CV: {cv:.0f}%) in {col_display}")
            else:
                insights.append(f"High variation (CV: {cv:.0f}%) — {col_display} varies significantly across records")
            
            # 4. Outliers
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr = q3 - q1
            outliers = values[(values < q1 - 1.5*iqr) | (values > q3 + 1.5*iqr)]
            if len(outliers) > 0:
                pct = len(outliers) / len(values) * 100
                insights.append(f"{len(outliers)} potential outliers ({pct:.1f}%) detected")
            
            return insights
            
        except Exception as e:
            return [f"Unable to analyze: {str(e)}"]
    
    def scatter_insights(self, x_col: str, y_col: str) -> List[str]:
        """
        Scatter Plot: Analyze relationship between two variables.
        """
        try:
            x_display = DatasetProfiler.clean_name(x_col)
            y_display = DatasetProfiler.clean_name(y_col)
            
            data = self.df[[x_col, y_col]].dropna()
            if len(data) < 5:
                return [f"Insufficient data points for {x_display} vs {y_display}"]
            
            x, y = data[x_col], data[y_col]
            corr = x.corr(y)
            
            insights = []
            
            # 1. Correlation strength and direction
            if abs(corr) >= 0.7:
                strength = "strong"
            elif abs(corr) >= 0.4:
                strength = "moderate"
            elif abs(corr) >= 0.2:
                strength = "weak"
            else:
                strength = "very weak/no"
            
            direction = "positive" if corr > 0 else "negative"
            
            insights.append(f"**{strength.title()} {direction} correlation** (r={corr:.2f}) between {x_display} and {y_display}")
            
            # 2. Practical interpretation
            if abs(corr) >= 0.4:
                if corr > 0:
                    insights.append(f"As {x_display} increases, {y_display} tends to increase")
                else:
                    insights.append(f"As {x_display} increases, {y_display} tends to decrease")
            else:
                insights.append(f"No clear linear pattern between {x_display} and {y_display}")
            
            # 3. Extreme points
            x_max_idx = x.idxmax()
            y_max_idx = y.idxmax()
            if x_max_idx != y_max_idx:
                insights.append(f"Highest {x_display} and highest {y_display} are from different records")
            
            return insights
            
        except Exception as e:
            return [f"Unable to analyze: {str(e)}"]
    
    def correlation_matrix_insights(self, num_cols: List[str]) -> List[str]:
        """
        Correlation Matrix: Find strongest relationships between numeric columns.
        """
        try:
            if len(num_cols) < 2:
                return ["Need at least 2 numeric columns for correlation analysis"]
            
            corr_matrix = self.df[num_cols].corr()
            insights = []
            
            # Find all correlations (upper triangle only)
            pairs = []
            for i, col1 in enumerate(num_cols):
                for j, col2 in enumerate(num_cols):
                    if i < j:
                        corr_val = corr_matrix.loc[col1, col2]
                        if not pd.isna(corr_val):
                            pairs.append((col1, col2, corr_val))
            
            if not pairs:
                return ["Unable to calculate correlations"]
            
            # Sort by absolute correlation
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # 1. Strongest relationship
            top = pairs[0]
            col1_name = DatasetProfiler.clean_name(top[0])
            col2_name = DatasetProfiler.clean_name(top[1])
            
            direction = "positive" if top[2] > 0 else "negative"
            if abs(top[2]) >= 0.5:
                insights.append(f"**Strongest correlation**: {col1_name} and {col2_name} (r={top[2]:.2f}, {direction})")
            else:
                insights.append(f"**No strong correlations found**. Strongest is {col1_name} ↔ {col2_name} (r={top[2]:.2f})")
            
            # 2. Multicollinearity warning
            high_corr = [p for p in pairs if abs(p[2]) > 0.8]
            if high_corr:
                names = [f"{DatasetProfiler.clean_name(p[0])} & {DatasetProfiler.clean_name(p[1])}" for p in high_corr]
                insights.append(f"**High correlation** (>0.8) between: {', '.join(names)}")
            
            # 3. Independent features
            low_corr = [p for p in pairs if abs(p[2]) < 0.2]
            if len(low_corr) > len(pairs) * 0.5:
                insights.append("Most features are independent (low correlations)")
            
            return insights
            
        except Exception as e:
            return [f"Unable to analyze: {str(e)}"]
    
    def heatmap_insights(self, row_col: str, col_col: str, value_col: str) -> List[str]:
        """
        Heatmap: Identify hotspots and patterns in cross-tabulated data.
        """
        try:
            row_display = DatasetProfiler.clean_name(row_col)
            col_display = DatasetProfiler.clean_name(col_col)
            val_display = DatasetProfiler.clean_name(value_col)
            
            pivot = self.df.pivot_table(values=value_col, index=row_col, columns=col_col, aggfunc='mean')
            
            if pivot.empty:
                return [f"No data for {val_display} by {row_display} and {col_display}"]
            
            insights = []
            
            # Find max and min
            max_val = pivot.max().max()
            min_val = pivot.min().min()
            
            # Get positions
            max_pos = pivot.stack().idxmax()
            min_pos = pivot.stack().idxmin()
            
            max_fmt = DatasetProfiler.format_number(max_val, value_col)
            min_fmt = DatasetProfiler.format_number(min_val, value_col)
            
            # 1. Peak insight
            insights.append(f"**Highest {val_display}**: {max_pos[0]} × {max_pos[1]} = {max_fmt}")
            
            # 2. Lowest insight
            insights.append(f"**Lowest {val_display}**: {min_pos[0]} × {min_pos[1]} = {min_fmt}")
            
            # 3. Best row
            row_means = pivot.mean(axis=1)
            best_row = row_means.idxmax()
            best_row_val = DatasetProfiler.format_number(row_means.max(), value_col)
            insights.append(f"**Best performing {row_display}**: {best_row} (avg: {best_row_val})")
            
            # 4. Best column
            col_means = pivot.mean(axis=0)
            best_col = col_means.idxmax()
            best_col_val = DatasetProfiler.format_number(col_means.max(), value_col)
            insights.append(f"**Best performing {col_display}**: {best_col} (avg: {best_col_val})")
            
            return insights
            
        except Exception as e:
            return [f"Unable to analyze: {str(e)}"]


def generate_deep_insights(df: pd.DataFrame, chart_type: str, **kwargs) -> List[str]:
    """
    Main entry point - generates insights based on chart type and actual data.
    
    Args:
        df: DataFrame with the data
        chart_type: 'bar', 'pie', 'donut', 'histogram', 'scatter', 'correlation', 'heatmap'
        **kwargs: Column names specific to chart type
    
    Returns:
        List of insight strings describing the data
    """
    try:
        gen = InsightGenerator(df)
        
        if chart_type == 'bar':
            return gen.bar_chart_insights(kwargs.get('category'), kwargs.get('value'))
        elif chart_type in ('pie', 'donut'):
            return gen.pie_chart_insights(kwargs.get('category'), kwargs.get('value'))
        elif chart_type == 'histogram':
            return gen.histogram_insights(kwargs.get('column'))
        elif chart_type == 'scatter':
            return gen.scatter_insights(kwargs.get('x'), kwargs.get('y'))
        elif chart_type == 'correlation':
            return gen.correlation_matrix_insights(kwargs.get('columns', []))
        elif chart_type == 'heatmap':
            return gen.heatmap_insights(kwargs.get('row'), kwargs.get('col'), kwargs.get('value'))
        else:
            return ["Chart type not supported"]
    except Exception as e:
        return [f"Error generating insights: {str(e)}"]


# ==================== PARAMETER-WISE ANALYSIS ====================

def analyze_single_column(df: pd.DataFrame, column: str) -> Dict:
    """
    Detailed analysis of a single column/parameter.
    Returns a dictionary with comprehensive statistics.
    """
    col_display = DatasetProfiler.clean_name(column)
    result = {
        'column': column,
        'display_name': col_display,
        'dtype': str(df[column].dtype),
        'total_records': len(df),
        'non_null': df[column].notna().sum(),
        'null_count': df[column].isna().sum(),
        'null_pct': round(df[column].isna().sum() / len(df) * 100, 1),
        'unique_values': df[column].nunique(),
        'insights': []
    }
    
    values = df[column].dropna()
    
    # Numeric column analysis
    if df[column].dtype in ['int64', 'float64']:
        result['type'] = 'numeric'
        result['min'] = DatasetProfiler.format_number(values.min(), column)
        result['max'] = DatasetProfiler.format_number(values.max(), column)
        result['mean'] = DatasetProfiler.format_number(values.mean(), column)
        result['median'] = DatasetProfiler.format_number(values.median(), column)
        result['std'] = DatasetProfiler.format_number(values.std(), column)
        result['sum'] = DatasetProfiler.format_number(values.sum(), column)
        
        # Percentiles
        result['percentiles'] = {
            '25%': DatasetProfiler.format_number(values.quantile(0.25), column),
            '50%': DatasetProfiler.format_number(values.quantile(0.50), column),
            '75%': DatasetProfiler.format_number(values.quantile(0.75), column),
            '90%': DatasetProfiler.format_number(values.quantile(0.90), column)
        }
        
        # Distribution insights
        skew = values.skew()
        if abs(skew) < 0.5:
            result['insights'].append(f"{col_display} has a symmetric distribution")
        elif skew > 0:
            result['insights'].append(f"{col_display} is right-skewed (most values are low, few are very high)")
        else:
            result['insights'].append(f"{col_display} is left-skewed (most values are high, few are very low)")
        
        # Variability
        cv = (values.std() / values.mean() * 100) if values.mean() != 0 else 0
        if cv < 25:
            result['insights'].append(f"Low variability (CV: {cv:.0f}%) - consistent values")
        elif cv > 75:
            result['insights'].append(f"High variability (CV: {cv:.0f}%) - wide range of values")
        
    # Categorical column analysis
    else:
        result['type'] = 'categorical'
        value_counts = values.value_counts()
        
        # Top values
        result['top_values'] = []
        for i, (val, count) in enumerate(value_counts.head(5).items()):
            pct = count / len(values) * 100
            result['top_values'].append({
                'value': str(val),
                'count': int(count),
                'percentage': round(pct, 1)
            })
        
        # Dominant value check
        if len(value_counts) > 0:
            top_pct = value_counts.iloc[0] / len(values) * 100
            if top_pct > 50:
                result['insights'].append(f"**{value_counts.index[0]}** dominates ({top_pct:.0f}% of records)")
            elif top_pct > 30:
                result['insights'].append(f"**{value_counts.index[0]}** is most common ({top_pct:.0f}%)")
        
        # Diversity
        if result['unique_values'] == len(values):
            result['insights'].append("All values are unique (possible identifier column)")
        elif result['unique_values'] < 5:
            result['insights'].append(f"Low cardinality ({result['unique_values']} unique values)")
    
    return result


def generate_parameter_report(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Dict]:
    """
    Generate detailed parameter-wise report for selected columns.
    
    Args:
        df: DataFrame with data
        columns: Optional list of columns to analyze. If None, analyzes all columns.
    
    Returns:
        Dictionary with column name as key and analysis dict as value
    """
    if columns is None:
        columns = df.columns.tolist()
    
    report = {}
    for col in columns:
        if col in df.columns:
            try:
                report[col] = analyze_single_column(df, col)
            except Exception as e:
                report[col] = {'column': col, 'error': str(e)}
    
    return report


def format_parameter_insights(df: pd.DataFrame, columns: List[str] = None) -> List[str]:
    """
    Generate human-readable parameter-wise insights.
    
    Args:
        df: DataFrame with data
        columns: Optional list of columns to analyze
    
    Returns:
        List of formatted insight strings
    """
    report = generate_parameter_report(df, columns)
    insights = []
    
    for col, data in report.items():
        if 'error' in data:
            continue
            
        col_display = data.get('display_name', col)
        
        if data.get('type') == 'numeric':
            insights.append(f"**{col_display}**: {data['min']} to {data['max']} (avg: {data['mean']}, median: {data['median']})")
        else:
            top_vals = data.get('top_values', [])
            if top_vals:
                top_str = ", ".join([f"{v['value']} ({v['percentage']}%)" for v in top_vals[:3]])
                insights.append(f"**{col_display}**: {data['unique_values']} unique - Top: {top_str}")
        
        # Add column-specific insights
        for insight in data.get('insights', []):
            insights.append(f"  - {insight}")
    
    return insights


def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive dataset summary.
    
    Returns:
        Dictionary with overall dataset statistics
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    
    return {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(num_cols),
        'categorical_columns': len(cat_cols),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB",
        'missing_cells': int(df.isna().sum().sum()),
        'missing_pct': round(df.isna().sum().sum() / df.size * 100, 1),
        'duplicate_rows': int(df.duplicated().sum()),
        'numeric_column_names': num_cols,
        'categorical_column_names': cat_cols
    }

