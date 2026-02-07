"""
InsightGenie AI - Data Sanitizer
Handles CSV cleaning, type detection, and data quality reporting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any
from datetime import datetime
import re


class DataSanitizer:
    """
    Robust CSV sanitizer that handles messy data with smart cleaning strategies.
    """
    
    def __init__(self):
        self.quality_report = {}
        self.cleaning_log = []
    
    def sanitize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Main sanitization pipeline.
        Returns cleaned DataFrame and quality report.
        """
        original_shape = df.shape
        
        # Step 1: Clean column names
        df = self._clean_column_names(df)
        
        # Step 2: Handle missing values
        df = self._handle_missing_values(df)
        
        # Step 3: Auto-detect and cast types
        df = self._auto_cast_types(df)
        
        # Step 4: Remove duplicates
        df = self._remove_duplicates(df)
        
        # Step 5: Normalize text columns
        df = self._normalize_text(df)
        
        # Generate quality report
        self.quality_report = self._generate_report(df, original_shape)
        
        return df, self.quality_report
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names: lowercase, no special chars, underscores for spaces."""
        original_cols = df.columns.tolist()
        
        def clean_name(name):
            # Convert to string and strip
            name = str(name).strip()
            # Replace spaces and special chars with underscore
            name = re.sub(r'[^\w\s]', '', name)
            name = re.sub(r'\s+', '_', name)
            # Lowercase
            name = name.lower()
            # Remove leading/trailing underscores
            name = name.strip('_')
            return name if name else 'unnamed'
        
        new_cols = [clean_name(col) for col in df.columns]
        
        # Handle duplicate column names
        seen = {}
        final_cols = []
        for col in new_cols:
            if col in seen:
                seen[col] += 1
                final_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                final_cols.append(col)
        
        df.columns = final_cols
        
        self.cleaning_log.append({
            "step": "Column Cleaning",
            "original": original_cols,
            "cleaned": final_cols
        })
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart imputation based on column type and missing percentage."""
        missing_info = {}
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_count > 0:
                missing_info[col] = {
                    "count": int(missing_count),
                    "percentage": round(missing_pct, 2)
                }
                
                # Strategy based on missing percentage and type
                if missing_pct > 50:
                    # Too many missing - flag but keep
                    pass
                elif df[col].dtype in ['int64', 'float64'] or self._is_numeric(df[col]):
                    # Numeric: fill with median
                    try:
                        median_val = pd.to_numeric(df[col], errors='coerce').median()
                        df[col] = df[col].fillna(median_val)
                    except:
                        df[col] = df[col].fillna(0)
                else:
                    # Categorical: fill with mode or 'Unknown'
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        df[col] = df[col].fillna('Unknown')
        
        self.cleaning_log.append({
            "step": "Missing Value Handling",
            "details": missing_info
        })
        
        return df
    
    def _is_numeric(self, series: pd.Series) -> bool:
        """Check if a series can be converted to numeric."""
        try:
            pd.to_numeric(series.dropna(), errors='raise')
            return True
        except:
            return False
    
    def _auto_cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and cast column types."""
        type_changes = {}
        
        for col in df.columns:
            original_type = str(df[col].dtype)
            
            # Try numeric conversion
            if df[col].dtype == 'object':
                try:
                    # Check if it's a percentage
                    if df[col].astype(str).str.contains('%').any():
                        df[col] = df[col].astype(str).str.replace('%', '').astype(float)
                        type_changes[col] = f"{original_type} -> float (percentage)"
                        continue
                    
                    # Try integer
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if numeric_col.notna().sum() / len(df) > 0.8:  # 80% success rate
                        if (numeric_col.dropna() == numeric_col.dropna().astype(int)).all():
                            df[col] = numeric_col.fillna(0).astype(int)
                            type_changes[col] = f"{original_type} -> int"
                        else:
                            df[col] = numeric_col
                            type_changes[col] = f"{original_type} -> float"
                        continue
                except:
                    pass
                
                # Try datetime
                try:
                    date_col = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if date_col.notna().sum() / len(df) > 0.8:
                        df[col] = date_col
                        type_changes[col] = f"{original_type} -> datetime"
                        continue
                except:
                    pass
        
        self.cleaning_log.append({
            "step": "Type Casting",
            "changes": type_changes
        })
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        original_len = len(df)
        df = df.drop_duplicates()
        removed = original_len - len(df)
        
        self.cleaning_log.append({
            "step": "Duplicate Removal",
            "removed": removed
        })
        
        return df
    
    def _normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text columns: strip whitespace, consistent casing."""
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            
            # Check if it looks like a category (limited unique values)
            if df[col].nunique() < len(df) * 0.1:  # Less than 10% unique
                df[col] = df[col].str.title()
        
        return df
    
    def _generate_report(self, df: pd.DataFrame, original_shape: Tuple) -> Dict:
        """Generate comprehensive data quality report."""
        return {
            "original_shape": {"rows": original_shape[0], "cols": original_shape[1]},
            "cleaned_shape": {"rows": len(df), "cols": len(df.columns)},
            "columns": {
                col: {
                    "dtype": str(df[col].dtype),
                    "null_count": int(df[col].isna().sum()),
                    "unique_count": int(df[col].nunique()),
                    "sample_values": df[col].dropna().head(3).tolist()
                }
                for col in df.columns
            },
            "cleaning_log": self.cleaning_log,
            "timestamp": datetime.now().isoformat()
        }


def sanitize_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to sanitize a DataFrame.
    Returns (cleaned_df, quality_report)
    """
    sanitizer = DataSanitizer()
    return sanitizer.sanitize(df)
