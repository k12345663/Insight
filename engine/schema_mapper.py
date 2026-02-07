"""
InsightGenie AI - Schema Mapper
Auto-detects column semantics and maps to visualization templates
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
import re


class SchemaMapper:
    """
    Intelligent schema mapper that detects column roles for dashboard generation.
    Maps columns to roles like: state, party, votes, winner, margin, turnout, etc.
    """
    
    # Column role patterns (regex-based)
    ROLE_PATTERNS = {
        "state": [r"state", r"province", r"region", r"location", r"area", r"zone"],
        "constituency": [r"constituency", r"district", r"seat", r"ward", r"electorate"],
        "party": [r"party", r"parties", r"political", r"coalition"],
        "candidate": [r"candidate", r"winner", r"elected", r"name", r"member"],
        "votes": [r"vote", r"votes", r"total_votes", r"vote_count", r"ballots"],
        "margin": [r"margin", r"lead", r"difference", r"gap", r"spread"],
        "turnout": [r"turnout", r"participation", r"voter_turnout", r"voting_rate"],
        "percentage": [r"percent", r"share", r"pct", r"rate", r"%"],
        "year": [r"year", r"election_year", r"date", r"period"],
        "seats": [r"seat", r"seats", r"won", r"wins", r"elected"],
        "contested": [r"contested", r"fought", r"fielded", r"candidates_fielded"],
    }
    
    # Template mappings for different dashboard types
    TEMPLATES = {
        "election": {
            "required": ["state", "party", "votes"],
            "optional": ["margin", "turnout", "seats", "contested", "candidate"],
            "kpi_metrics": ["total_votes", "avg_turnout", "total_seats"],
            "visualizations": ["heatmap", "quadrant", "distribution", "efficiency"]
        },
        "generic": {
            "required": ["category"],
            "optional": ["value", "date", "subcategory"],
            "kpi_metrics": ["total", "average", "count"],
            "visualizations": ["bar", "line", "pie", "scatter"]
        }
    }
    
    def __init__(self):
        self.column_mapping = {}
        self.detected_template = None
        self.confidence_scores = {}
    
    def detect_schema(self, df: pd.DataFrame) -> Dict:
        """
        Analyze DataFrame and detect column roles.
        Returns mapping of column names to semantic roles.
        """
        self.column_mapping = {}
        self.confidence_scores = {}
        
        for col in df.columns:
            role, confidence = self._detect_column_role(col, df[col])
            if role:
                self.column_mapping[col] = role
                self.confidence_scores[col] = confidence
        
        # Detect best-fit template
        self.detected_template = self._detect_template()
        
        return {
            "mapping": self.column_mapping,
            "confidence": self.confidence_scores,
            "template": self.detected_template,
            "unmapped": [c for c in df.columns if c not in self.column_mapping]
        }
    
    def _detect_column_role(self, col_name: str, series: pd.Series) -> Tuple[Optional[str], float]:
        """
        Detect the semantic role of a column based on name and content.
        Returns (role, confidence_score)
        """
        col_lower = col_name.lower()
        
        # Name-based detection
        for role, patterns in self.ROLE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, col_lower):
                    return role, 0.9  # High confidence for name match
        
        # Content-based detection
        return self._detect_from_content(series)
    
    def _detect_from_content(self, series: pd.Series) -> Tuple[Optional[str], float]:
        """Infer column role from its data content."""
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            sample = series.dropna()
            
            # Check if percentage (0-100 range)
            if sample.min() >= 0 and sample.max() <= 100:
                if sample.mean() > 10:  # Likely percentage
                    return "percentage", 0.6
            
            # Check if year
            if sample.min() > 1900 and sample.max() < 2100:
                if (sample == sample.astype(int)).all():
                    return "year", 0.7
            
            # Large numbers likely votes
            if sample.mean() > 10000:
                return "votes", 0.5
            
            return "value", 0.4
        
        # Check if categorical
        if series.dtype == 'object':
            unique_ratio = series.nunique() / len(series)
            
            # Low unique ratio = likely category
            if unique_ratio < 0.1:
                # Check content patterns
                sample_values = series.dropna().head(10).str.lower().tolist()
                
                # Party-like names
                party_keywords = ['party', 'congress', 'bjp', 'inc', 'aap', 'democrat', 'republican']
                if any(kw in ' '.join(sample_values) for kw in party_keywords):
                    return "party", 0.7
                
                # State-like patterns (2-3 letter codes or title case)
                if all(len(str(v)) <= 30 for v in sample_values[:5]):
                    return "state", 0.5
                
                return "category", 0.4
            
            # High unique ratio = likely identifier
            if unique_ratio > 0.9:
                return "identifier", 0.5
        
        return None, 0.0
    
    def _detect_template(self) -> str:
        """Detect the best-fit dashboard template based on mapped columns."""
        roles = set(self.column_mapping.values())
        
        # Check election template
        election_required = set(self.TEMPLATES["election"]["required"])
        election_optional = set(self.TEMPLATES["election"]["optional"])
        
        if election_required.issubset(roles):
            return "election"
        elif len(roles & (election_required | election_optional)) >= 2:
            return "election"
        
        return "generic"
    
    def get_column_for_role(self, role: str) -> Optional[str]:
        """Get the column name mapped to a specific role."""
        for col, mapped_role in self.column_mapping.items():
            if mapped_role == role:
                return col
        return None
    
    def get_columns_by_type(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize columns by data type for visualization suggestions."""
        return {
            "numeric": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "categorical": [c for c in df.select_dtypes(include=['object']).columns 
                           if df[c].nunique() < len(df) * 0.3],
            "datetime": df.select_dtypes(include=['datetime64']).columns.tolist(),
            "identifier": [c for c in df.select_dtypes(include=['object']).columns 
                          if df[c].nunique() > len(df) * 0.9]
        }
    
    def suggest_visualizations(self, df: pd.DataFrame) -> List[Dict]:
        """
        Suggest visualizations based on detected schema.
        Returns list of visualization configs.
        """
        suggestions = []
        roles = self.column_mapping
        col_types = self.get_columns_by_type(df)
        
        # KPI Cards - numeric columns
        for col in col_types["numeric"][:4]:
            suggestions.append({
                "type": "kpi",
                "column": col,
                "aggregation": "sum" if "vote" in col.lower() else "mean",
                "title": col.replace("_", " ").title()
            })
        
        # Heatmap - if we have state and a numeric value
        state_col = self.get_column_for_role("state")
        if state_col and col_types["numeric"]:
            suggestions.append({
                "type": "heatmap",
                "x": state_col,
                "y": col_types["numeric"][0],
                "title": f"{state_col.title()} Intensity"
            })
        
        # Quadrant Plot - if we have two numeric columns
        if len(col_types["numeric"]) >= 2:
            suggestions.append({
                "type": "scatter",
                "x": col_types["numeric"][0],
                "y": col_types["numeric"][1],
                "title": "Quadrant Analysis"
            })
        
        # Distribution - for margin or percentage columns
        margin_col = self.get_column_for_role("margin") or self.get_column_for_role("percentage")
        if margin_col:
            suggestions.append({
                "type": "distribution",
                "column": margin_col,
                "title": f"{margin_col.replace('_', ' ').title()} Distribution"
            })
        
        # Bar chart - categorical vs numeric
        if col_types["categorical"] and col_types["numeric"]:
            suggestions.append({
                "type": "bar",
                "x": col_types["categorical"][0],
                "y": col_types["numeric"][0],
                "title": f"{col_types['categorical'][0].title()} Breakdown"
            })
        
        return suggestions


def detect_schema(df: pd.DataFrame) -> Dict:
    """
    Convenience function to detect schema.
    Returns schema mapping with roles, confidence, and template.
    """
    mapper = SchemaMapper()
    return mapper.detect_schema(df)
