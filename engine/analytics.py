"""
InsightGenie AI - Analytics Engine
ENP, Strike Rate, Conversion Efficiency, and advanced metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class AnalyticsEngine:
    """
    Advanced analytics engine for calculating competitive metrics.
    Includes ENP, Strike Rate, Conversion Efficiency, and more.
    """
    
    def __init__(self, df: pd.DataFrame, schema: Optional[Dict] = None):
        self.df = df
        self.schema = schema or {}
        self.mapping = schema.get('mapping', {}) if schema else {}
    
    def calculate_all_metrics(self) -> Dict:
        """
        Calculate all available metrics based on the data.
        Returns a dictionary of metric results.
        """
        metrics = {}
        
        # Try to calculate ENP
        enp_result = self.calculate_enp()
        if enp_result:
            metrics['enp'] = enp_result
        
        # Try to calculate strike rates
        strike_result = self.calculate_strike_rates()
        if strike_result:
            metrics['strike_rates'] = strike_result
        
        # Calculate conversion efficiency
        efficiency_result = self.calculate_conversion_efficiency()
        if efficiency_result:
            metrics['conversion_efficiency'] = efficiency_result
        
        # Calculate competitiveness metrics
        comp_result = self.calculate_competitiveness()
        if comp_result:
            metrics['competitiveness'] = comp_result
        
        return metrics
    
    def calculate_enp(self, group_col: Optional[str] = None) -> Optional[Dict]:
        """
        Calculate Effective Number of Parties (ENP).
        ENP = 1 / Σ(share²)
        
        A value of 2 indicates a direct two-party contest.
        A value of 4+ indicates a multi-cornered fragmented fight.
        """
        # Find vote and party columns
        party_col = group_col
        votes_col = None
        
        if not party_col:
            for col, role in self.mapping.items():
                if role == 'party':
                    party_col = col
                    break
        
        for col, role in self.mapping.items():
            if role == 'votes':
                votes_col = col
                break
        
        # Fallback: find columns by name
        if not party_col:
            for col in self.df.columns:
                if 'party' in col.lower() or 'winner' in col.lower():
                    party_col = col
                    break
        
        if not votes_col:
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if 'vote' in col.lower():
                    votes_col = col
                    break
            if not votes_col and len(numeric_cols) > 0:
                votes_col = numeric_cols[0]
        
        if not party_col or not votes_col:
            return None
        
        try:
            # Calculate vote shares
            total_votes = self.df[votes_col].sum()
            party_votes = self.df.groupby(party_col)[votes_col].sum()
            vote_shares = party_votes / total_votes
            
            # ENP formula: 1 / Σ(share²)
            enp = 1 / (vote_shares ** 2).sum()
            
            # Interpretation
            if enp < 2:
                interpretation = "Dominant single-party system"
            elif enp < 3:
                interpretation = "Two-party competitive system"
            elif enp < 4:
                interpretation = "Moderate multi-party system"
            else:
                interpretation = "Fragmented multi-party system"
            
            return {
                'value': round(enp, 2),
                'interpretation': interpretation,
                'party_shares': vote_shares.to_dict(),
                'columns_used': {'party': party_col, 'votes': votes_col}
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_enp_per_group(self, group_by: str, party_col: str, votes_col: str) -> pd.DataFrame:
        """
        Calculate ENP for each group (e.g., per state/constituency).
        """
        results = []
        
        for group_val in self.df[group_by].unique():
            subset = self.df[self.df[group_by] == group_val]
            total = subset[votes_col].sum()
            
            if total > 0:
                party_votes = subset.groupby(party_col)[votes_col].sum()
                shares = party_votes / total
                enp = 1 / (shares ** 2).sum()
                
                results.append({
                    group_by: group_val,
                    'enp': round(enp, 2),
                    'num_parties': len(party_votes),
                    'total_votes': total
                })
        
        return pd.DataFrame(results)
    
    def calculate_strike_rates(self) -> Optional[Dict]:
        """
        Calculate Strike Rate: (Wins / Contested) * 100
        Measures win efficiency for each party/group.
        """
        party_col = None
        winner_col = None
        
        # Find relevant columns
        for col, role in self.mapping.items():
            if role == 'party':
                party_col = col
            elif role in ['candidate', 'winner']:
                winner_col = col
        
        # Fallback
        if not party_col:
            for col in self.df.columns:
                if 'winner' in col.lower() and 'party' in col.lower():
                    party_col = col
                    break
                elif 'party' in col.lower():
                    party_col = col
        
        if not party_col:
            return None
        
        try:
            # Count total contests per party
            contested = self.df[party_col].value_counts()
            
            # For winner-based calculation, we assume each row is a constituency
            # and the party_col indicates the winner
            strike_rates = {}
            for party in contested.index[:20]:  # Top 20 parties
                wins = contested[party]
                # In this context, each appearance is a win (winner column)
                strike_rate = 100.0  # If party_col is winner, all entries are wins
                
                strike_rates[party] = {
                    'wins': int(wins),
                    'contested': int(wins),  # Simplified
                    'strike_rate': round(strike_rate, 2)
                }
            
            return {
                'rates': strike_rates,
                'column_used': party_col,
                'note': 'Strike rate assumes party column represents winners'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_conversion_efficiency(self) -> Optional[Dict]:
        """
        Calculate Conversion Efficiency: Votes per Seat
        Identifies if a party has widespread support but lacks concentrated strongholds.
        """
        party_col = None
        votes_col = None
        
        for col, role in self.mapping.items():
            if role == 'party':
                party_col = col
            elif role == 'votes':
                votes_col = col
        
        # Fallback
        if not party_col:
            for col in self.df.columns:
                if 'party' in col.lower():
                    party_col = col
                    break
        
        if not votes_col:
            for col in self.df.select_dtypes(include=['int64', 'float64']).columns:
                if 'vote' in col.lower():
                    votes_col = col
                    break
        
        if not party_col or not votes_col:
            return None
        
        try:
            # Votes per seat (assuming each row is a seat won)
            result = self.df.groupby(party_col).agg({
                votes_col: ['sum', 'count', 'mean']
            }).reset_index()
            result.columns = [party_col, 'total_votes', 'seats', 'avg_votes_per_seat']
            
            # Calculate efficiency
            result['votes_per_seat'] = result['total_votes'] / result['seats']
            result = result.sort_values('votes_per_seat')
            
            # Lower votes per seat = more efficient
            efficient_parties = result.head(5)[party_col].tolist()
            inefficient_parties = result.tail(5)[party_col].tolist()
            
            return {
                'data': result.to_dict('records'),
                'most_efficient': efficient_parties,
                'least_efficient': inefficient_parties,
                'columns_used': {'party': party_col, 'votes': votes_col}
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_competitiveness(self) -> Optional[Dict]:
        """
        Calculate competitiveness metrics based on margin.
        """
        margin_col = None
        
        for col, role in self.mapping.items():
            if role == 'margin':
                margin_col = col
                break
        
        if not margin_col:
            for col in self.df.columns:
                if 'margin' in col.lower():
                    margin_col = col
                    break
        
        if not margin_col:
            return None
        
        try:
            margins = self.df[margin_col].dropna()
            
            # Determine if margin is in percentage or absolute
            is_percentage = margins.max() <= 100
            
            if is_percentage:
                razor_thin = len(margins[margins < 2])
                competitive = len(margins[(margins >= 2) & (margins < 5)])
                moderate = len(margins[(margins >= 5) & (margins < 10)])
                comfortable = len(margins[(margins >= 10) & (margins < 20)])
                landslide = len(margins[margins >= 20])
            else:
                # Absolute margins - adjust thresholds
                threshold_2pct = margins.quantile(0.1)
                threshold_20pct = margins.quantile(0.8)
                
                razor_thin = len(margins[margins < threshold_2pct])
                landslide = len(margins[margins >= threshold_20pct])
                competitive = len(margins) - razor_thin - landslide
                moderate = 0
                comfortable = 0
            
            return {
                'razor_thin': razor_thin,
                'competitive': competitive if is_percentage else competitive,
                'moderate': moderate,
                'comfortable': comfortable,
                'landslide': landslide,
                'avg_margin': round(margins.mean(), 2),
                'median_margin': round(margins.median(), 2),
                'is_percentage': is_percentage,
                'column_used': margin_col
            }
        except Exception as e:
            return {'error': str(e)}


def calculate_enp(vote_shares: List[float]) -> float:
    """
    Standalone ENP calculation.
    ENP = 1 / Σ(share²)
    """
    shares = np.array(vote_shares)
    # Normalize to ensure shares sum to 1
    shares = shares / shares.sum()
    return 1 / (shares ** 2).sum()


def calculate_strike_rate(wins: int, contested: int) -> float:
    """Calculate strike rate percentage."""
    if contested == 0:
        return 0.0
    return (wins / contested) * 100


def calculate_conversion_efficiency(votes: int, seats: int) -> float:
    """Calculate votes per seat."""
    if seats == 0:
        return float('inf')
    return votes / seats
