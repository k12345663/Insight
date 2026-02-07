"""
InsightGenie AI - Story Mode (Automated Insight Generation)
LLM-driven narrative generation highlighting key patterns
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class StoryGenerator:
    """
    Generates automated narrative insights from data.
    Identifies key patterns like Razor's Edge, Unstoppables, and efficiency metrics.
    """
    
    def __init__(self, df: pd.DataFrame, schema: Optional[Dict] = None):
        self.df = df
        self.schema = schema or {}
        self.mapping = schema.get('mapping', {}) if schema else {}
        self.insights = []
    
    def generate_all_insights(self) -> List[Dict]:
        """Generate all available insights from the data."""
        self.insights = []
        
        # Razor's Edge - closest fights
        razor_insights = self._find_razors_edge()
        self.insights.extend(razor_insights)
        
        # Unstoppables - landslide victories
        landslide_insights = self._find_unstoppables()
        self.insights.extend(landslide_insights)
        
        # Efficiency metrics
        efficiency_insights = self._analyze_efficiency()
        self.insights.extend(efficiency_insights)
        
        # Top performers
        top_insights = self._find_top_performers()
        self.insights.extend(top_insights)
        
        return self.insights
    
    def _find_razors_edge(self) -> List[Dict]:
        """Find the closest fights where margin < 2%."""
        insights = []
        margin_col = self._find_column_by_role(['margin', 'percentage'])
        
        if not margin_col:
            for col in self.df.columns:
                if 'margin' in col.lower():
                    margin_col = col
                    break
        
        if margin_col and pd.api.types.is_numeric_dtype(self.df[margin_col]):
            # Determine if percentage or absolute
            max_val = self.df[margin_col].max()
            
            if max_val <= 100:  # Percentage
                close_fights = self.df[self.df[margin_col] < 2].copy()
                threshold_desc = "< 2%"
            else:  # Absolute
                threshold = self.df[margin_col].quantile(0.05)
                close_fights = self.df[self.df[margin_col] < threshold].copy()
                threshold_desc = f"< {threshold:,.0f} votes"
            
            if len(close_fights) > 0:
                close_fights = close_fights.sort_values(margin_col)
                
                # Get identifiers
                id_cols = self._get_identifier_columns()
                
                top_5_desc = []
                for _, row in close_fights.head(5).iterrows():
                    desc_parts = []
                    for col in id_cols[:2]:
                        if col in row.index:
                            desc_parts.append(str(row[col]))
                    desc_parts.append(f"margin: {row[margin_col]:.2f}")
                    top_5_desc.append(" - ".join(desc_parts))
                
                insights.append({
                    'type': 'razors_edge',
                    'icon': '🔪',
                    'title': "The Razor's Edge",
                    'subtitle': f'Closest fights ({threshold_desc})',
                    'count': len(close_fights),
                    'percentage': round(len(close_fights) / len(self.df) * 100, 1),
                    'top_items': top_5_desc,
                    'severity': 'high' if len(close_fights) > len(self.df) * 0.1 else 'normal'
                })
        
        return insights
    
    def _find_unstoppables(self) -> List[Dict]:
        """Find landslide victories where margin > 20%."""
        insights = []
        margin_col = self._find_column_by_role(['margin', 'percentage'])
        
        if not margin_col:
            for col in self.df.columns:
                if 'margin' in col.lower():
                    margin_col = col
                    break
        
        if margin_col and pd.api.types.is_numeric_dtype(self.df[margin_col]):
            max_val = self.df[margin_col].max()
            
            if max_val <= 100:  # Percentage
                landslides = self.df[self.df[margin_col] > 20].copy()
                threshold_desc = "> 20%"
            else:  # Absolute
                threshold = self.df[margin_col].quantile(0.90)
                landslides = self.df[self.df[margin_col] > threshold].copy()
                threshold_desc = f"> {threshold:,.0f} votes"
            
            if len(landslides) > 0:
                landslides = landslides.sort_values(margin_col, ascending=False)
                
                id_cols = self._get_identifier_columns()
                
                top_5_desc = []
                for _, row in landslides.head(5).iterrows():
                    desc_parts = []
                    for col in id_cols[:2]:
                        if col in row.index:
                            desc_parts.append(str(row[col]))
                    desc_parts.append(f"margin: {row[margin_col]:.2f}")
                    top_5_desc.append(" - ".join(desc_parts))
                
                insights.append({
                    'type': 'unstoppables',
                    'icon': '🏆',
                    'title': 'The Unstoppables',
                    'subtitle': f'Landslide victories ({threshold_desc})',
                    'count': len(landslides),
                    'percentage': round(len(landslides) / len(self.df) * 100, 1),
                    'top_items': top_5_desc,
                    'severity': 'success'
                })
        
        return insights
    
    def _analyze_efficiency(self) -> List[Dict]:
        """Analyze votes per seat efficiency."""
        insights = []
        
        party_col = self._find_column_by_role(['party'])
        votes_col = self._find_column_by_role(['votes'])
        
        if not party_col:
            for col in self.df.columns:
                if 'party' in col.lower() or 'winner' in col.lower():
                    party_col = col
                    break
        
        if not votes_col:
            for col in self.df.select_dtypes(include=['int64', 'float64']).columns:
                if 'vote' in col.lower():
                    votes_col = col
                    break
        
        if party_col and votes_col:
            try:
                efficiency = self.df.groupby(party_col).agg({
                    votes_col: ['sum', 'count']
                }).reset_index()
                efficiency.columns = ['party', 'votes', 'seats']
                efficiency['votes_per_seat'] = efficiency['votes'] / efficiency['seats']
                efficiency = efficiency.sort_values('votes_per_seat')
                
                most_efficient = efficiency.head(3)
                least_efficient = efficiency.tail(3).iloc[::-1]
                
                insights.append({
                    'type': 'efficiency',
                    'icon': '⚡',
                    'title': 'Conversion Efficiency',
                    'subtitle': 'Votes required per seat won',
                    'most_efficient': most_efficient[['party', 'votes_per_seat']].to_dict('records'),
                    'least_efficient': least_efficient[['party', 'votes_per_seat']].to_dict('records'),
                    'severity': 'info'
                })
            except Exception:
                pass
        
        return insights
    
    def _find_top_performers(self) -> List[Dict]:
        """Find top performers in various categories."""
        insights = []
        
        party_col = self._find_column_by_role(['party'])
        votes_col = self._find_column_by_role(['votes'])
        
        if party_col:
            try:
                top_parties = self.df[party_col].value_counts().head(5)
                
                insights.append({
                    'type': 'top_performers',
                    'icon': '🥇',
                    'title': 'Top Performers',
                    'subtitle': f'By number of records in {party_col}',
                    'ranking': [
                        {'name': name, 'count': int(count)} 
                        for name, count in top_parties.items()
                    ],
                    'severity': 'info'
                })
            except Exception:
                pass
        
        return insights
    
    def _find_column_by_role(self, roles: List[str]) -> Optional[str]:
        """Find column matching any of the given roles."""
        for col, role in self.mapping.items():
            if role in roles:
                return col
        return None
    
    def _get_identifier_columns(self) -> List[str]:
        """Get columns that could serve as identifiers."""
        identifiers = []
        
        for col, role in self.mapping.items():
            if role in ['state', 'constituency', 'party', 'candidate']:
                identifiers.append(col)
        
        if not identifiers:
            identifiers = [c for c in self.df.columns if self.df[c].dtype == 'object'][:3]
        
        return identifiers
    
    def generate_narrative(self) -> str:
        """Generate a narrative summary of all insights."""
        if not self.insights:
            self.generate_all_insights()
        
        narrative_parts = []
        
        for insight in self.insights:
            if insight['type'] == 'razors_edge':
                narrative_parts.append(
                    f"**{insight['icon']} {insight['title']}:** Found {insight['count']} extremely close contests "
                    f"({insight['percentage']}% of total). These represent the most competitive battlegrounds."
                )
            elif insight['type'] == 'unstoppables':
                narrative_parts.append(
                    f"**{insight['icon']} {insight['title']}:** {insight['count']} constituencies saw landslide victories "
                    f"({insight['percentage']}% of total), indicating strong regional dominance."
                )
            elif insight['type'] == 'efficiency':
                narrative_parts.append(
                    f"**{insight['icon']} {insight['title']}:** Vote conversion efficiency varies significantly. "
                    f"Some parties convert votes to seats more efficiently than others."
                )
            elif insight['type'] == 'top_performers':
                top = insight['ranking'][0] if insight['ranking'] else None
                if top:
                    narrative_parts.append(
                        f"**{insight['icon']} {insight['title']}:** {top['name']} leads with {top['count']} records."
                    )
        
        return "\n\n".join(narrative_parts)
    
    def generate_llm_narrative(self, api_key: Optional[str] = None) -> Optional[str]:
        """Generate an LLM-enhanced narrative if API key is available."""
        if not GEMINI_AVAILABLE or not api_key:
            return None
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Prepare data summary
            data_summary = {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'column_names': self.df.columns.tolist()[:10],
                'insights': self.insights
            }
            
            prompt = f"""You are a data analyst providing insights on a dataset. 
            Based on the following data summary and insights, generate a concise, 
            engaging narrative summary (3-4 paragraphs) highlighting the key findings:
            
            Data Summary:
            - Total records: {data_summary['rows']}
            - Key columns: {', '.join(data_summary['column_names'])}
            
            Key Insights:
            {self.generate_narrative()}
            
            Write in a professional but accessible tone, highlighting actionable insights."""
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"LLM narrative generation failed: {str(e)}"


def render_story_mode(df: pd.DataFrame, schema: Optional[Dict] = None):
    """Render the Story Mode component in Streamlit."""
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="background: linear-gradient(90deg, #FF6B35, #9B59B6);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📖 Story Mode
        </h2>
        <p style="color: #8B949E;">Automated insights extracted from your data</p>
    </div>
    """, unsafe_allow_html=True)
    
    generator = StoryGenerator(df, schema)
    insights = generator.generate_all_insights()
    
    if not insights:
        st.info("📭 No specific insights could be extracted. Try uploading election-related data for richer analysis.")
        return
    
    # Display insights as cards
    for insight in insights:
        severity_colors = {
            'high': '#E74C3C',
            'success': '#2ECC71',
            'info': '#3498DB',
            'normal': '#F39C12'
        }
        color = severity_colors.get(insight.get('severity', 'normal'), '#F39C12')
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E2329 0%, #2D3748 100%);
                    border-left: 4px solid {color};
                    border-radius: 8px;
                    padding: 1.5rem;
                    margin-bottom: 1rem;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">
                {insight.get('icon', '📊')} {insight.get('title', 'Insight')}
            </div>
            <div style="color: #8B949E; margin-bottom: 0.75rem;">
                {insight.get('subtitle', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show details based on type
        if insight['type'] in ['razors_edge', 'unstoppables']:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Count", insight.get('count', 0))
            with col2:
                st.metric("Percentage", f"{insight.get('percentage', 0)}%")
            
            if insight.get('top_items'):
                st.markdown("**Top Examples:**")
                for item in insight['top_items'][:3]:
                    st.markdown(f"- {item}")
        
        elif insight['type'] == 'efficiency':
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Most Efficient:**")
                for item in insight.get('most_efficient', [])[:3]:
                    st.markdown(f"- {item['party']}: {item['votes_per_seat']:,.0f} votes/seat")
            with col2:
                st.markdown("**Least Efficient:**")
                for item in insight.get('least_efficient', [])[:3]:
                    st.markdown(f"- {item['party']}: {item['votes_per_seat']:,.0f} votes/seat")
        
        elif insight['type'] == 'top_performers':
            for i, item in enumerate(insight.get('ranking', [])[:5], 1):
                medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i-1]
                st.markdown(f"{medal} **{item['name']}**: {item['count']:,} records")
    
    # Generate narrative summary
    st.markdown("---")
    st.markdown("### 📝 Executive Summary")
    narrative = generator.generate_narrative()
    st.markdown(narrative)
