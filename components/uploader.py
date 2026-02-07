"""
InsightGenie AI - CSV Uploader Component
Drag-and-drop file upload with preview and validation
"""

import streamlit as st
import pandas as pd
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.sanitizer import sanitize_csv
from engine.schema_mapper import SchemaMapper


def render_uploader() -> Optional[Tuple[pd.DataFrame, dict, dict]]:
    """
    Render the CSV uploader component.
    Returns (cleaned_df, quality_report, schema_mapping) or None if no file uploaded.
    """
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h2 style="background: linear-gradient(90deg, #FF6B35, #F39C12); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.5rem; margin-bottom: 0.5rem;">
            📊 Upload Your Data
        </h2>
        <p style="color: #8B949E; font-size: 1.1rem;">
            Drop a CSV file to instantly generate your analytical dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv", "xlsx", "xls"],
        help="Supports CSV and Excel files up to 200MB",
        key="main_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Show loading state
            with st.spinner("🔄 Processing your data..."):
                # Read file based on type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Store original for comparison
                original_df = df.copy()
                
                # Sanitize data
                cleaned_df, quality_report = sanitize_csv(df)
                
                # Detect schema
                mapper = SchemaMapper()
                schema_info = mapper.detect_schema(cleaned_df)
                
            # Success message
            st.success(f"✅ Successfully loaded **{uploaded_file.name}**")
            
            # Show data summary in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Rows",
                    f"{len(cleaned_df):,}",
                    delta=f"-{len(original_df) - len(cleaned_df)}" if len(original_df) != len(cleaned_df) else None,
                    help="Number of rows after cleaning"
                )
            
            with col2:
                st.metric(
                    "Total Columns",
                    len(cleaned_df.columns),
                    help="Number of data columns"
                )
            
            with col3:
                mapped_count = len(schema_info["mapping"])
                st.metric(
                    "Auto-Mapped",
                    f"{mapped_count}/{len(cleaned_df.columns)}",
                    help="Columns automatically mapped to semantic roles"
                )
            
            with col4:
                template = schema_info["template"].title()
                st.metric(
                    "Template",
                    template,
                    help="Detected dashboard template"
                )
            
            # Expandable data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(
                    cleaned_df.head(10),
                    use_container_width=True,
                    height=300
                )
            
            # Column mapping display
            with st.expander("🗺️ Detected Schema Mapping"):
                if schema_info["mapping"]:
                    mapping_cols = st.columns(3)
                    for i, (col, role) in enumerate(schema_info["mapping"].items()):
                        with mapping_cols[i % 3]:
                            confidence = schema_info["confidence"].get(col, 0)
                            conf_color = "#2ECC71" if confidence > 0.7 else "#F39C12" if confidence > 0.5 else "#E74C3C"
                            st.markdown(f"""
                            <div style="background: #1E2329; padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem;">
                                <div style="color: #8B949E; font-size: 0.8rem;">{col}</div>
                                <div style="color: #FFFFFF; font-weight: 600;">→ {role.title()}</div>
                                <div style="color: {conf_color}; font-size: 0.75rem;">
                                    Confidence: {confidence:.0%}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No semantic mappings detected. Data will use generic visualizations.")
                
                if schema_info["unmapped"]:
                    st.caption(f"Unmapped columns: {', '.join(schema_info['unmapped'])}")
            
            # Quality report
            with st.expander("📊 Data Quality Report"):
                report = quality_report
                
                st.markdown("**Cleaning Operations Performed:**")
                for log in report.get("cleaning_log", []):
                    step = log.get("step", "")
                    if step == "Missing Value Handling":
                        details = log.get("details", {})
                        if details:
                            st.markdown("- **Missing Values Fixed:**")
                            for col, info in details.items():
                                st.markdown(f"  - `{col}`: {info['count']} values ({info['percentage']}%)")
                    elif step == "Type Casting":
                        changes = log.get("changes", {})
                        if changes:
                            st.markdown("- **Type Conversions:**")
                            for col, change in changes.items():
                                st.markdown(f"  - `{col}`: {change}")
                    elif step == "Duplicate Removal":
                        removed = log.get("removed", 0)
                        if removed > 0:
                            st.markdown(f"- **Duplicates Removed:** {removed} rows")
            
            # Store in session state
            st.session_state['uploaded_data'] = cleaned_df
            st.session_state['quality_report'] = quality_report
            st.session_state['schema_mapping'] = schema_info
            st.session_state['file_name'] = uploaded_file.name
            
            return cleaned_df, quality_report, schema_info
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
            return None
    
    # Show sample data option
    st.markdown("---")
    st.markdown("**🎯 No data? Try our sample dataset:**")
    
    if st.button("Load Sample Election Data", type="secondary"):
        sample_df = _generate_sample_data()
        cleaned_df, quality_report = sanitize_csv(sample_df)
        mapper = SchemaMapper()
        schema_info = mapper.detect_schema(cleaned_df)
        
        st.session_state['uploaded_data'] = cleaned_df
        st.session_state['quality_report'] = quality_report
        st.session_state['schema_mapping'] = schema_info
        st.session_state['file_name'] = "sample_election_data.csv"
        
        st.rerun()
    
    return None


def _generate_sample_data() -> pd.DataFrame:
    """Generate sample election dataset for demo purposes."""
    import random
    
    states = ["Maharashtra", "Uttar Pradesh", "Bihar", "West Bengal", "Tamil Nadu",
              "Karnataka", "Gujarat", "Rajasthan", "Madhya Pradesh", "Kerala"]
    parties = ["BJP", "INC", "AAP", "TMC", "SP", "BSP", "DMK", "JDU", "SS", "NCP"]
    
    data = []
    for state in states:
        for i in range(random.randint(15, 30)):
            winner_party = random.choice(parties)
            total_votes = random.randint(50000, 500000)
            margin = random.randint(500, 100000)
            turnout = random.uniform(55, 85)
            
            data.append({
                "state": state,
                "constituency": f"{state} - Constituency {i+1}",
                "winner_party": winner_party,
                "winner_candidate": f"Candidate_{random.randint(1000, 9999)}",
                "total_votes": total_votes,
                "margin": margin,
                "margin_percent": round((margin / total_votes) * 100, 2),
                "turnout_percent": round(turnout, 2),
                "year": 2024
            })
    
    return pd.DataFrame(data)


def get_uploaded_data() -> Optional[pd.DataFrame]:
    """Get the currently uploaded data from session state."""
    return st.session_state.get('uploaded_data')


def get_schema_mapping() -> Optional[dict]:
    """Get the schema mapping from session state."""
    return st.session_state.get('schema_mapping')
