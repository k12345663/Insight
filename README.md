# InsightGenie AI - Premium Analytics Dashboard

**InsightGenie AI** is a state-of-the-art, machine learning-powered analytics platform designed to transform raw datasets into actionable intelligence. It combines premium aesthetics with elite AI capabilities to provide a "Data Scientist in a Box" experience.

---

## 🚀 Key Features

### 1. 🤖 Elite AI Chat ("Answer Anything")
- **Engine**: Powered by **Google Gemini 1.5 Flash**.
- **Context-Aware**: The AI receives a full statistical profile of your data (missing values, correlations, categorical samples) to answer complex queries.
- **Robust Fallback**: Automatically switches to a local pattern-matching engine if the API is unavailable.
- **System Prompt**: "You are an elite Data Scientist... Answer explicitly."

### 2. 🧹 Intelligent Data Cleaning Pipeline
- **Auto-Cleaning**: Automatically processes every upload.
  - **Deduplication**: Removes duplicate records.
  - **Type Inference**: Converts text to numbers where possible.
  - **Sanitization**: Trims whitespace and handles missing values (`Unknown`).
- **Transparency**: Provides a detailed **Cleaning Report** showing exactly what was fixed.
- **Export**: One-click download of the Verified Cleaned Dataset.

### 3. 📊 Premium Visualizations
- **Aesthetics**: Glassmorphism UI, "Dark Mode" locked for premium feel, and custom gradient color palettes.
- **Smart Charts**:
  - **Dynamic Bar Charts**: Top performers with gradient coloring.
  - **Correction Heatmaps**: Purple-to-Pink scale for relationship discovery.
  - **Dual-Axis Plots**: Compare magnitude (Bar) vs efficiency (Line).
- **Auto-Insights**: Every chart includes a generated text explanation highlighting key findings (e.g., "Top Performer is X with 30% share").

### 4. 🔬 Advanced Machine Learning
- **Anomaly Detection**: Uses **Isolation Forest** to identify outliers and unusual patterns.
- **Clustering**: Implements **K-Means** to segment data into logical groups.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Streamlit

### Installation
```bash
pip install streamlit pandas numpy plotly scikit-learn google-generativeai openpyxl
```

### Running the App
```bash
streamlit run app.py
```
*Note: If `streamlit` is not recognized, try running it as a Python module:*
```bash
python -m streamlit run app.py
```
The application will launch on `http://localhost:8501` (or next available port).

---

## 📂 Project Structure

```
├── app.py                # Main Application Logic
├── requirements.txt      # Dependencies
├── README.md             # Documentation
└── .streamlit/
    └── config.toml       # (Optional) Theme configuration
```

### Code Architecture (`app.py`)
- **`clean_data(df)`**: The core data pipeline. Returns cleaned dataframe + status report.
- **`render_chat(df)`**: Manages AI interactions and prompt engineering.
- **`render_insights(df)`**: Generates KPI cards and automated discovery charts.
- **`process_query()`**: Helper function to build rich context for the AI.

---

## 💡 Usage Guide

1.  **Upload**: Drag & drop your CSV/Excel file (or load sample data).
2.  **Verify**: Check the "Cleaning Report" to see what was fixed.
3.  **Explore**: Use the **Insights** tab for automatic discoveries.
4.  **Analyze**: Use **Dashboard** for interactive charts and **Analysis** for ML clustering.
5.  **Chat**: Ask questions like "Who has the highest margin?" or "Is there a correlation between price and demand?".
6.  **Export**: Download your cleaned dataset or specific charts.

---

**Built with ❤️ by InsightGenie Team**

