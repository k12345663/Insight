# InsightGenie AI - Comprehensive Project Report

**Date**: February 2026
**Version**: 1.0
**Status**: Production Ready

---

## 1. Executive Summary

**InsightGenie AI** is an advanced, machine learning-powered analytics platform designed to bridge the gap between raw data and actionable intelligence. By integrating **Generative AI (Google Gemini 1.5 Flash)** with traditional **Data Science Algorithms (Scikit-Learn)**, the platform serves as an automated "Data Scientist in a Box," capable of cleaning, analyzing, visualizing, and explaining complex datasets to non-technical users.

---

## 2. Problem Statement

Modern businesses generate vast amounts of data, but extracting value is hindered by:
1.  **Data Quality Issues**: Duplicates, missing values, and formatting errors.
2.  **Technical Barriers**: Advanced analysis requires Python/SQL skills.
3.  **Static Reporting**: Traditional dashboards cannot answer "why" or "what if" questions.

**InsightGenie AI** solves this by automating the data pipeline and providing a natural language interface for exploration.

---

## 3. Technical Architecture & Tech Stack

The solution is built on a modular Python architecture:

| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Frontend Framework** | **Streamlit** | Rapid prototyping, interactive widgets, Python-native. |
| **Data Processing** | **Pandas + NumPy** | High-performance manipulation of structural data. |
| **AI / LLM Engine** | **Google Gemini 1.5 Flash** | Low latency, high reasoning capability, large context window. |
| **Machine Learning** | **Scikit-Learn** | Robust implementation of unsupervised learning algorithms. |
| **Visualization** | **Plotly Express** | Interactive charts, zooming/panning, publication-quality export. |
| **Environment** | **Docker** (Optional) | Containerized deployment for consistent runtime. |

---

## 4. Methodology & Algorithms

### A. Automatic Data Cleaning Pipeline (`clean_data`)
Before any analysis, data undergoes rigorous sanitization:
1.  **Deduplication**: Rows with identical values across all columns are removed.
2.  **Type Inference**: Columns containing numeric strings (e.g., "1,234") are converted to `float/int`.
3.  **String Normalization**: Whitespace is trimmed from categorical fields.
4.  **Imputation**: Missing categorical values are filled with "Unknown" to preserve row integrity.

### B. Machine Learning Models
The platform employs unsupervised learning to uncover hidden patterns:

1.  **Anomaly Detection (Isolation Forest)**
    -   **Goal**: Identify outliers (data points that significantly deviate from the norm).
    -   **Algorithm**: Constructs random decision trees. Anomalies are isolated closer to the root of the tree (shorter path length).
    -   **Application**: Fraud detection, error spotting, finding unique high-performers.

2.  **Clustering (K-Means)**
    -   **Goal**: Segment data into logical groups.
    -   **Algorithm**: Partitions $n$ observations into $k$ clusters, minimizing within-cluster variance.
    -   **Application**: Customer segmentation, product categorization.

3.  **Correlation Analysis (Pearson)**
    -   **Goal**: Quantify linear relationships between variables.
    -   **Algorithm**: Calculates correlation coefficient $r$ (-1 to +1).
    -   **Application**: Identifying key drivers (e.g., Price vs. Demand).

### C. The "Elite AI" Engine (`process_query`)
The Chat interface is context-aware. Instead of a naive LLM call, the system:
1.  **Profiles Data**: Calculates summary statistics (mean, median, mode) and samples unique values.
2.  **Injects Context**: Adds this profile to the System Prompt.
3.  **Generates Response**: The AI answers based *strictly* on the provided context, minimizing hallucinations.

---

## 5. Feature Breakdown

### 📊 Interactive Dashboard
-   **KPI Cards**: Auto-calculated totals, averages, and counts.
-   **Dynamic Charts**: Donut (Composition), Area (Trend), Bar (Comparison), Scatter (Relationship).
-   **Dual-Axis Analysis**: Compare magnitude (Bar) vs. Rate (Line) on the same chart.

### 🤖 Intelligent Chat
-   **"Ask Anything"**: Supports natural language queries (e.g., "Why is sales down?").
-   **Smart Fallback**: If the API is offline, a local regex-based engine provides basic insights.

### 🧹 Integrity Reporting
-   **Cleaning Report**: Transparently shows how many duplicates were removed and types fixed.
-   **Verified Export**: Parameters for downloading the scientifically cleaned dataset.

---

## 6. How It Works (Workflow)

1.  **Ingestion**: User uploads CSV/Excel.
2.  **Preprocessing**: Pipeline cleans data -> Report Generated.
3.  **Exploration**: User interacts with Dashboard/Insights tbas.
4.  **Deep Dive**: User runs Anomaly Detection or Clustering.
5.  **Q&A**: User asks the AI specific questions about findings.

---

## 7. Conclusion & Future Roadmap

**InsightGenie AI** successfully demonstrates how combining deterministic data pipelines with probabilistic AI models creates a robust analytics tool. 

**Future Enhancements**:
-   **Forecasting**: Integration of Prophet/ARIMA for time-series prediction.
-   **Authentication**: Multi-user login with Row-Level Security (RLS).
-   **AutoML**: Automated model selection for regression/classification tasks.

---

**Start the App:**
```bash
python -m streamlit run app.py
```
