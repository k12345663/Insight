"""
InsightGenie AI - LLM Agent
ReAct agent for processing natural language commands
"""

import json
import re
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from llm.prompts import SYSTEM_PROMPT, CHART_MODIFICATION_PROMPT


class InsightAgent:
    """
    ReAct (Reason + Act) agent for processing user commands.
    Maps natural language to data operations and visualizations.
    """
    
    def __init__(self, df: pd.DataFrame, schema: Optional[Dict] = None, api_key: Optional[str] = None):
        self.df = df
        self.schema = schema or {}
        self.api_key = api_key
        self.model = None
        
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
            except Exception:
                pass
    
    def process_command(self, command: str) -> Dict:
        """
        Process a natural language command and return the action to take.
        Uses ReAct pattern: Reason about the request, then Act.
        """
        command_lower = command.lower()
        
        # Step 1: Reason - Understand the intent
        intent = self._classify_intent(command_lower)
        
        # Step 2: Act - Execute based on intent
        if intent == "filter":
            return self._handle_filter(command_lower)
        elif intent == "chart_change":
            return self._handle_chart_change(command_lower)
        elif intent == "color_change":
            return self._handle_color_change(command_lower)
        elif intent == "analysis":
            return self._handle_analysis(command)
        elif intent == "question":
            return self._handle_question(command)
        else:
            # Try LLM for complex requests
            if self.model:
                return self._handle_llm_request(command)
            else:
                return {
                    "success": True,
                    "action": "message",
                    "message": "I understand you want to: " + command + ". Try being more specific with commands like 'filter by X', 'change chart to Y', or 'show analysis of Z'.",
                    "suggestions": ["Filter close fights", "Change to pie chart", "Show efficiency analysis"]
                }
    
    def _classify_intent(self, command: str) -> str:
        """Classify the intent of the command."""
        
        filter_keywords = ["filter", "show only", "where", "less than", "greater than", "between", "close", "exclude"]
        chart_keywords = ["change to", "switch to", "make it a", "convert to", "bar", "pie", "scatter", "line", "heatmap"]
        color_keywords = ["color", "colour", "purple", "blue", "red", "green", "orange", "theme"]
        analysis_keywords = ["analyze", "analysis", "calculate", "efficiency", "strike rate", "enp"]
        question_keywords = ["what", "which", "who", "how", "why", "tell me"]
        
        if any(kw in command for kw in filter_keywords):
            return "filter"
        elif any(kw in command for kw in chart_keywords):
            return "chart_change"
        elif any(kw in command for kw in color_keywords):
            return "color_change"
        elif any(kw in command for kw in analysis_keywords):
            return "analysis"
        elif any(kw in command for kw in question_keywords):
            return "question"
        else:
            return "unknown"
    
    def _handle_filter(self, command: str) -> Dict:
        """Handle filter requests."""
        result = {
            "success": True,
            "action": "filter",
            "filter_column": None,
            "filter_condition": None,
            "message": ""
        }
        
        # Parse common filter patterns
        if "close fight" in command or "razor" in command or "margin < 2" in command:
            margin_col = self._find_column("margin")
            if margin_col:
                result["filter_column"] = margin_col
                result["filter_condition"] = f"df['{margin_col}'] < 2"
                result["message"] = f"Filtering to show close fights where {margin_col} < 2%"
            else:
                result["message"] = "Filtering to close fights (margin < 2%)"
                result["filter_condition"] = "margin < 2"
        
        elif "landslide" in command or "margin > 20" in command:
            margin_col = self._find_column("margin")
            if margin_col:
                result["filter_column"] = margin_col
                result["filter_condition"] = f"df['{margin_col}'] > 20"
                result["message"] = f"Filtering to landslides where {margin_col} > 20%"
        
        elif "high turnout" in command:
            turnout_col = self._find_column("turnout")
            if turnout_col:
                median = self.df[turnout_col].median()
                result["filter_column"] = turnout_col
                result["filter_condition"] = f"df['{turnout_col}'] > {median}"
                result["message"] = f"Filtering to high turnout states (above median of {median:.1f}%)"
        
        elif "state" in command or "region" in command:
            # Extract state name from command
            state_col = self._find_column("state")
            if state_col:
                # Try to extract the state name
                states = self.df[state_col].unique()
                for state in states:
                    if state.lower() in command:
                        result["filter_column"] = state_col
                        result["filter_condition"] = f"df['{state_col}'] == '{state}'"
                        result["message"] = f"Filtering to {state}"
                        break
        
        else:
            result["message"] = "I'll apply the filter based on your criteria."
            result["suggestions"] = ["Try: 'show close fights'", "'filter high turnout states'", "'show landslides'"]
        
        return result
    
    def _handle_chart_change(self, command: str) -> Dict:
        """Handle chart type change requests."""
        result = {
            "success": True,
            "action": "chart_update",
            "chart_type": None,
            "message": ""
        }
        
        chart_mappings = {
            "bar": ["bar", "bar chart", "bars"],
            "pie": ["pie", "pie chart", "donut"],
            "scatter": ["scatter", "scatter plot", "points"],
            "line": ["line", "line chart", "trend"],
            "heatmap": ["heatmap", "heat map", "intensity"],
            "violin": ["violin", "density", "distribution"],
            "box": ["box", "boxplot", "box plot"]
        }
        
        for chart_type, keywords in chart_mappings.items():
            if any(kw in command for kw in keywords):
                result["chart_type"] = chart_type
                result["message"] = f"I'll update the chart to a {chart_type} visualization."
                break
        
        if not result["chart_type"]:
            result["message"] = "What type of chart would you like? Options: bar, pie, scatter, line, heatmap, violin"
            result["suggestions"] = ["bar chart", "pie chart", "scatter plot", "heatmap"]
        
        return result
    
    def _handle_color_change(self, command: str) -> Dict:
        """Handle color/theme change requests."""
        color_mappings = {
            "purple": "#9B59B6",
            "blue": "#3498DB",
            "green": "#2ECC71",
            "orange": "#FF6B35",
            "red": "#E74C3C",
            "yellow": "#F1C40F",
            "pink": "#E91E63",
            "teal": "#00BCD4"
        }
        
        result = {
            "success": True,
            "action": "color_change",
            "color": None,
            "message": ""
        }
        
        for color_name, hex_value in color_mappings.items():
            if color_name in command:
                result["color"] = hex_value
                result["color_name"] = color_name
                result["message"] = f"I'll update the color scheme to {color_name}."
                break
        
        if not result["color"]:
            result["message"] = "What color would you like? Options: purple, blue, green, orange, red, yellow"
        
        return result
    
    def _handle_analysis(self, command: str) -> Dict:
        """Handle analysis requests."""
        result = {
            "success": True,
            "action": "analysis",
            "analysis_type": None,
            "data": None,
            "message": ""
        }
        
        if "efficiency" in command.lower() or "strike rate" in command.lower():
            result["analysis_type"] = "strike_rate"
            result["message"] = "Calculating strike rate and efficiency metrics..."
            
            # Calculate strike rates
            party_col = self._find_column("party")
            if party_col:
                counts = self.df[party_col].value_counts().head(10)
                result["data"] = counts.to_dict()
        
        elif "enp" in command.lower() or "fragmentation" in command.lower():
            result["analysis_type"] = "enp"
            result["message"] = "Calculating Effective Number of Parties..."
        
        else:
            result["analysis_type"] = "general"
            result["message"] = "I'll provide a general analysis of your data."
        
        return result
    
    def _handle_question(self, command: str) -> Dict:
        """Handle question-type requests."""
        question_lower = command.lower()
        result = {
            "success": True,
            "action": "answer",
            "message": ""
        }
        
        if "most" in question_lower or "top" in question_lower or "highest" in question_lower:
            # Find top performers
            if "efficient" in question_lower:
                result["message"] = "Let me analyze efficiency metrics for you."
            elif "competitive" in question_lower:
                result["message"] = "I'll identify the most competitive regions."
            else:
                party_col = self._find_column("party")
                if party_col:
                    top = self.df[party_col].value_counts().head(1)
                    result["message"] = f"The top performer is {top.index[0]} with {top.values[0]} records."
        
        elif "average" in question_lower or "mean" in question_lower:
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                means = self.df[numeric_cols].mean()
                result["message"] = "Average values:\n" + "\n".join([f"- {col}: {val:.2f}" for col, val in means.items()])
        
        else:
            result["message"] = "I can answer questions about your data. Try asking about top performers, averages, or specific metrics."
        
        return result
    
    def _handle_llm_request(self, command: str) -> Dict:
        """Handle complex requests using LLM."""
        if not self.model:
            return {
                "success": False,
                "action": "error",
                "message": "LLM not available. Please configure an API key."
            }
        
        try:
            # Prepare context
            context = SYSTEM_PROMPT.format(
                schema=json.dumps(self.schema.get('mapping', {}), indent=2),
                columns=", ".join(self.df.columns.tolist())
            )
            
            prompt = f"""{context}

User request: {command}

Provide a helpful response. If the user wants to modify visualizations, explain what you would do.
Keep your response concise and actionable."""

            response = self.model.generate_content(prompt)
            
            return {
                "success": True,
                "action": "llm_response",
                "message": response.text
            }
        except Exception as e:
            return {
                "success": False,
                "action": "error",
                "message": f"Error processing request: {str(e)}"
            }
    
    def _find_column(self, role: str) -> Optional[str]:
        """Find column by semantic role or name pattern."""
        mapping = self.schema.get('mapping', {})
        
        # Check schema mapping first
        for col, mapped_role in mapping.items():
            if mapped_role == role:
                return col
        
        # Fallback to name matching
        for col in self.df.columns:
            if role in col.lower():
                return col
        
        return None
    
    def get_suggestions(self) -> list:
        """Get contextual suggestions based on data."""
        suggestions = [
            "Show close fights (margin < 2%)",
            "Filter high turnout states",
            "Calculate strike rate by party",
            "Change chart to pie",
            "Make the colors purple"
        ]
        
        # Add data-specific suggestions
        party_col = self._find_column("party")
        if party_col:
            top_party = self.df[party_col].value_counts().index[0]
            suggestions.append(f"Show data for {top_party}")
        
        state_col = self._find_column("state")
        if state_col:
            suggestions.append(f"Filter by specific state")
        
        return suggestions[:5]
