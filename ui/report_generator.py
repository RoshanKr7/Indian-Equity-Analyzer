"""
PDF report generator using fpdf2 + kaleido.

Creates a professional multi-page report with:
  - Cover page with company info
  - Price charts (exported as PNG via kaleido)
  - Fundamental metrics table
  - Sentiment summary
  - Prediction dashboard table
  - Methodology note and disclaimer
"""

import io
import tempfile
import os
from datetime import datetime

from fpdf import FPDF
import plotly.graph_objects as go

from config.settings import REPORT_TITLE, DISCLAIMER


class ReportGenerator:
    """Generate a downloadable PDF report."""

    def __init__(self):
        self.pdf = FPDF(orientation="P", unit="mm", format="A4")
        self.pdf.set_auto_page_break(auto=True, margin=20)
        self.temp_dir = tempfile.mkdtemp()

    def _save_chart(self, fig: go.Figure, name: str) -> str:
        """Export a Plotly figure to PNG and return the path."""
        path = os.path.join(self.temp_dir, f"{name}.png")
        try:
            fig.update_layout(
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                font_color="#E0E0E0",
            )
            fig.write_image(path, width=1000, height=500, scale=2)
        except Exception:
            return ""
        return path

    def _add_header(self, text: str):
        self.pdf.set_font("Helvetica", "B", 16)
        self.pdf.set_text_color(0, 212, 170)
        self.pdf.cell(0, 12, text, new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(3)
        self.pdf.set_text_color(220, 220, 220)

    def _add_subheader(self, text: str):
        self.pdf.set_font("Helvetica", "B", 12)
        self.pdf.set_text_color(160, 174, 192)
        self.pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.pdf.set_text_color(220, 220, 220)

    def _add_text(self, text: str, size: int = 10):
        self.pdf.set_font("Helvetica", "", size)
        self.pdf.multi_cell(0, 6, text)
        self.pdf.ln(2)

    def generate(
        self,
        company_info: dict,
        fundamentals: dict,
        factor_scores: dict,
        composite_score: float,
        sentiment: dict,
        predictions: dict,
        charts: dict[str, go.Figure] | None = None,
    ) -> bytes:
        """
        Generate the full PDF report.

        Parameters
        ----------
        charts : dict
            Keys like "candlestick", "technical", "rsi", "macd" → Plotly Figure.

        Returns
        -------
        bytes
            PDF file content.
        """
        # ── Cover Page ──
        self.pdf.add_page()
        self.pdf.set_fill_color(14, 17, 23)
        self.pdf.rect(0, 0, 210, 297, "F")

        self.pdf.set_font("Helvetica", "B", 28)
        self.pdf.set_text_color(0, 212, 170)
        self.pdf.ln(40)
        self.pdf.cell(0, 15, REPORT_TITLE, align="C", new_x="LMARGIN", new_y="NEXT")

        self.pdf.ln(10)
        self.pdf.set_font("Helvetica", "B", 22)
        self.pdf.set_text_color(255, 255, 255)
        self.pdf.cell(0, 12, company_info.get("long_name", ""), align="C",
                      new_x="LMARGIN", new_y="NEXT")

        self.pdf.set_font("Helvetica", "", 14)
        self.pdf.set_text_color(160, 174, 192)
        self.pdf.cell(0, 10,
                      f"{company_info.get('symbol', '')}  |  {company_info.get('exchange', '')}  |  {company_info.get('sector', '')}",
                      align="C", new_x="LMARGIN", new_y="NEXT")

        self.pdf.ln(10)
        self.pdf.set_font("Helvetica", "", 11)
        self.pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M IST')}",
                      align="C", new_x="LMARGIN", new_y="NEXT")

        # ── Charts Page ──
        if charts:
            for chart_name, fig in charts.items():
                path = self._save_chart(fig, chart_name)
                if path and os.path.exists(path):
                    self.pdf.add_page()
                    self.pdf.set_fill_color(14, 17, 23)
                    self.pdf.rect(0, 0, 210, 297, "F")
                    self._add_header(chart_name.replace("_", " ").title())
                    self.pdf.image(path, x=10, w=190)

        # ── Fundamentals Page ──
        self.pdf.add_page()
        self.pdf.set_fill_color(14, 17, 23)
        self.pdf.rect(0, 0, 210, 297, "F")
        self._add_header("Fundamental Analysis")
        self._add_text(f"Composite Score: {composite_score*100:.0f}%")

        self.pdf.set_font("Helvetica", "", 10)
        for factor, data in factor_scores.items():
            score_pct = round(data["score"] * 100)
            self._add_text(f"  {factor.replace('_', ' ').title()}: {score_pct}%")

        self._add_subheader("Key Metrics")
        metrics = [
            ("P/E Ratio", fundamentals.get("pe_trailing")),
            ("P/B Ratio", fundamentals.get("pb_ratio")),
            ("ROE", fundamentals.get("roe")),
            ("Debt/Equity", fundamentals.get("debt_to_equity")),
            ("Profit Margin", fundamentals.get("profit_margin")),
            ("Revenue Growth", fundamentals.get("revenue_growth")),
            ("Dividend Yield", fundamentals.get("dividend_yield")),
            ("Beta", fundamentals.get("beta")),
        ]
        for label, val in metrics:
            display = str(val) if val is not None else "N/A"
            self._add_text(f"  {label}: {display}")

        # ── Sentiment Page ──
        self.pdf.add_page()
        self.pdf.set_fill_color(14, 17, 23)
        self.pdf.rect(0, 0, 210, 297, "F")
        self._add_header("News Sentiment")
        self._add_text(f"Overall: {sentiment.get('label', 'N/A')} (Score: {sentiment.get('score', 0):+.3f})")

        for item in sentiment.get("details", [])[:10]:
            self._add_text(f"  [{item.get('sentiment', '?')}] {item.get('title', '')[:80]}")

        # ── Predictions Page ──
        self.pdf.add_page()
        self.pdf.set_fill_color(14, 17, 23)
        self.pdf.rect(0, 0, 210, 297, "F")
        self._add_header("AI Predictions")

        # Table header
        self.pdf.set_font("Helvetica", "B", 10)
        self.pdf.set_fill_color(26, 31, 46)
        self.pdf.cell(30, 8, "Timeframe", border=1, fill=True)
        self.pdf.cell(25, 8, "Signal", border=1, fill=True)
        self.pdf.cell(35, 8, "Pred. Price", border=1, fill=True)
        self.pdf.cell(30, 8, "Return", border=1, fill=True)
        self.pdf.cell(35, 8, "Confidence", border=1, fill=True)
        self.pdf.ln()

        self.pdf.set_font("Helvetica", "", 10)
        for tf_code, result in predictions.items():
            if result.get("gated"):
                continue
            self.pdf.cell(30, 7, result.get("timeframe", tf_code), border=1)
            self.pdf.cell(25, 7, result.get("signal", "N/A"), border=1)
            price = result.get("predicted_price", 0)
            self.pdf.cell(35, 7, f"Rs.{price:,.2f}", border=1)
            ret = result.get("predicted_return", 0) * 100
            self.pdf.cell(30, 7, f"{ret:+.1f}%", border=1)
            conf = result.get("confidence", 0) * 100
            self.pdf.cell(35, 7, f"{conf:.0f}%", border=1)
            self.pdf.ln()

        # ── Disclaimer ──
        self.pdf.ln(10)
        self._add_subheader("Disclaimer")
        self.pdf.set_font("Helvetica", "", 8)
        self.pdf.set_text_color(200, 130, 130)
        self.pdf.multi_cell(0, 5, DISCLAIMER)

        # Output
        return bytes(self.pdf.output())
