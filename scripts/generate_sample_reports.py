"""Generate a small corpus of synthetic, cross-domain business reports.

These are entirely fictional but internally consistent, so the assistant has
specific numbers/dates/names to ground its answers on. Domains: Finance, HR,
Operations/Supply chain, Marketing, Sales, Product. One file is written as a PDF
(if `fpdf2` is installed) to exercise the PDF loader; otherwise as markdown.

This script is intentionally dependency-light (stdlib + optional PyYAML) so the
sample data can be created before installing anything else.
"""
from __future__ import annotations

import pathlib

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

ROOT = pathlib.Path(__file__).resolve().parents[1]


def reports_dir() -> pathlib.Path:
    """Resolve the reports directory from config.yaml, defaulting to data/reports."""
    rel = "data/reports"
    cfg_path = ROOT / "config.yaml"
    if yaml is not None and cfg_path.exists():
        try:
            data = yaml.safe_load(cfg_path.read_text()) or {}
            rel = (data.get("paths") or {}).get("reports_dir", rel)
        except Exception:
            pass
    p = pathlib.Path(rel)
    return p if p.is_absolute() else ROOT / p


REPORTS = {
    "2024_Q3_financial_review.md": """# Q3 FY2024 Financial Review — Acme Analytics Inc.

## Headline results
Total revenue for Q3 FY2024 was **$48.2M**, up **12% year over year** from $43.0M in Q3 FY2023.
Gross margin expanded to **61%**, up from 58% a year ago, driven by a higher mix of software subscriptions.
Operating expenses were **$22.4M**. Net income was **$6.1M**, a net margin of **12.7%**.

## Drivers
Growth was led by **Cloud Analytics subscriptions, which grew 24% YoY**, more than offsetting a
**9% decline in legacy on-premise license** revenue. Annual recurring revenue (ARR) reached **$182M**,
and net revenue retention (NRR) was **114%**. A foreign-exchange headwind reduced reported revenue by
approximately **$0.8M**.

## Cash and liquidity
Cash and equivalents ended the quarter at **$34.5M**. Free cash flow was **$5.2M**.

## Outlook
Management guided Q4 FY2024 revenue to a range of **$50M–$52M**, with full-year gross margin around 60–62%.
""",
    "hr_attrition_report_2024.md": """# People & Attrition Report — FY2024

## Headcount
Total headcount reached **1,240**, up from 1,150 at the start of the year.

## Attrition
Trailing-twelve-month **voluntary attrition was 14.2%**, an improvement from 17.5% the prior year.
By function, **Engineering had the highest attrition at 18%**, followed by Sales at 15% and G&A at 8%.

## Hiring
Average **time-to-hire was 38 days** and the offer acceptance rate was 82%.

## Why people leave
The top cited reasons for leaving were **compensation (34%)**, career growth (28%), and management (15%).

## Engagement & diversity
Employee net promoter score (eNPS) improved to **+28** from +19.
Women represent **37% of the workforce** and 29% of engineering.

## Actions taken
The company introduced **quarterly compensation reviews** and a company-wide mentorship program in Q2.
""",
    "supply_chain_operations_review_2024.md": """# Supply Chain & Operations Review — FY2024

## Service levels
**On-time-in-full (OTIF) delivery was 94.3%**, just short of the 95% target.
Average order cycle time improved to **4.2 days** from 5.1 days last year.

## Inventory & cost
Inventory turns were **6.8x** with days inventory outstanding of 53.
**Logistics cost was 8.7% of revenue.** Warehouse capacity utilization averaged 88%.

## Quality & risk
Supplier defect rate was **1.4%**. The most significant risk is a
**single-source dependency for component X, sourced from a supplier in Shenzhen**;
a second-source qualification is in progress.

## Forecasting
Demand **forecast accuracy (MAPE) was 22%**, flagged as the key improvement area.
A new demand-planning tool rollout is planned for Q1 next year.

## Sustainability
Carbon intensity per shipment fell **9%** following a route-optimization program.
""",
    "marketing_campaign_performance_q3_2024.md": """# Marketing Performance — Q3 FY2024

## Spend & pipeline
Total marketing spend was **$3.6M**, generating **$28M of marketing-sourced pipeline**,
**4,200 MQLs** and 980 SQLs. Blended customer acquisition cost (CAC) was **$1,150** with an
11-month payback.

## Channel ROI
**Webinars delivered the best ROI at 5.2x.** Paid search returned 2.1x, while
**events were the weakest at 1.4x**. Email open rate was 24% with a 3.1% click-through rate.

## Flagship campaign
The **"Analytics Reimagined" campaign generated 1,300 MQLs at a $410 cost-per-lead**.
Website conversion was 2.8% and organic traffic grew 18% YoY.

## Recommendation
Reallocate **15% of the events budget to webinars and content**, where marginal ROI is highest.
""",
    "regional_sales_review_2024.txt": """Regional Sales Review - FY2024

Total bookings were $54M across all regions.

Regional split:
- North America: $29M (54% of bookings)
- EMEA: $15M (28%)
- APAC: $7M (13%)
- LATAM: $3M (5%)

Growth:
APAC was the fastest-growing region at +31% year over year. EMEA grew 9%,
North America 7%, and LATAM was roughly flat.

Sales efficiency:
- Win rate: 23%
- Average deal size: $42K
- Average sales cycle: 74 days
- Pipeline coverage: 3.1x

Notable:
The largest single deal was $2.1M with a global retailer in the EMEA region.
Gross logo retention was 91%, with 11 churned logos during the year.
88% of reps finished at or above 80% of quota.
""",
    "product_analytics_review_2024.md": """# Product Analytics Review — FY2024

## Usage
Monthly active users (MAU) reached **86,000**, up **21% YoY**. The DAU/MAU stickiness ratio was 0.34,
and the average session lasted **9.4 minutes**.

## Feature adoption
Dashboards were used by **78%** of accounts, Alerts by 41%, and the new **AI Insights beta by 12%**.

## Retention & satisfaction
**30-day retention was 62%** and 90-day retention was 48%. Product **NPS was 44**.
The top feature request was **scheduled report export, with 1,900 votes**.

## Reliability
There were **two P1 outages totalling 47 minutes** of downtime; the 99.95% availability SLA was still met.
Mobile accounted for 28% of sessions and is growing.
""",
}

EXEC_SUMMARY_TITLE = "FY2024 Executive Summary"
EXEC_SUMMARY_BODY = (
    "Acme Analytics enters FY2025 from a position of strength. The strategic priorities for the "
    "year are: (1) accelerate the shift to cloud subscriptions, (2) improve operational resilience by "
    "removing single-source supplier dependencies, (3) invest in employee retention through clearer "
    "career paths and competitive pay, and (4) double down on the highest-ROI marketing motions. "
    "Leadership views demand forecasting and the AI Insights product as the two biggest opportunities "
    "to compound growth, and customer retention as the foundation underpinning all of them."
)


def _write_exec_summary(out_dir: pathlib.Path) -> str:
    try:
        from fpdf import FPDF  # fpdf2

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, EXEC_SUMMARY_TITLE, ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, EXEC_SUMMARY_BODY)
        out = out_dir / "executive_summary.pdf"
        pdf.output(str(out))
        return out.name
    except Exception:
        out = out_dir / "executive_summary.md"
        out.write_text(f"# {EXEC_SUMMARY_TITLE}\n\n{EXEC_SUMMARY_BODY}\n", encoding="utf-8")
        return out.name


def main() -> None:
    out_dir = reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in REPORTS.items():
        (out_dir / name).write_text(content, encoding="utf-8")
    exec_name = _write_exec_summary(out_dir)
    written = list(REPORTS) + [exec_name]
    print(f"Wrote {len(written)} reports to {out_dir}:")
    for n in written:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
