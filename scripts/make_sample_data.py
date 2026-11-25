"""Generate a synthetic retail-sales dataset for demos.

Run:  python scripts/make_sample_data.py
All data is fake and randomly generated (fixed seed, so it's reproducible).
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1_200

products = {
    "Laptop": ("Electronics", 950), "Monitor": ("Electronics", 240), "Headphones": ("Electronics", 85),
    "Desk Chair": ("Furniture", 180), "Standing Desk": ("Furniture", 420), "Bookshelf": ("Furniture", 130),
    "Notebook Pack": ("Office Supplies", 12), "Pen Set": ("Office Supplies", 8), "Printer Paper": ("Office Supplies", 25),
}
regions = ["North", "South", "East", "West"]
segments = ["Consumer", "Small Business", "Enterprise"]

names = rng.choice(list(products), N)
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
region = rng.choice(regions, N, p=[0.3, 0.2, 0.3, 0.2])
units = rng.integers(1, 20, N)
base_price = np.array([products[p][1] for p in names])
price = np.round(base_price * rng.uniform(0.9, 1.1, N), 2)
discount = rng.choice([0, 0, 0, 0.05, 0.1, 0.15], N)
revenue = np.round(units * price * (1 - discount), 2)
cost = np.round(revenue * rng.uniform(0.55, 0.8, N), 2)

df = pd.DataFrame({
    "Order Date": dates.strftime("%Y-%m-%d"),
    "Order ID": [f"ORD-{10000 + i}" for i in range(N)],
    "Region": region,
    "Customer Segment": rng.choice(segments, N, p=[0.5, 0.3, 0.2]),
    "Product": names,
    "Category": [products[p][0] for p in names],
    "Units": units,
    "Unit Price": [f"${p:,.2f}" for p in price],   # stored as text on purpose, to show cleaning
    "Discount": [f"{d:.0%}" for d in discount],     # e.g. "10%"
    "Revenue": revenue,
    "Cost": cost,
}).sort_values("Order Date")

# A few realistic imperfections
df.loc[df.sample(15, random_state=1).index, "Customer Segment"] = np.nan
df.loc[df.sample(8, random_state=2).index, "Cost"] = np.nan

out = Path(__file__).resolve().parent.parent / "sample_data" / "sales_sample.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df):,} rows to {out}")


# ---------------------------------------------------------------- second sample: support tickets
# Fake customer-support data: no money columns, a partly covered last month and
# satisfaction scores that are often missing (not every customer answers the survey).

rng = np.random.default_rng(7)
N = 1_500
areas = {"Billing": 1.0, "Login & Access": 0.6, "Mobile App": 1.4, "Payments": 1.2, "Reporting": 0.9,
         "Integrations": 1.8}
channels = ["Email", "Phone", "Chat", "Web form"]
priorities = ["Low", "Medium", "High", "Urgent"]

opened = pd.to_datetime("2025-03-01") + pd.to_timedelta(rng.integers(0, 229 * 24 * 60, N), unit="min")
area = rng.choice(list(areas), N, p=[0.2, 0.18, 0.2, 0.17, 0.13, 0.12])
priority = rng.choice(priorities, N, p=[0.35, 0.4, 0.18, 0.07])
channel = rng.choice(channels, N, p=[0.38, 0.17, 0.33, 0.12])
team = np.where(np.isin(priority, ["High", "Urgent"]), rng.choice(["Tier 2", "Specialists"], N),
                rng.choice(["Tier 1", "Tier 2"], N, p=[0.75, 0.25]))
complexity = np.array([areas[a] for a in area])
resolution = np.round(rng.gamma(2.0, 6.0, N) * complexity * np.where(channel == "Chat", 0.7, 1.0), 1)
first_response = np.round(rng.gamma(2.0, 20.0, N) * np.where(channel == "Phone", 0.3, 1.0))
satisfaction = np.clip(np.round(5.2 - resolution / 18 + rng.normal(0, 0.8, N)), 1, 5)
satisfaction = np.where(rng.random(N) < 0.27, np.nan, satisfaction)

tickets = pd.DataFrame({
    "Ticket ID": [f"TCK-{50000 + i}" for i in range(N)],
    "Opened": opened.strftime("%Y-%m-%d %H:%M"),
    "Channel": channel,
    "Priority": priority,
    "Product Area": area,
    "Team": team,
    "First Response Minutes": first_response,
    "Resolution Hours": resolution,
    "Satisfaction Score": satisfaction,
    "Reopened": np.where(rng.random(N) < 0.08 * complexity, "Yes", "No"),
}).sort_values("Opened")

out2 = out.parent / "support_tickets.csv"
tickets.to_csv(out2, index=False)
print(f"Wrote {len(tickets):,} rows to {out2}")
