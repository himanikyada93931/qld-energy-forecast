# AEMO Data — Notes

File: `PRICE_AND_DEMAND_202608_QLD1.csv`
Source: https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data

---

## Columns

| Column | Meaning | Type | Unit |
|---|---|---|---|
| `REGION` | State code. Always `QLD1` in this file | text | — |
| `SETTLEMENTDATE` | Time of the interval | text | — |
| `TOTALDEMAND` | Power used from the grid | float | MW |
| `RRP` | Wholesale price | float | AUD/MWh |
| `PERIODTYPE` | Always `TRADE` in this file | text | — |

Filename pattern: `PRICE_AND_DEMAND_YYYYMM_REGION.csv` — URLs can be built in code.

---

## Four traps

**1. Interval is 5 minutes, not 30.**
288 rows per day. The NEM moved to 5-minute settlement in October 2021.
Older tutorials say 30 minutes. They are out of date.

**2. The time is the END of the interval.**
`00:05` = the 5 minutes from 00:00 to 00:05.
The first row of the day is `00:05`, not `00:00`.

**3. AEMO time is UTC+10. Always. No daylight saving.**
Queensland has no daylight saving either, so for now they match.

```python
df["SETTLEMENTDATE"] = (
    pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
      .dt.tz_localize("Etc/GMT-10")
      .dt.tz_convert("UTC")
)
```

**4. Date format is `YYYY/MM/DD` with slashes.**
Always pass `format=` explicitly. Letting pandas guess is slow and can silently
swap day and month.

---

## My findings

| | |
|---|---|
| Rows | 1,728 |
| Date range | 2026-08-01 00:05 → 2026-08-07 00:00 (6 days) |
| Interval | 5 minutes, perfectly consistent |
| Duplicate timestamps | 0 |
| Missing rows | 0 (1,728 expected, 1,728 found) |
| Null values | 0 in every column |
| Demand range | 3,807 – 7,866 MW (mean 5,935) |
| Price range | −25.01 – 294.40 AUD/MWh (mean 67.12) |
| Negative prices | 173 rows = 10% of the file |
| `REGION` varies? | No — always `QLD1`. **Drop it** |
| `PERIODTYPE` varies? | No — always `TRADE`. **Drop it** |

The file is the current month, so it is partial — it stops at the 7th.
A full month is about 8,928 rows. Two years is about 210,000 rows.

---

## The important finding: demand is lowest at midday

| | |
|---|---|
| Highest demand | 6pm — 7,587 MW |
| **Lowest demand** | **11am — 4,336 MW** |
| Cheapest price | 10am — $13.25 (vs $67 average) |

This looks wrong at first. Surely 3am should be the quietest?

It is rooftop solar. `TOTALDEMAND` measures power taken **from the grid**. At midday,
Queensland's rooftop panels are supplying homes directly, so the grid sees less
demand than it does overnight. Prices collapse at the same time for the same reason.

This shape is called the **duck curve**.

**Two consequences for the model:**

- Sunshine matters as much as temperature. Cloud cover is a real feature to fetch.
- Do not assume demand rises with heat in a simple line. Midday is hot *and* low-demand.

**Timezone confirmed:** the 6pm peak is exactly where human behaviour puts it.
If the peak had landed at 3am, the timezone would be wrong.

---

## Decisions for my code

1. Store time as UTC, converted from a fixed +10 offset
2. Parse with `format="%Y/%m/%d %H:%M:%S"` — never guess
3. Comment that the time is the interval END
4. Unique key = timestamp (region is constant, but keep it if I ever add states)
5. Drop `PERIODTYPE` — no information
6. Keep `RRP` — free, and useful for a second project
7. Expect ~288 rows per day when checking for gaps
8. Raw CSVs stay unchanged in `data/raw/`
