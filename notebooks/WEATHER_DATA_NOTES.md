# Weather Data - Notes

Source: **Open-Meteo Historical Weather API**
Docs: https://open-meteo.com/en/docs/historical-weather-api

\---

## Why this source

I needed four things: hourly history, **solar radiation**, forecasts in the same format,
and free access. My AEMO analysis showed midday demand is driven by rooftop solar, so
radiation is not optional.

|Source|History|Forecast|Solar radiation|Resolution|Key needed|
|-|-|-|-|-|-|
|**Open-Meteo**|1940+|Yes same format|Yes|9–25 km|No|
|NASA POWER|1984+|none|Yes|50–100 km|No|
|BOM (official AU)|Limited|Yes|Partial|—|Awkward access|
|OpenWeatherMap|paid|Yes|Yes|\~10 km|Yes|
|Visual Crossing|Yes|Yes|Yes|\~10 km|Yes, 1k/day free|

**Why not NASA POWER**, which was the closest rival:

1. **No forecast at all.** I need history to train *and* forecasts to predict tomorrow.
POWER is historical only, so I would need a second source anyway.
2. **Too coarse.** Solar is a 1°×1° grid (\~100 km). Brisbane and Toowoomba would get
the same weather. Open-Meteo is 9–25 km.
3. **Data changes retroactively.** POWER replaces near-real-time values with revised
ones 2–3 months later. My training data would silently change under me.
4. **Known radiation delays** since August 2024 due to a CERES processing issue.

**Licence:** CC BY 4.0, free for non-commercial use up to 10,000 calls/day, no signup.
Server code is open source (AGPLv3). Cite: Zippenfenig, P. (2023), DOI 10.5281/ZENODO.7970649

\---

## The endpoint

```
https://archive-api.open-meteo.com/v1/archive
```

|Parameter|What it does|
|-|-|
|`latitude`, `longitude`|Location. Brisbane: -27.4698, 153.0251|
|`start\_date`, `end\_date`|YYYY-MM-DD|
|`hourly`|Comma-separated variable list|
|`timezone`|**Default is GMT.** Omit it and you get UTC - which is what I want|

\---

## My five variables

|Variable|Unit|Valid time|Why|
|-|-|-|-|
|`temperature\_2m`|°C|**Instant**|Heating below \~18°C, cooling above \~24°C|
|`apparent\_temperature`|°C|**Instant**|"Feels like" - humidity changes aircon use|
|`relative\_humidity\_2m`|%|**Instant**|Drives the above|
|`cloud\_cover`|%|**Instant**|Total cloud, all altitudes|
|`shortwave\_radiation`|W/m²|**Preceding hour mean**|Actual sunlight hitting the ground|

\---

## Three traps

**1. Mixed time conventions in the same response.**

`shortwave\_radiation` is the **average over the hour before** the timestamp.
Everything else is the value **at** that timestamp.

So at `07:00`, radiation describes 06:00–07:00, but temperature describes 07:00 exactly.

I confirmed this from the data before finding it in the docs: at 06:00 radiation was 0
and at 07:00 it was 10 W/m². Brisbane sunrise in early August is \~6:25am, so a value
*at* 7am would be far higher than 10. A backward average over a mostly-dark hour fits.

There are `\_instant` versions if I ever want instantaneous radiation instead.

**2. Total cloud cover includes high cloud that barely blocks sun.**

Aug 6 showed 77% midday cloud but 653 W/m² peak radiation - near clear-sky.
`cloud\_cover` counts all altitudes, and high cirrus above 6 km lets most light through.

**Conclusion: `shortwave\_radiation` is the better feature.** It measures the thing that
actually matters. If I want cloud detail, use `cloud\_cover\_low` / `\_mid` / `\_high`
separately rather than the total.

**3. Your coordinates get moved.**

I asked for `-27.4698, 153.0251` and got back `-27.45167, 153.02014` - about 2 km north.
The API snaps to its grid cell and returns the cell centre. Fine for a whole-city
forecast, but it is not the exact point I asked for.

\---

## Response structure

**Parallel lists, not rows.** Unlike a CSV, each variable is its own array, aligned by
position: `time\[0]` goes with `temperature\_2m\[0]`.

```
"hourly": {
  "time":           \["2026-08-01T00:00", "2026-08-01T01:00", ...],
  "temperature\_2m": \[10.5, 10.2, ...],
  "cloud\_cover":    \[4, 26, ...]
}
```

Units are in a separate `hourly\_units` object. The timezone is stated explicitly in
`timezone`, `timezone\_abbreviation` and `utc\_offset\_seconds` - AEMO gave me none of that.

\---

## My findings (sample: Brisbane, 1–7 Aug 2026)

|||
|-|-|
|Rows|168 = 7 full days, hourly|
|Null values|0|
|Timezone returned|`Australia/Brisbane`, `GMT+10`, offset 36000 s|
|Elevation|18 m|
|Cloud vs radiation|Consistent on 6 of 7 days. Aug 6 was the exception (see trap 2)|

\---

## Decisions for my code

1. **Request UTC** - omit the `timezone` parameter. No conversion needed
2. **Use hourly resolution.** Average AEMO's 5-minute data down to hourly, because
weather only exists hourly and stretching it would invent precision
3. **Use `shortwave\_radiation` as the main solar feature**, not `cloud\_cover`
4. **Note in code that radiation is a backward average** and everything else is instant
5. Store raw JSON responses unchanged in `data/raw/`
6. Add attribution to the README (CC BY 4.0 requires it)

\---

## Still to check

* Does the **forecast** endpoint use the same conventions as the archive?
* The archive has a **5-day delay** on ERA5. Which model does "best match" use for
recent days, and is it consistent with older data?

