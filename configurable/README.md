# Railway route data JSON conversion

## Files
- `speed_restrictions.json`: all 123 speed restriction records from `Speed_Restrictions.xlsx`.
- `gradients.json`: all 629 gradient records from `Gradients.xlsx`, converted to signed per-mille values.
- `curves.json`: all 384 curve records from `Curves.xlsx`.
- `config.json`: combined configuration with the three route arrays inserted into the HWH-NDLS optimiser config, limited to the timetable endpoint at km 1447.
- `conversion_summary.json`: counts, validation results, and conversion assumptions.

## Field formats
Speed restrictions:
`{"from_km": 0.0, "to_km": 0.8, "speed_kmh": 10}`

Gradients:
`{"from_km": 21.4, "to_km": 22.5, "gradient_permille": -1}`

Curves:
`{"from_km": 0.3, "to_km": 0.37, "degree": 5.7}`

## Gradient conversion
The source sheet provides a direction code and a denominator. The conversion used is:
- direction `1`: positive (uphill)
- direction `-1`: negative (downhill)
- direction `0`: level, giving `0` per mille
- otherwise, signed gradient = `direction * 1000 / denominator`

This follows the source column heading. Please confirm that `1` is intended to mean uphill for the direction of travel.

## Route endpoint note
The timetable destination is New Delhi at km 1447, but some source records extend beyond that point to km 1459. The individual JSON files preserve those complete records. The combined optimiser `config.json` clips intersecting records at km 1447 and omits records beginning at or beyond km 1447, so its segment ranges remain within the route model.

## Data quality
- No blank cells were found in the relevant source columns.
- No reversed chainage ranges or duplicate complete rows were found.
- No overlaps were found within each of the three datasets when sorted by start chainage.
- Gaps are present in speed restriction and curve records; these were not filled or inferred.
- The curve-radius model and curve-speed calculation remain governed by the existing Python optimiser configuration and should be verified against the railway's applicable engineering standard.
