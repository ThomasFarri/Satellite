# STK cross-check

This check compares this tool with an independent Ansys STK scenario of the same
satellite and station. It has not been run yet. `docs/VALIDATION.md` reports it as
"Not yet run" until the two CSV files below exist.

Both sides use the same TLE and SGP4, so the comparison tests frames, Earth
orientation, the station model and the pass finder, not the orbit model.

## 1. Inputs

- TLE: `validation/stk/satellite.tle` (regenerate with `python validation/make_tle.py`).
  Its epoch is 2027-03-01 00:00:00 UTC (`27060.00000000`).
- Station: latitude 28.30 deg, longitude -16.51 deg, height 2393 m above the WGS84
  ellipsoid (approximate public values for the Tenerife optical ground station).
- Minimum elevation: 20 deg. No Sun constraints (this check is geometric only).

## 2. Build the scenario (STK 12 or later)

1. **New scenario.** Analysis period start `1 Mar 2027 00:00:00.000 UTCG`, stop
   `8 Mar 2027 00:00:00.000 UTCG`. Check that the scenario time unit is UTCG.
2. **Satellite.** Insert a Satellite with "Insert Default", then open Properties >
   Basic > Orbit:
   - Propagator: `SGP4`.
   - Under the element sets, use "Add..." or "Import from file" and select
     `satellite.tle`. Confirm SSC number 99999 and that the element set epoch is
     `1 Mar 2027 00:00:00.000`.
   - Step size 10 s. Start and stop as the scenario.
   - Apply. In the 2D window the ground track should be a near-polar orbit with
     about 15.2 revolutions per day.
3. **Facility.** Insert a Facility, Properties > Basic > Position:
   - Type `Geodetic`, latitude `28.30 deg`, longitude `-16.51 deg`.
   - Altitude `2.393 km` with altitude reference **WGS84 ellipsoid** (not MSL).
   - Untick "Use terrain".
4. **Constraint.** Facility Properties > Constraints > Basic > Elevation Angle:
   tick Min, set `20 deg`. Leave every other constraint off.
5. **Access settings.** Right-click the Facility > Access, pick the Satellite. Before
   computing, open the Access "Advanced" settings and:
   - turn **off** light time delay,
   - set aberration to **None**,
   - keep refraction off on the facility (Properties > Basic > Atmosphere, or
     Constraints, depending on version).
   This tool computes instantaneous geometric positions, so these effects must be
   off on both sides.
6. **Earth orientation.** STK uses its own EOP file. Update it (Utilities > Data
   Update) and note the file date. A different UT1-UTC table moves results by
   up to about 0.01 deg in elevation and 0.2 s in rise/set time, as the Skyfield
   check in `docs/VALIDATION.md` shows.

## 3. Export reports

1. **Access report.** In the Access tool click "Reports > Access", then use the
   report window's export button to save as CSV:
   `validation/stk/data/access.csv`.
   Expected columns: `Access, Start Time (UTCG), Stop Time (UTCG), Duration (sec)`.
   STK's statistics rows at the end (Min, Max, Mean) are skipped by the script.
2. **AER report.** In the Access tool click "Reports > AER". Set the step to 10 s
   (Report & Graph Manager > Report options > Time step), then export as CSV:
   `validation/stk/data/aer.csv`.
   Expected columns: `Time (UTCG), Azimuth (deg), Elevation (deg), Range (km)`.
   Use degrees and km (STK defaults).

`validation/stk/data/` is ignored by Git. Commit the CSVs only if you are happy for
them to be public.

## 4. Compare

```bash
python validation/compare_stk.py --access validation/stk/data/access.csv --aer validation/stk/data/aer.csv --out outputs/stk
```

This writes `stk_access_residuals.csv`, `stk_aer_residuals.csv` and
`stk_residuals.png` (this tool minus STK). Then regenerate the report:

```bash
python validation/run_validation.py
```

## 5. What to accept

| Quantity | Accept | Typical cause if exceeded |
|---|---|---|
| Rise and set time | < 1 s | Different EOP data, light time delay on in STK, wrong altitude reference |
| Elevation | < 0.01 deg | UT1-UTC or polar motion differences, aberration on in STK |
| Azimuth | < 0.01 deg above 30 deg elevation | Same as elevation; azimuth is very sensitive near zenith |
| Range | < 100 m | Light time delay, WGS84 vs MSL altitude, TLE rounding if the TLE was retyped |

If differences sit at about 0.2 s and 0.01 deg, compare the UT1-UTC value in STK's
EOP file with astropy's (`docs/validation_results.json`, key `astropy_dut1_s`).
