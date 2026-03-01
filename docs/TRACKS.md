# Track Inventory

This file summarizes the `.mat` tracks available in `maps/`.

Columns:
- `Length` is the stored track length in meters.
- `Width` is derived from `track_width_m` (min/max/mean).
- `Min radius` is `1/max(|kappa|)` from `psi_s_radpm` (approx).
- `Obstacles` indicates whether obstacle arrays exist in the file.

| Track | Length (m) | Width min (m) | Width max (m) | Width mean (m) | Min radius (m) | Obstacles | N obs |
|---|---:|---:|---:|---:|---:|---|---:|
| `MAP1.mat` | 300.00 | 8.67 | 13.62 | 10.00 | 2.26 | no | 0 |
| `Medium_Oval_Map_260m.mat` | 260.00 | 6.00 | 6.00 | 6.00 | 18.00 | no | 0 |
| `Medium_Oval_Map_260m_Obstacles.mat` | 260.00 | 6.00 | 6.00 | 6.00 | 18.00 | yes | 5 |
