# Track Inventory

This file summarizes the `.mat` tracks available in `maps/`.

Columns:
- `Length` is the stored track length in meters.
- `Width` is derived from `track_width_m` (min/max/mean).
- `Min radius` is `1/max(|kappa|)` from `psi_s_radpm` (approx).
- `Obstacles` indicates whether obstacle arrays exist in the file.

| Track | Length (m) | Width min (m) | Width max (m) | Width mean (m) | Min radius (m) | Obstacles | N obs |
|---|---:|---:|---:|---:|---:|---|---:|
| `Oval_Track_260m.mat` | 260.00 | 6.00 | 6.00 | 6.00 | 18.00 | no | 0 |
| `Oval_Track_260m_Obstacles.mat` | 260.00 | 6.00 | 6.00 | 6.00 | 18.00 | yes | 5 |
| `TRACK1_280m.mat` | 280.00 | 6.00 | 6.00 | 6.00 | 12.00 | no | 0 |
| `TRACK1_280m_Obstacles.mat` | 280.00 | 6.00 | 6.00 | 6.00 | 12.00 | yes | 5 |
| `TRACK2_280m.mat` | 280.00 | 6.00 | 6.00 | 6.00 | 3.89 | no | 0 |
| `TRACK2_280m_Obstacles.mat` | 280.00 | 6.00 | 6.00 | 6.00 | 3.89 | yes | 5 |
| `TRACK3_300m.mat` | 300.00 | 6.00 | 6.00 | 6.00 | 3.21 | no | 0 |
| `TRACK3_300m_Obstacles.mat` | 300.00 | 6.00 | 6.00 | 6.00 | 3.21 | yes | 5 |
| `TRACK4_330m.mat` | 330.00 | 6.00 | 6.00 | 6.00 | 6.53 | no | 0 |
| `TRACK4_330m_Obstacles.mat` | 330.00 | 6.00 | 6.00 | 6.00 | 6.53 | yes | 5 |
| `TRACK5_350m.mat` | 350.00 | 8.67 | 13.62 | 10.00 | 2.64 | no | 0 |
| `TRACK5_350m_Obstacles.mat` | 350.00 | 8.67 | 13.62 | 10.00 | 2.64 | yes | 6 |
