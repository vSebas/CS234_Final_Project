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
| `Oval_Track_260m_Obstacles.mat` | 260.00 | 6.00 | 6.00 | 6.00 | 18.00 | yes | 4 |
| `TRACK1_280m.mat` | 280.00 | 6.00 | 6.00 | 6.00 | 12.00 | no | 0 |
| `TRACK1_280m_Obstacles.mat` | 280.00 | 6.00 | 6.00 | 6.00 | 12.00 | yes | 4 |
| `TRACK2.mat` | 248.52 | 6.00 | 6.00 | 6.00 | 8.14 | no | 0 |
| `TRACK3_300m.mat` | 300.00 | 6.00 | 6.00 | 6.00 | 3.21 | no | 0 |
| `TRACK3_300m_Obstacles.mat` | 300.00 | 6.00 | 6.00 | 6.00 | 3.21 | yes | 4 |
| `TRACK4_315m.mat` | 315.00 | 6.00 | 6.00 | 6.00 | 6.23 | no | 0 |
| `TRACK4_315m_Obstacles.mat` | 315.00 | 6.00 | 6.00 | 6.00 | 6.23 | yes | 4 |
| `TRACK5_330m.mat` | 332.60 | 6.00 | 6.00 | 6.00 | 5.47 | no | 0 |
| `TRACK5_330m_Obstacles.mat` | 332.60 | 6.00 | 6.00 | 6.00 | 5.47 | yes | 4 |
