# Data licences

This repository **does not redistribute any data**. `scripts/download_data.py` fetches ETTh1/ETTh2
from their public source and checks the SHA-256 of every file. The electricity and traffic files
must be placed by hand (see `docs/PROVENANCE.md`). Licences were read from the sources on
2026-11-16:

| dataset | used as | licence (as stated by the source) | source |
|---|---|---|---|
| ETTh1, ETTh2 (ETDataset) | `data/raw/ETTh{1,2}.csv` | Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0), stated in the repository's LICENSE file | https://github.com/zhouhaoyi/ETDataset (LICENSE: https://github.com/zhouhaoyi/ETDataset/blob/main/LICENSE) |
| ElectricityLoadDiagrams20112014 (UCI) | basis of `data/raw/electricity.csv` (370 clients, resampled hourly) | CC BY 4.0: "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license." | https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 |
| Traffic (Caltrans PeMS road occupancy) | `data/raw/traffic.csv` (862 sensors, hourly) | **No licence named.** Caltrans PeMS Conditions of Use: "In general, information presented on this web site, unless otherwise indicated, is considered in the public domain." PeMS requires an account to access the site. | https://pems.dot.ca.gov/ (Conditions of Use: https://pems.dot.ca.gov/?view=tou) |

Notes:
- The UCI "PEMS-SF" dataset is CC BY 4.0 (https://archive.ics.uci.edu/dataset/204/pems+sf), but
  it is a different packaging (963 sensors, 10-minute samples) from the hourly 862-sensor file
  used here. Its licence statement therefore does not settle the terms of the file we used.
  Reuse of the underlying PeMS data is governed by Caltrans' conditions above.
- ETDataset is CC BY-ND 4.0. We distribute no copies or derivatives of the files; the download
  script fetches the originals unchanged.
