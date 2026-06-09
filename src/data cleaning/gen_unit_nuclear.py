"""
Fetch actual nuclear unit generation from the RTE Open Data API for representative days.
Outputs production_nucleaire.csv to data/final/.
"""

import requests, base64, time, pandas as pd

CLIENT_ID     = "a8ecdc57-12c0-4a28-9df9-04dbf26bd266"
CLIENT_SECRET = "e3bc47d8-5587-4a40-98da-78f9833d4bd7"

# 1. Token
creds = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
token = requests.post(
    "https://digital.iservices.rte-france.com/token/oauth/",
    headers={"Authorization": f"Basic {creds}",
             "Content-Type": "application/x-www-form-urlencoded"}
).json()["access_token"]

headers = {"Authorization": f"Bearer {token}"}

units = {
    "BUGEY-2":      "17W100P100P00966",
    "BUGEY-3":      "17W100P100P00974",
    "BUGEY-4":      "17W100P100P00982",
    "BUGEY-5":      "17W100P100P01004",
    "GRAVELINES-1": "17W100P100P0130W",
    "GRAVELINES-2": "17W100P100P0131U",
    "GRAVELINES-3": "17W100P100P0132S",
    "DAMPIERRE-1":  "17W100P100P0120Z",
    "DAMPIERRE-2":  "17W100P100P0121X",
    "TRICASTIN-1":  "17W100P100P0149B",
    "TRICASTIN-2":  "17W100P100P0150Q",
}

dates = [
    ("2024-01-18T00:00:00+01:00", "2024-01-19T00:00:00+01:00"),  # winter CET
    ("2024-02-04T00:00:00+01:00", "2024-02-05T00:00:00+01:00"),  # winter CET
    ("2024-03-12T00:00:00+01:00", "2024-03-13T00:00:00+01:00"),  # winter CET (before DST switch on 31/03)
    ("2024-06-12T00:00:00+02:00", "2024-06-13T00:00:00+02:00"),  # summer CEST
    ("2024-06-22T00:00:00+02:00", "2024-06-23T00:00:00+02:00"),  # summer CEST
    ("2024-08-13T00:00:00+02:00", "2024-08-14T00:00:00+02:00"),  # summer CEST
    ("2024-09-17T00:00:00+02:00", "2024-09-18T00:00:00+02:00"),  # summer CEST
    ("2024-10-04T00:00:00+02:00", "2024-10-05T00:00:00+02:00"),  # summer CEST (before DST switch on 27/10)
    ("2024-11-04T00:00:00+01:00", "2024-11-05T00:00:00+01:00"),  # winter CET
    ("2024-12-02T00:00:00+01:00", "2024-12-03T00:00:00+01:00"),  # winter CET
]

BASE_URL = "https://digital.iservices.rte-france.com/open_api/actual_generation/v1/actual_generations_per_unit"

all_rows = []
for start, end in dates:
    for unit_name, eic in units.items():
        r = requests.get(BASE_URL, headers=headers, params={
            "start_date": start, "end_date": end, "unit_eic_code": eic
        })
        for unit in r.json().get("actual_generations_per_unit", []):
            for val in unit.get("values", []):
                all_rows.append({
                    "unit": unit_name,
                    "start_date": val["start_date"],
                    "value_MW": val["value"],
                })
        time.sleep(1)

df = pd.DataFrame(all_rows)
df.to_csv("production_nucleaire.csv", index=False)
print(f"✅ {len(df)} rows exported")
print(df.head())