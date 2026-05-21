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
    ("2024-01-29T00:00:00+01:00", "2024-01-30T00:00:00+01:00"),
    ("2024-02-14T00:00:00+01:00", "2024-02-15T00:00:00+01:00"),
    ("2024-03-14T00:00:00+01:00", "2024-03-15T00:00:00+01:00"),
    ("2024-04-17T00:00:00+02:00", "2024-04-18T00:00:00+02:00"),
    ("2024-05-13T00:00:00+02:00", "2024-05-14T00:00:00+02:00"),
    ("2024-06-11T00:00:00+02:00", "2024-06-12T00:00:00+02:00"),
    ("2024-07-10T00:00:00+02:00", "2024-07-11T00:00:00+02:00"),
    ("2024-08-20T00:00:00+02:00", "2024-08-21T00:00:00+02:00"),
    ("2024-09-12T00:00:00+02:00", "2024-09-13T00:00:00+02:00"),
    ("2024-10-02T00:00:00+02:00", "2024-10-03T00:00:00+02:00"),
    ("2024-11-08T00:00:00+01:00", "2024-11-09T00:00:00+01:00"),
    ("2024-12-20T00:00:00+01:00", "2024-12-21T00:00:00+01:00"),
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
print(f"✅ {len(df)} lignes exportées")
print(df.head())