import numpy as np
import pandas as pd
import holidays as pyholidays

US_HOLIDAYS = pyholidays.UnitedStates()

def simulate_weather(ts: pd.Series) -> pd.DataFrame:
    t = pd.to_datetime(ts, errors="coerce")
    month = t.dt.month
    temp = np.zeros(len(t)); cond = np.empty(len(t), dtype=object)
    for i, m in enumerate(month):
        if m in [12, 1, 2]:
            base = np.random.normal(3, 5)
            cond[i] = np.random.choice(["Clear","Cloudy","Snow","Rain"], p=[0.35,0.35,0.15,0.15])
        elif m in [3, 4, 5]:
            base = np.random.normal(15, 5)
            cond[i] = np.random.choice(["Clear","Cloudy","Rain"], p=[0.45,0.35,0.20])
        elif m in [6, 7, 8]:
            base = np.random.normal(28, 4)
            cond[i] = np.random.choice(["Clear","Cloudy","Rain"], p=[0.55,0.25,0.20])
        else:
            base = np.random.normal(16, 5)
            cond[i] = np.random.choice(["Clear","Cloudy","Rain"], p=[0.45,0.35,0.20])
        temp[i] = base + np.random.normal(0, 1.2)
    dfw = pd.DataFrame({"temperature_c": np.round(temp, 1), "weather": cond})
    dfw["precip"] = dfw["weather"].isin(["Rain","Snow"]).astype(int)
    return dfw

def holiday_flag(ts: pd.Series) -> pd.Series:
    d = pd.to_datetime(ts, errors="coerce").dt.date
    return d.apply(lambda x: 1 if x in US_HOLIDAYS else 0)

def simulate_traffic(hour: pd.Series, dow: pd.Series, is_holiday: pd.Series) -> pd.Series:
    res = []
    for h, d, hol in zip(hour, dow, is_holiday):
        if hol == 1: res.append("Light"); continue
        if d < 5:
            if 7 <= h <= 9 or 16 <= h <= 19: res.append("Heavy")
            elif 11 <= h <= 14: res.append("Medium")
            else: res.append("Light")
        else:
            if 12 <= h <= 15: res.append("Medium")
            else: res.append("Light")
    return pd.Series(res, index=hour.index)