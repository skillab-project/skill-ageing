from fastapi import FastAPI, APIRouter, Query
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
import requests
import os
import json
import math
import time
import re
import datetime

# ============================================================
#  Load env once at module level
# ============================================================
load_dotenv()

API      = os.getenv("TRACKER_API")
USERNAME = os.getenv("TRACKER_USERNAME")
PASSWORD = os.getenv("TRACKER_PASSWORD")
KU_API   = os.getenv("KU_API_URL", "").rstrip("/")

router = APIRouter(prefix="/forecast", tags=["SKILL Forecast"])


# ============================================================
#  Shared helpers
# ============================================================

def _get_token() -> str:
    """Authenticate and return a fresh Bearer token."""
    res = requests.post(
        f"{API}/login",
        json={"username": USERNAME, "password": PASSWORD},
        timeout=15
    )
    res.raise_for_status()
    return res.text.replace('"', "")


def _ensure_cache(folder: str = "Completed_Analyses") -> Path:
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_cache(file_path: Path):
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            
            if isinstance(data, dict) and data.get("status") == "processing":
                return {
                    "status": "processing", 
                    "message": "Forecast is currently being calculated. Please wait.",
                    "started_at": data.get("start_time")
                }

            print(f"✅ Cache hit — loaded from '{file_path}'.")
            return data
        except (json.JSONDecodeError, ValueError):
            file_path.unlink(missing_ok=True)
    return None

def _start_processing(file_path: Path):
    placeholder = {
        "status": "processing",
        "start_time": datetime.datetime.now().isoformat()
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(placeholder, f, indent=4)

def _save_cache(file_path: Path, result: dict):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"💾 Cached → '{file_path}'.")


def _resolve_skills(headers: dict) -> dict:
    """
    Fetch all ESCO skills and return {uri: label} map.
    Auto-paginates using count field.
    """
    print("🔗 Resolving ESCO skill labels (auto-paginated)...")
    page_size = 100

    probe = requests.post(
        f"{API}/skills?page=1&page_size={page_size}",
        headers=headers, timeout=60
    )
    probe.raise_for_status()
    probe_data  = probe.json()
    total_count = probe_data.get("count", 0)
    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
    print(f"📊 ESCO skills: {total_count} total → {total_pages} page(s)")

    esco_items = list(probe_data.get("items", []))

    for page in range(2, total_pages + 1):
        try:
            r = requests.post(
                f"{API}/skills?page={page}&page_size={page_size}",
                headers=headers, timeout=60
            )
            items = r.json().get("items", [])
            if not items:
                break
            esco_items.extend(items)
            if len(items) < page_size:
                break
        except Exception as e:
            print(f"⚠️ Skills page {page} failed: {e}")
            break

    mapping = {x["id"]: x.get("label", x["id"]).lower() for x in esco_items}
    print(f"✅ Resolved {len(mapping)} ESCO skill labels.")
    return mapping


def _forecast_series(series, horizon: int) -> tuple:
    """
    ARIMA → Trend → Moving Average fallback.
    Returns (predictions_list, method_str).
    """
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings("ignore")

    last_val = float(series.iloc[-1])
    last3    = series.tail(3)

    # Trend
    trend_pred = None
    if last3.nunique() > 1:
        trend      = last3.diff().mean()
        trend_pred = [max(0, last_val + (i + 1) * trend) for i in range(horizon)]

    # Moving Average
    ma_pred = [float(last3.mean())] * horizon

    # ARIMA
    arima_pred = None
    if len(series) >= 5:
        try:
            fit        = ARIMA(series, order=(1, 1, 1)).fit()
            raw        = fit.forecast(steps=horizon)
            arima_pred = [max(0, float(x)) for x in raw]
        except Exception as e:
            print(f"   ⚠️ ARIMA failed: {e}")

    # Reject flat ARIMA
    use_arima = (
        arima_pred is not None
        and max(arima_pred) - min(arima_pred) >= 0.3
    )

    if use_arima:
        return arima_pred, "arima"
    elif trend_pred is not None:
        return trend_pred, "trend"
    else:
        return ma_pred, "moving_average"


def _normalize_shares(results: dict):
    """Add 'share' field in-place to every prediction entry."""
    month_totals = {}
    for d in results.values():
        for p in d["prediction"]:
            month_totals[p["date"]] = month_totals.get(p["date"], 0) + p["absolute"]
    for d in results.values():
        for p in d["prediction"]:
            total    = month_totals[p["date"]]
            p["share"] = round(p["absolute"] / total, 6) if total > 0 else 0


# ============================================================
#  ENDPOINT 1: /forecast/ku_forecast_arima
# ============================================================

@router.get("/ku_forecast_arima")
def ku_forecast(
    horizon:      int = Query(6,    description="Forecast horizon in months (3, 6, or 12)"),
    start_date:   str = Query(None, description="Start date YYYY-MM"),
    end_date:     str = Query(None, description="End date YYYY-MM"),
    organization: str = Query(None, description="Filter by organization")
):
    """
    Forecast KU (Knowledge Unit) activity using ARIMA → Trend → Moving Average.
    Results cached in Completed_Analyses/.
    """
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")

    # === 1️⃣ Fetch KU records ===
    ku_url = f"{KU_API}/analysis_results"
    params = {}
    if start_date:   params["start_date"]   = start_date
    if end_date:     params["end_date"]     = end_date
    if organization: params["organization"] = organization

    print(f"🔍 Fetching KU records from {ku_url} with filters {params}")
    res     = requests.get(ku_url, params=params, timeout=60)
    res.raise_for_status()
    ku_data = res.json()

    if not ku_data:
        return {"error": "No KU data found for the selected filters."}

    print(f"📄 Retrieved {len(ku_data)} KU analysis records.")

    # === 2️⃣ Build monthly KU counts ===
    from datetime import datetime
    records = []
    for rec in ku_data:
        detected  = rec.get("detected_kus", {})
        timestamp = rec.get("timestamp")
        try:
            month = datetime.fromisoformat(timestamp).strftime("%Y-%m")
        except Exception:
            continue
        for ku, v in detected.items():
            if str(v) == "1":
                records.append({"date": month, "ku": ku})

    df = pd.DataFrame(records)
    if df.empty:
        return {"error": "No KU detections found after filtering."}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    print(f"📊 Raw KU records: {len(df)} | Unique KUs: {df['ku'].nunique()}")

    ts = df.pivot_table(index="date", columns="ku", aggfunc="size").fillna(0)
    print(f"📊 Timeline shape: {ts.shape}")

    if ts.shape[1] == 0:
        return {"error": "Pivot produced no KU columns."}

    # === 3️⃣ Forecast ===
    results = {}
    skipped = []

    for ku in ts.columns:
        series = ts[ku]
        if series.sum() < 2:
            skipped.append(ku); continue
        if (series.tail(4) > 0).sum() < 1:
            skipped.append(ku); continue

        final, method = _forecast_series(series, horizon)

        if max(final) < 0.3:
            skipped.append(ku); continue

        future_dates = pd.date_range(
            ts.index[-1] + pd.offsets.MonthBegin(1),
            periods=horizon, freq="MS"
        ).strftime("%Y-%m")

        results[ku] = {
            "method":        method,
            "history_total": int(series.sum()),
            "history":       [{"date": d.strftime("%Y-%m"), "count": int(series.loc[d])} for d in series.index],
            "prediction":    [{"date": future_dates[i], "absolute": round(final[i], 3)} for i in range(horizon)],
        }

    _normalize_shares(results)

    result = {
        "message": "✅ KU forecasting completed.",
        "summary": {
            "KUs detected":   int(df["ku"].nunique()),
            "KUs forecasted": len(results),
            "KUs skipped":    len(skipped),
            "Horizon":        horizon,
            "Time coverage":  f"{df['date'].min().strftime('%Y-%m')} → {df['date'].max().strftime('%Y-%m')}"
        },
        "results": results,
        "skipped": skipped
    }

    return result


# ============================================================
#  ENDPOINT 2: /forecast/policy_skill_forecast
# ============================================================

@router.get("/policy_skill_forecast")
def policy_skill_forecast(
    keywords: str = Query(..., description="Comma-separated keywords, e.g. ai,green,data"),
    horizon:  int = Query(6,   description="Forecast horizon in months (3, 6, 12)")
):
    """
    Forecast ESCO skills appearing in law/policy documents.
    Auto-paginates all available pages. Results cached in Completed_Analyses/.
    """
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")
    from datetime import datetime

    keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]

    # === Cache ===
    cache_folder = _ensure_cache()
    fname  = "completed_analysis_policy_skill_forecast"
    for kw in keywords_list:
        fname += f"_{kw}"
    fname += f"_h{horizon}.json"

    file_path = cache_folder / fname
    print(f"🗂️  Cache path: {file_path}")
    cached = _load_cache(file_path)
    if cached:
        return cached
    
    _start_processing(file_path)

    # === Auth ===
    try:
        print("🔐 Authenticating...")
        token   = _get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/x-www-form-urlencoded",
            "Accept":        "application/json"
        }
        print("✅ Authenticated.")

        # === 1️⃣ Fetch Policies — probe page 1, then auto-paginate all pages ===
        page_size = 100
        print(f"📡 Probing page 1 — keywords: {keywords_list}")
        payload = {"keywords": keywords_list, "keywords_logic": "or", "sources": ["eur_lex"]}
        probe_r = requests.post(f"{API}/law-policies?page=1&page_size={page_size}",
                                headers=headers, data=payload, timeout=60)
        probe_r.raise_for_status()
        probe_data  = probe_r.json()
        total_count = probe_data.get("count", 0)
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
        print(f"📊 Total policies available: {total_count} → {total_pages} page(s)")

        all_docs = list(probe_data.get("items", []))
        print(f"📦 Page 1/{total_pages}: {len(all_docs)} policies")

        for page in range(2, total_pages + 1):
            payload = {"keywords": keywords_list, "keywords_logic": "or", "sources": ["eur_lex"]}
            url = f"{API}/law-policies?page={page}&page_size={page_size}"
            try:
                r     = requests.post(url, headers=headers, data=payload, timeout=60)
                items = r.json().get("items", []) if r.status_code == 200 else []
            except Exception as e:
                print(f"⚠️ Page {page} failed: {e}")
                items = []
            print(f"📦 Page {page}/{total_pages}: {len(items)} policies (running total: {len(all_docs) + len(items)})")
            if not items:
                break
            all_docs.extend(items)
            if len(items) < page_size:
                print("✅ Last page reached.")
                break

        if not all_docs:
            return {"error": "No policies found."}

        print(f"🎯 Total policies: {len(all_docs)}")

        # === 2️⃣ Extract (date, skill_uri) ===
        records = []
        for p in all_docs:
            pub_date = p.get("publication_date")
            if not pub_date:
                continue
            try:
                month = datetime.fromisoformat(pub_date).strftime("%Y-%m")
            except Exception:
                continue
            for s in p.get("skills", []):
                if isinstance(s, str) and s.startswith("http"):
                    records.append({"date": month, "skill_uri": s})

        if not records:
            return {"error": "Policies contain no ESCO skills."}

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        print(f"📊 Skill-date records: {len(df)}")

        # === 3️⃣ Resolve ESCO labels ===
        id_to_label = _resolve_skills(headers)
        df["skill"] = df["skill_uri"].map(id_to_label)
        df = df[df["skill"].notnull()]

        if df.empty:
            return {"error": "No ESCO skills could be mapped (all URIs unmapped)."}

        # === 4️⃣ Build monthly TS ===
        ts = df.pivot_table(
            index="date", columns="skill",
            values="skill_uri", aggfunc="count"
        ).fillna(0)
        print(f"📊 Time series shape: {ts.shape}")

        # === 5️⃣ Forecast ===
        results = {}
        skipped = []

        for skill in ts.columns:
            series = ts[skill]
            if series.sum() < 1:
                skipped.append(skill); continue
            if (series.tail(4) > 0).sum() < 1:
                skipped.append(skill); continue

            final, method = _forecast_series(series, horizon)
            if max(final) < 0.3:
                skipped.append(skill); continue

            future_dates = pd.date_range(
                ts.index[-1] + pd.offsets.MonthBegin(1),
                periods=horizon, freq="MS"
            ).strftime("%Y-%m")

            results[skill] = {
                "method":        method,
                "history_total": int(series.sum()),
                "history":       [{"date": d.strftime("%Y-%m"), "count": int(series.loc[d])} for d in series.index],
                "prediction":    [{"date": future_dates[i], "absolute": round(final[i], 4)} for i in range(horizon)]
            }

        _normalize_shares(results)
        print(f"✅ Forecasted: {len(results)} | Skipped: {len(skipped)}")

        result = {
            "message": "✅ Policy skill forecasting completed.",
            "summary": {
                "Policies retrieved": len(all_docs),
                "Skills detected":    len(ts.columns),
                "Forecasted":         len(results),
                "Skipped":            len(skipped),
                "Horizon":            horizon
            },
            "results": results,
            "skipped": skipped
        }

        _save_cache(file_path, result)
        return result
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        return {"error": f"Forecasting failed: {str(e)}"}


# ============================================================
#  ENDPOINT 3: /forecast/jobs_skill_forecast_NEWONE
# ============================================================

@router.get("/jobs_skill_forecast_NEWONE")
def jobs_skill_forecast(
    keywords:        Optional[str] = Query(None, description="Comma-separated keywords (e.g. AI, data, software)"),
    occupation_ids:  Optional[str] = Query(None, description="Comma-separated occupation IDs (e.g. http://data.europa.eu/esco/isco/C2153)"),
    source:          Optional[str] = Query(None, description="Optional job source (e.g. linkedin, indeed)"),
    min_upload_date: Optional[str] = Query(None, description="Filter jobs uploaded after YYYY-MM-DD"),
    max_upload_date: Optional[str] = Query(None, description="Filter jobs uploaded before YYYY-MM-DD"),
    horizon:         int           = Query(6,    description="Forecast horizon in months (3, 6, or 12)")
):
    """
    Forecast ESCO skills from job postings.
    Supports occupation_ids filter. Auto-paginates ALL available pages.
    Results cached in Completed_Analyses/.
    """
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")
    from datetime import datetime

    keywords_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
    occ_ids_list  = [o.strip() for o in occupation_ids.split(",") if o.strip()] if occupation_ids else []

    # === Cache ===
    cache_folder = _ensure_cache()
    fname = "completed_analysis_jobs_skill_forecast"
    for kw in keywords_list:
        fname += f"_{kw}"
    for occ in occ_ids_list:
        match = re.search(r'C\d+$', occ)
        fname += f"_{match.group(0)}" if match else f"_{occ.replace('/', '_').replace(':', '').replace('.', '')}"
    if source:
        fname += f"_{source}"
    if min_upload_date:
        fname += f"_from{min_upload_date}"
    if max_upload_date:
        fname += f"_to{max_upload_date}"
    fname += f"_h{horizon}.json"

    file_path = cache_folder / fname
    print(f"🗂️  Cache path: {file_path}")
    cached = _load_cache(file_path)
    if cached:
        return cached
    
    _start_processing(file_path)

    try:
        # === Auth ===
        print("🔐 Authenticating...")
        token   = _get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/x-www-form-urlencoded",
            "Accept":        "application/json"
        }
        print("✅ Authenticated.")
        print(f"📡 Keywords     : {keywords_list or '(none)'}")
        print(f"🏢 OccupationIDs: {occ_ids_list  or '(none)'}")
        print(f"🗂️  Source       : {source or '(none)'}")
        print(f"📅 Date range   : {min_upload_date or '*'} → {max_upload_date or '*'}")
        print(f"⚙️  Horizon      : {horizon} months")

        # === 1️⃣ Build form builder ===
        def build_form():
            fd = [
                ("keywords_logic",      "or"),
                ("skill_ids_logic",     "or"),
                ("occupation_ids_logic","or"),
            ]
            for kw in keywords_list:
                fd.append(("keywords", kw))
            for occ in occ_ids_list:
                fd.append(("occupation_ids", occ))
            if source:
                fd.append(("sources", source))
            if min_upload_date:
                fd.append(("min_upload_date", min_upload_date))
            if max_upload_date:
                fd.append(("max_upload_date", max_upload_date))
            return fd

        # === 2️⃣ Auto-paginate jobs ===
        page_size       = 100
        REQUEST_TIMEOUT = 180
        MAX_RETRIES     = 3
        RETRY_BACKOFF   = 10

        def fetch_page(page_num: int) -> dict:
            url = f"{API}/jobs?page={page_num}&page_size={page_size}"
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"   ↪ Attempt {attempt}/{MAX_RETRIES} — page {page_num}...")
                    r = requests.post(url, headers=headers, data=build_form(), timeout=REQUEST_TIMEOUT)
                    if r.status_code != 200:
                        print(f"   ⚠️ HTTP {r.status_code}: {r.text[:200]}")
                        return {}
                    return r.json()
                except requests.exceptions.ReadTimeout:
                    print(f"   ⏱️ Timeout page {page_num}, attempt {attempt}.")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF)
                    else:
                        return {}
                except Exception as ex:
                    print(f"   ❌ {type(ex).__name__}: {ex}")
                    return {}

        print("🔍 Probing page 1 for total job count...")
        probe = fetch_page(1)
        if not probe:
            return {"error": "❌ Probe request (page 1) failed after all retries."}

        total_count = probe.get("count", 0)
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
        print(f"📊 Total jobs available: {total_count} → fetching up to {total_pages} page(s)")

        if total_count == 0:
            return {"error": "No jobs found for the given filters."}

        jobs = list(probe.get("items", []))
        print(f"📦 Page 1/{total_pages}: {len(jobs)} jobs")

        for page in range(2, total_pages + 1):
            print(f"📄 Fetching page {page}/{total_pages}...")
            data  = fetch_page(page)
            items = data.get("items", []) if data else []
            print(f"📦 Page {page}/{total_pages}: {len(items)} jobs (running total: {len(jobs) + len(items)})")
            if not items:
                break
            jobs.extend(items)
            if len(items) < page_size:
                print("✅ Last page reached.")
                break

        print(f"🎯 Total jobs retrieved: {len(jobs)} / {total_count}")

        # === 3️⃣ Extract (date, skill_uri) ===
        records = []
        for job in jobs:
            dt = job.get("upload_date")
            try:
                month = datetime.fromisoformat(dt).strftime("%Y-%m")
            except Exception:
                continue
            for s in job.get("skills", []):
                if isinstance(s, str) and s.startswith("http"):
                    records.append({"date": month, "skill_uri": s})

        if not records:
            return {"error": "No ESCO skill URIs found in jobs."}

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        print(f"📊 Skill-date records: {len(df)} | Unique URIs: {df['skill_uri'].nunique()}")

        # === 4️⃣ Resolve ESCO labels ===
        id_to_label = _resolve_skills(headers)
        df["skill"] = df["skill_uri"].map(id_to_label)
        df = df[df["skill"].notnull()]

        if df.empty:
            return {"error": "Could not map ESCO URIs to labels."}

        print(f"✅ Mapped skills: {df['skill'].nunique()} unique labels")

        # === 5️⃣ Build time-series matrix ===
        ts = df.pivot_table(
            index="date", columns="skill",
            values="skill_uri", aggfunc="count"
        ).fillna(0)
        print(f"📊 Time series shape: {ts.shape}")

        # === 6️⃣ Forecast ===
        results = {}
        skipped = []

        for skill in ts.columns:
            series = ts[skill]
            if series.sum() < 2:
                skipped.append(skill); continue
            if (series.tail(4) > 0).sum() < 1:
                skipped.append(skill); continue

            final, method = _forecast_series(series, horizon)
            if max(final) < 0.3:
                skipped.append(skill); continue

            future = pd.date_range(
                ts.index[-1] + pd.offsets.MonthBegin(1),
                periods=horizon, freq="MS"
            ).strftime("%Y-%m")

            results[skill] = {
                "method":        method,
                "history_total": int(series.sum()),
                "history":       [{"date": d.strftime("%Y-%m"), "count": int(series.loc[d])} for d in series.index],
                "prediction":    [{"date": future[i], "absolute": round(final[i], 3)} for i in range(horizon)]
            }

        _normalize_shares(results)
        print(f"✅ Forecasted: {len(results)} skills | Skipped: {len(skipped)}")

        result = {
            "message": "✅ Job skill forecasting completed.",
            "filters_used": {
                "keywords":        keywords_list or None,
                "occupation_ids":  occ_ids_list  or None,
                "source":          source,
                "min_upload_date": min_upload_date,
                "max_upload_date": max_upload_date,
            },
            "summary": {
                "Total jobs retrieved": len(jobs),
                "Total jobs available": total_count,
                "Skills detected":      len(ts.columns),
                "Skills forecasted":    len(results),
                "Skills skipped":       len(skipped),
                "Horizon":              horizon
            },
            "results": results,
            "skipped": skipped
        }

        _save_cache(file_path, result)
        return result
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        return {"error": f"Forecasting failed: {str(e)}"}


# ============================================================
#  Entry point — run with: python skill_forecast.py
#  or:          uvicorn skill_forecast:app --reload
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ageing_forec:app", host="0.0.0.0", port=8001, reload=True)
