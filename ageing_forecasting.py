
# ==========================================================
#  SKILLAB Skill Intelligence API
#  Module: Job Skill Timeline & Forecasting Preprocessing
#  Endpoint: /api/analysis/jobs_ultra_forecasting
# ==========================================================

from fastapi import FastAPI, APIRouter, Query

# -----------------------------
# Initialize FastAPI + Router
# -----------------------------
app = FastAPI(title="SKILLAB Skill Intelligence API")
analysis_router = APIRouter(prefix="/api/analysis", tags=["SKILL Analysis"])
# forecasting_router = APIRouter()


# ==========================================================
#    JOB SKILL TIMELINE ENDPOINT (BASE FOR FORECASTING)
# ==========================================================

@analysis_router.get("/ku_forecast_arima")
def ku_forecast(
    horizon: int = Query(6, description="Forecast horizon in months (3, 6, or 12)"),
    start_date: str = Query(None, description="Start date YYYY-MM"),
    end_date: str = Query(None, description="End date YYYY-MM"),
    organization: str = Query(None, description="Filter by organization")
):
    """
    Forecast KU (Knowledge Unit) activity:
    - Builds monthly KU counts directly from SKILLAB API
    - Forecasts with ARIMA (if non-flat), otherwise trend fallback, else moving average
    - Normalized share included
    """

    import requests
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings("ignore")

    # === 1️⃣ Fetch KU detection data from SKILLAB ===
    BASE_URL = "https://portal.skillab-project.eu/ku-detection/analysis_results"

    params = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if organization:
        params["organization"] = organization

    print(f"🔍 Fetching KU records with filters {params}")
    res = requests.get(BASE_URL, params=params, timeout=60)
    res.raise_for_status()
    ku_data = res.json()

    if not ku_data:
        return {"error": "No KU data found for the selected filters."}

    print(f"📄 Retrieved {len(ku_data)} KU analysis records.")

    # === 2️⃣ Build monthly KU counts ===========================
    records = []
    for rec in ku_data:
        detected = rec.get("detected_kus", {})
        timestamp = rec.get("timestamp")
        try:
            month = datetime.fromisoformat(timestamp).strftime("%Y-%m")
        except Exception:
            continue

        for ku, v in detected.items():
            # keep active KUs (value == "1" or 1)
            if str(v) == "1":
                records.append({"date": month, "ku": ku})

    df = pd.DataFrame(records)
    if df.empty:
        return {"error": "No KU detections found after filtering."}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    print(f"📊 Raw KU records: {len(df)}")
    print(f"🔢 Unique KUs in df: {df['ku'].nunique()}")

    # === 2b️⃣ Pivot to time series: rows = month, columns = KU, values = counts ===
    # IMPORTANT: we do NOT use `values="ku"` here, only `aggfunc='size'`
    ts = df.pivot_table(
        index="date",
        columns="ku",
        aggfunc="size"   # counts rows per (date, ku)
    ).fillna(0)

    print(f"📊 Timeline shape: {ts.shape}  (rows={ts.shape[0]}, KUs={ts.shape[1]})")

    if ts.shape[1] == 0:
        return {
            "error": "Pivot produced no KU columns. Check if detected_kus are actually active (value == 1)."
        }

    # === 3️⃣ Forecasting =======================================
    forecast_results = {}
    skipped = []

    min_total_count = 2         # minimum total occurrences across whole period
    min_recent_months = 1       # at least 1 non-zero in last 4 months
    min_len_arima = 5           # minimum length for ARIMA

    for ku in ts.columns:
        series = ts[ku]

        # Skip very rare KUs
        if series.sum() < min_total_count:
            skipped.append(ku)
            continue

        recent = series.tail(4)
        if (recent > 0).sum() < min_recent_months:
            skipped.append(ku)
            continue

        last_val = float(series.iloc[-1])
        last3 = series.tail(3)

        # 1️⃣ Trend model
        if last3.nunique() > 1:
            trend = last3.diff().mean()
            trend_pred = [max(0, last_val + (i + 1) * trend) for i in range(horizon)]
        else:
            trend_pred = None

        # 2️⃣ Moving Average
        ma_val = float(last3.mean())
        ma_pred = [ma_val] * horizon

        # 3️⃣ ARIMA
        arima_pred = None
        if len(series) >= min_len_arima:
            try:
                model = ARIMA(series, order=(1, 1, 1))
                fit = model.fit()
                raw = fit.forecast(steps=horizon)
                arima_pred = [max(0, float(x)) for x in raw]
            except Exception as e:
                print(f"⚠️ ARIMA failed for {ku}: {e}")
                arima_pred = None

        # Flat-forecast rejection
        use_arima = False
        if arima_pred is not None:
            if max(arima_pred) - min(arima_pred) >= 0.3:
                use_arima = True

        # Model selection
        if use_arima:
            final = arima_pred
            method = "arima"
        elif trend_pred is not None:
            final = trend_pred
            method = "trend"
        else:
            final = ma_pred
            method = "moving_average"

        if max(final) < 0.3:
            skipped.append(ku)
            continue

        # Build future months
        future_dates = pd.date_range(
            ts.index[-1] + pd.offsets.MonthBegin(1),
            periods=horizon,
            freq="MS"
        ).strftime("%Y-%m")

        forecast_results[ku] = {
            "method": method,
            "history_total": int(series.sum()),
            "history": [
                {"date": d.strftime("%Y-%m"), "count": int(series.loc[d])}
                for d in series.index
            ],
            "prediction": [
                {"date": future_dates[i], "absolute": round(final[i], 3)}
                for i in range(horizon)
            ],
        }

    # === 4️⃣ Normalize (share per month) ======================
    month_totals = {}
    for ku, data in forecast_results.items():
        for p in data["prediction"]:
            month_totals[p["date"]] = month_totals.get(p["date"], 0) + p["absolute"]

    for ku, data in forecast_results.items():
        for p in data["prediction"]:
            total = month_totals[p["date"]]
            p["share"] = round(p["absolute"] / total, 5) if total > 0 else 0

    # === RETURN ==============================================
    return {
        "message": "✅ KU forecasting completed.",
        "summary": {
            "KUs detected": int(df["ku"].nunique()),
            "KUs forecasted": len(forecast_results),
            "KUs skipped": len(skipped),
            "Horizon": horizon,
            "Time coverage": f"{df['date'].min().strftime('%Y-%m')} → {df['date'].max().strftime('%Y-%m')}"
        },
        "results": forecast_results,
        "skipped": skipped
    }


@analysis_router.get("/policy_skill_forecast")
def policy_skill_forecast(
    keywords: str = Query(..., description="Comma-separated keywords, e.g. ai,green,data"),
    horizon: int = Query(6, description="Forecast horizon in months (3, 6, 12)"),
    max_pages: int = Query(40, description="Pages to fetch from Tracker policies")
):
    """
    Forecast ESCO skills appearing in policies (from publication_date).
    Steps:
      1. Fetch policies
      2. Extract publication_date + ESCO skill URIs
      3. Map URI -> ESCO label
      4. Remove unmapped URIs
      5. Build monthly time series
      6. ARIMA → TREND → MOVING AVERAGE forecasting
      7. Compute normalized shares
    """

    import os, requests, pandas as pd, numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from datetime import datetime
    from dotenv import load_dotenv
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings("ignore")

    try:
        # === AUTH ===
        load_dotenv()
        API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
        USER = os.getenv("TRACKER_USERNAME", "")
        PASS = os.getenv("TRACKER_PASSWORD", "")

        res = requests.post(f"{API}/login",
                            json={"username": USER, "password": PASS},
                            timeout=15)
        token = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}"}

        # === FETCH POLICIES ===
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        payload = {
            "keywords": keywords_list,
            "keywords_logic": "or",
            "sources": ["eur_lex"]
        }

        all_docs = []
        for page in range(1, max_pages + 1):
            url = f"{API}/law-policies?page={page}&page_size=100"
            r = requests.post(url, headers=headers, data=payload, timeout=60)
            if r.status_code != 200:
                break
            items = r.json().get("items", [])
            if not items:
                break
            all_docs.extend(items)
            if len(items) < 100:
                break

        if not all_docs:
            return {"error": "No policies found."}

        # === EXTRACT (date, skill_uri) ===
        records = []
        for p in all_docs:
            pub_date = p.get("publication_date")
            if not pub_date:
                continue

            try:
                month = datetime.fromisoformat(pub_date).strftime("%Y-%m")
            except:
                continue

            for s in p.get("skills", []):
                if isinstance(s, str) and s.startswith("http"):
                    records.append({"date": month, "skill_uri": s})

        if not records:
            return {"error": "Policies contain no ESCO skills."}

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # === MAP ESCO LABELS ===
        esco_items = []
        for page in range(1, 40):
            r = requests.post(f"{API}/skills?page={page}&page_size=100",
                              headers=headers, timeout=30)
            items = r.json().get("items", [])
            if not items:
                break
            esco_items.extend(items)

        id_to_label = {e["id"]: e["label"].lower() for e in esco_items}

        df["skill"] = df["skill_uri"].map(id_to_label)

        # REMOVE ANYTHING THAT IS STILL A URI (UNMAPPED)
        df = df[df["skill"].notnull()]

        if df.empty:
            return {"error": "No ESCO skills could be mapped (all URIs unmapped)."}

        # === BUILD MONTHLY TS ===
        ts = df.pivot_table(index="date", columns="skill", values="skill", aggfunc="count").fillna(0)

        # === FORECASTING ===
        results = {}
        skipped = []

        min_total = 1
        min_recent = 1
        min_len_arima = 5

        for skill in ts.columns:
            series = ts[skill]

            if series.sum() < min_total:
                skipped.append(skill)
                continue
            if (series.tail(4) > 0).sum() < min_recent:
                skipped.append(skill)
                continue

            last_val = float(series.iloc[-1])
            last3 = series.tail(3)

            # Trend
            if last3.nunique() > 1:
                trend = last3.diff().mean()
                trend_pred = [max(0, last_val + (i + 1) * trend) for i in range(horizon)]
            else:
                trend_pred = None

            # Moving Avg
            ma_val = float(last3.mean())
            ma_pred = [ma_val] * horizon

            # ARIMA
            arima_pred = None
            if len(series) >= min_len_arima:
                try:
                    model = ARIMA(series, order=(1, 1, 1))
                    fit = model.fit()
                    raw = fit.forecast(steps=horizon)
                    arima_pred = [max(0, float(x)) for x in raw]
                except:
                    arima_pred = None

            # ARIMA must not be flat
            use_arima = False
            if arima_pred is not None:
                if max(arima_pred) - min(arima_pred) >= 0.3:
                    use_arima = True

            if use_arima:
                final = arima_pred
                method = "arima"
            elif trend_pred is not None:
                final = trend_pred
                method = "trend"
            else:
                final = ma_pred
                method = "moving_average"

            if max(final) < 0.3:
                skipped.append(skill)
                continue

            future_dates = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1),
                                         periods=horizon, freq="MS").strftime("%Y-%m")

            # HISTORY (for frontend time series plot)
            history = [
                {"date": d.strftime("%Y-%m"), "count": int(series.loc[d])}
                for d in series.index
            ]

            results[skill] = {
                "method": method,
                "history_total": int(series.sum()),
                "history": history,
                "prediction": [
                    {"date": future_dates[i], "absolute": round(final[i], 4)}
                    for i in range(horizon)
                ]
            }

        # === SHARE NORMALIZATION ===
        total_by_month = {}
        for skill, d in results.items():
            for p in d["prediction"]:
                total_by_month[p["date"]] = total_by_month.get(p["date"], 0) + p["absolute"]

        for skill, d in results.items():
            for p in d["prediction"]:
                total = total_by_month[p["date"]]
                p["share"] = round(p["absolute"] / total, 6) if total > 0 else 0

        return {
            "message": "✅ Policy skill forecasting completed.",
            "summary": {
                "Skills detected": len(ts.columns),
                "Forecasted": len(results),
                "Skipped": len(skipped)
            },
            "results": results,
            "skipped": skipped
        }

    except Exception as e:
        return {"error": str(e)}


@analysis_router.get("/jobs_skill_forecast_NEWONE")
def jobs_skill_forecast(
    keywords: str = Query(..., description="Comma-separated keywords (e.g. AI, data, software)"),
    source: str = Query(None, description="Optional job source (e.g. linkedin, indeed)"),
    min_upload_date: str = Query(None, description="Filter jobs uploaded after YYYY-MM-DD"),
    max_upload_date: str = Query(None, description="Filter jobs uploaded before YYYY-MM-DD"),
    horizon: int = Query(6, description="Forecast horizon in months (3, 6, or 12)"),
    max_pages: int = Query(10, description="Pages to fetch (100 jobs/page)")
):
    """
    Fully dynamic job-skill forecasting:
    - Fetch jobs directly from Tracker
    - Remove URI skills
    - Build monthly time series of ESCO skill usage
    - Forecast using ARIMA → Trend → MovingAvg
    - Return history + predictions + normalized share
    """
    import os, requests, pandas as pd, numpy as np
    from datetime import datetime
    from dotenv import load_dotenv
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings("ignore")

    # === 1️⃣ Authenticate ===
    load_dotenv()
    API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
    USER = os.getenv("TRACKER_USERNAME", "")
    PASS = os.getenv("TRACKER_PASSWORD", "")

    auth = requests.post(
        f"{API}/login",
        json={"username": USER, "password": PASS},
        timeout=15
    )
    token = auth.text.replace('"', "")
    headers = {"Authorization": f"Bearer {token}"}

    # === 2️⃣ Fetch Jobs ===
    keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
    jobs = []

    for page in range(1, max_pages + 1):
        form = [
            ("keywords_logic", "or"),
            ("skill_ids_logic", "or"),
            ("occupation_ids_logic", "or")
        ]
        for kw in keywords_list:
            form.append(("keywords", kw))
        if source:
            form.append(("sources", source))
        if min_upload_date:
            form.append(("min_upload_date", min_upload_date))
        if max_upload_date:
            form.append(("max_upload_date", max_upload_date))

        r = requests.post(
            f"{API}/jobs?page={page}&page_size=100",
            headers=headers,
            data=form,
            timeout=60
        )

        if r.status_code != 200:
            break

        items = r.json().get("items", [])
        if not items:
            break

        jobs.extend(items)
        if len(items) < 100:
            break

    if not jobs:
        return {"error": "No jobs found."}

    # === 3️⃣ Extract skill URIs per month ===
    records = []
    for job in jobs:
        dt = job.get("upload_date")
        try:
            month = datetime.fromisoformat(dt).strftime("%Y-%m")
        except:
            continue

        for s in job.get("skills", []):
            if isinstance(s, str) and s.startswith("http"):
                records.append({"date": month, "skill_uri": s})

    if not records:
        return {"error": "No ESCO skills found in jobs."}

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # === 4️⃣ Map ESCO URIs → labels ===
    esco_items = []
    for page in range(1, 50):
        rr = requests.post(
            f"{API}/skills?page={page}&page_size=100",
            headers=headers,
            timeout=20
        )
        items = rr.json().get("items", [])
        if not items:
            break
        esco_items.extend(items)

    id_to_label = {x["id"]: x["label"].lower() for x in esco_items}

    df["skill"] = df["skill_uri"].map(id_to_label)

    # REMOVE unmapped URIs
    df = df[df["skill"].notnull()]

    if df.empty:
        return {"error": "Could not map ESCO URIs to labels."}

    # === 5️⃣ Build time-series matrix ===
    ts = df.pivot_table(
        index="date",
        columns="skill",
        values="skill",
        aggfunc="count"
    ).fillna(0)

    # === 6️⃣ Forecast for each skill ===
    results = {}
    skipped = []

    min_total = 2
    min_recent = 1
    min_arima_len = 5

    for skill in ts.columns:
        series = ts[skill]

        if series.sum() < min_total:
            skipped.append(skill)
            continue
        if (series.tail(4) > 0).sum() < min_recent:
            skipped.append(skill)
            continue

        last_val = float(series.iloc[-1])
        last3 = series.tail(3)

        # Trend
        if last3.nunique() > 1:
            trend = last3.diff().mean()
            trend_pred = [max(0, last_val + (i + 1) * trend) for i in range(horizon)]
        else:
            trend_pred = None

        # Moving Avg
        ma_val = float(last3.mean())
        ma_pred = [ma_val] * horizon

        # ARIMA
        arima_pred = None
        if len(series) >= min_arima_len:
            try:
                model = ARIMA(series, order=(1, 1, 1))
                fit = model.fit()
                raw = fit.forecast(steps=horizon)
                arima_pred = [max(0, float(x)) for x in raw]
            except:
                pass

        # ARIMA must NOT be flat
        use_arima = False
        if arima_pred is not None:
            if max(arima_pred) - min(arima_pred) >= 0.3:
                use_arima = True

        # model choice
        if use_arima:
            final = arima_pred
            method = "arima"
        elif trend_pred is not None:
            final = trend_pred
            method = "trend"
        else:
            final = ma_pred
            method = "moving_average"

        if max(final) < 0.3:
            skipped.append(skill)
            continue

        future = pd.date_range(
            ts.index[-1] + pd.offsets.MonthBegin(1),
            periods=horizon,
            freq="MS"
        ).strftime("%Y-%m")

        # Build history
        history = [{"date": d.strftime("%Y-%m"), "count": int(series.loc[d])} for d in series.index]

        # Save result
        results[skill] = {
            "method": method,
            "history_total": int(series.sum()),
            "history": history,
            "prediction": [
                {"date": future[i], "absolute": round(final[i], 3)}
                for i in range(horizon)
            ]
        }

    # === 7️⃣ Normalize forecast share ===
    totals = {}
    for skill, d in results.items():
        for p in d["prediction"]:
            totals[p["date"]] = totals.get(p["date"], 0) + p["absolute"]

    for skill, d in results.items():
        for p in d["prediction"]:
            total = totals[p["date"]]
            p["share"] = round(p["absolute"] / total, 6) if total > 0 else 0

    # === Return ===
    return {
        "message": "✅ Job skill forecasting completed.",
        "summary": {
            "Total jobs": len(jobs),
            "Skills detected": len(ts.columns),
            "Skills forecasted": len(results),
            "Skills skipped": len(skipped),
            "Horizon": horizon
        },
        "results": results,
        "skipped": skipped
    }







# -----------------------------
# Register router into FastAPI
# -----------------------------
app.include_router(analysis_router)
