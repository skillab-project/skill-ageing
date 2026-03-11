from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from collections import defaultdict
from itertools import combinations
from itertools import islice
from sklearn.linear_model import LinearRegression
import json
import uuid
from pathlib import Path
import os
from dotenv import load_dotenv
from ageing_forecasting import router as forecasting_router

load_dotenv()

app = FastAPI(
    title="Skill Ageing API",
    root_path="/skill-ageing"
)
app.include_router(forecasting_router)

import os
print("🧩 TRACKER_API =", os.getenv("TRACKER_API"))
print("🧩 TRACKER_USERNAME =", os.getenv("TRACKER_USERNAME"))
print("🧩 TRACKER_PASSWORD =", os.getenv("TRACKER_PASSWORD"))

API = os.getenv("TRACKER_API")
USERNAME = os.getenv("TRACKER_USERNAME")
PASSWORD = os.getenv("TRACKER_PASSWORD")
KU_API_URL = os.getenv("KU_API_URL")

# ============================================================
#  SHARED HELPERS
# ============================================================

def get_token():
    """Authenticate and return a fresh Bearer token."""
    res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD})
    res.raise_for_status()
    return res.text.replace('"', "")


def get_total_jobs_in_tracker():
    """Get total job count from the Tracker (no filters)."""
    token = get_token()
    res = requests.post(f"{API}/jobs", headers={"Authorization": f"Bearer {token}"}, data={})
    res.raise_for_status()
    return res.json().get("count", 0)


# ============================================================
#  SKILL ANALYSIS CORE
# ============================================================

def run_skill_analysis_from_list(job_list):
    """Run the full Skill Ageing / Epidemiological analysis on a list of items."""

    skill_occurrences = defaultdict(list)
    for job in job_list:
        try:
            date_str = job.get("upload_date")
            if not date_str:
                continue
            date = datetime.strptime(date_str, "%Y-%m-%d")
            skills = job.get("skills", [])
            if not isinstance(skills, list):
                continue
            for skill in skills:
                skill_occurrences[skill].append(date)
        except:
            continue

    biology_summary = []
    combined_index = pd.date_range(start="2020-01-01", end="2025-12-31", freq="M")

    def get_slope(ts):
        if len(ts) < 3:
            return 0
        X = np.arange(len(ts)).reshape(-1, 1)
        y = ts.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]

    for skill, dates in skill_occurrences.items():
        df = pd.DataFrame(dates, columns=["date"])
        df["year_month"] = df["date"].dt.to_period("M")
        birth = df["date"].min()
        peak = df["year_month"].value_counts().idxmax()
        total_jobs = len(df)
        recent_use = df["date"].max().year > 2022
        immunity = "High" if total_jobs > 20 and recent_use else "Low"

        s = pd.Series(1, index=pd.to_datetime(dates))
        s = s.resample("M").sum().reindex(combined_index, fill_value=0)
        slope = get_slope(s)

        if slope < -0.01:
            trend = "Declining"
        elif slope > 0.01:
            trend = "Rising"
        else:
            trend = "Stable"

        biology_summary.append({
            "Skill": skill,
            "Date of Birth": birth.strftime("%Y-%m-%d"),
            "Peak Activity Date": str(peak),
            "Total Jobs": total_jobs,
            "Immunity Score": immunity,
            "Trend": trend,
            "Slope": round(slope, 4)
        })

    # === Time Series Construction ===
    tag_series = {}
    for skill, dates in skill_occurrences.items():
        s = pd.Series(1, index=pd.to_datetime(dates))
        s = s.resample("M").sum().reindex(combined_index, fill_value=0)
        tag_series[skill] = s

    all_tags_df = pd.DataFrame(tag_series).fillna(0)
    filtered_tags = [tag for tag in all_tags_df.columns if all_tags_df[tag].sum() >= 10]

    # === Competing Skills ===
    competing_results = []
    for tag1, tag2 in combinations(filtered_tags, 2):
        s1, s2 = all_tags_df[tag1], all_tags_df[tag2]
        overlap = (s1 > 0) & (s2 > 0)
        if overlap.sum() < 5:
            continue
        corr = s1[overlap].corr(s2[overlap])
        if pd.notna(corr) and corr < -0.5:
            competing_results.append({"Skill A": tag1, "Skill B": tag2, "Correlation": round(corr, 3)})

    # === Inverse Trends ===
    inverse_results = []
    top_skills = sorted(filtered_tags, key=lambda x: all_tags_df[x].sum(), reverse=True)[:100]
    for tag1, tag2 in combinations(top_skills, 2):
        s1, s2 = all_tags_df[tag1], all_tags_df[tag2]
        mask = (s1 > 0) & (s2 > 0)
        if mask.sum() < 6:
            continue
        slope1, slope2 = get_slope(s1[mask]), get_slope(s2[mask])
        if slope1 < -0.005 and slope2 > 0.005:
            inverse_results.append({
                "Declining Skill": tag1, "Competing Skill": tag2,
                "Slope A": round(slope1, 4), "Slope B": round(slope2, 4),
                "Overlapping Months": int(mask.sum())
            })

    # === Rapid Obsolescence ===
    rapid_drops = []
    for tag, series in tag_series.items():
        if (series > 0).sum() < 12:
            continue
        peak_value = series.max()
        if peak_value < 5:
            continue
        peak_idx = series.idxmax()
        peak_loc = series.index.get_loc(peak_idx)
        post_peak = series.iloc[peak_loc:peak_loc + 7]
        drop_ratio = (peak_value - post_peak.min()) / peak_value
        if drop_ratio >= 0.3:
            rapid_drops.append({
                "Skill": tag,
                "Peak Month": peak_idx.strftime("%Y-%m"),
                "Peak Value": int(peak_value),
                "Min Value After Peak": int(post_peak.min()),
                "Drop %": round(drop_ratio * 100, 2)
            })

    # === Epidemiological Metrics ===
    epi_metrics = []
    shock_start, shock_end = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")
    old_start, old_end = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")

    for tag, series in tag_series.items():
        series = series.fillna(0)
        total_jobs = series.sum()
        if total_jobs == 0:
            continue
        incidence = series.loc[shock_start:shock_end].sum()
        old_incidence = series.loc[old_start:old_end].sum()
        pct_change = 100 * (incidence - old_incidence) / old_incidence if old_incidence > 0 else (999 if incidence > 0 else 0)
        ip_ratio = incidence / total_jobs if total_jobs else 0
        recent_activity = series[series.index >= datetime(2023, 7, 1)].sum()
        is_dead = recent_activity == 0
        revival = "Yes" if old_incidence < incidence and old_incidence > 0 else "No"
        mortality_ratio = incidence / (total_jobs - incidence) if (total_jobs - incidence) > 0 else 999
        was_active = old_incidence > 0 or incidence > 0
        cfr = 1.0 if (was_active and is_dead) else 0.0
        attack_rate = (series > 0).sum() / len(series)

        epi_metrics.append({
            "Skill": tag,
            "Total Jobs": int(total_jobs),
            "Incidence (2023)": int(incidence),
            "Incidence (2022)": int(old_incidence),
            "% Change in Incidence": round(pct_change, 2),
            "Incidence : Prevalence": round(ip_ratio, 4),
            "Mortality Risk": "☠️" if is_dead else "🟢",
            "Revived?": revival,
            "Incidence : Mortality Ratio": round(mortality_ratio, 2),
            "CFR": round(cfr, 2),
            "Attack Rate": round(attack_rate, 4)
        })

    output = {
        "skill_biology_summary": biology_summary,
        "competing_skills": competing_results,
        "inverse_trends": inverse_results,
        "rapid_obsolescence": rapid_drops,
        "epidemiological_metrics": epi_metrics,
        "total_jobs_analyzed": len(job_list)
    }

    Path("results").mkdir(parents=True, exist_ok=True)
    filename = f"results/skill_analysis_{uuid.uuid4().hex[:6]}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return {
        "message": "✅ Skill analysis complete",
        "file_saved": filename,
        "summary": {
            "Total Skills Found": len(biology_summary),
            "Competing Skill Pairs": len(competing_results),
            "Inverse Trend Pairs": len(inverse_results),
            "Rapidly Obsolete Skills": len(rapid_drops),
            "Epidemiological Metrics": len(epi_metrics),
        },
        "data": output
    }


# ============================================================
#  ✅ CORRECTED ENDPOINT: /skill-ageing
#  Changes vs original:
#    1. occupation_ids is Optional — user may pass one or many
#    2. source is Optional
#    3. Removed hardcoded limit=100000 — auto-paginates ALL pages
#    4. Retry logic (3 attempts, 180s timeout, 10s backoff)
#    5. Results cached in Completed_Analyses/<filename>.json
#    6. Skill URI → label resolution preserved
#    7. Full print monitoring throughout
# ============================================================

@app.get("/skill-ageing")
def analyze_skills(
    occupation_ids: Optional[str] = Query(
        None, description="Comma-separated occupation IDs (e.g. http://data.europa.eu/esco/isco/C2153)"
    ),
    source: Optional[str] = Query(
        None, description="Optional source filter (e.g. linkedin, indeed)"
    ),
    min_upload_date: Optional[str] = Query(None, description="Minimum upload date (YYYY-MM-DD)"),
    max_upload_date: Optional[str] = Query(None, description="Maximum upload date (YYYY-MM-DD)"),
):
    """
    Fetch ALL available job pages for given occupation IDs and source, resolve skill URIs to labels,
    run the full Skill Ageing + Epidemiological analysis, and cache the result locally.
    """
    API_URL = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
    USERNAME_ENV = os.getenv("TRACKER_USERNAME", "skillab_staff")
    PASSWORD_ENV = os.getenv("TRACKER_PASSWORD", "skillroadtrip00")

    # === 📁 Setup local cache folder ===
    folder = Path("Completed_Analyses")
    if not folder.exists():
        folder.mkdir(parents=True)
        print(f"📁 Folder '{folder}' created.")
    else:
        print(f"📁 Folder '{folder}' already exists, moving on.")

    # === 🏷️ Build deterministic cache filename ===
    occ_ids_list = [o.strip() for o in occupation_ids.split(",") if o.strip()] if occupation_ids else []

    filename = "completed_analysis_skill_ageing"
    for occ in occ_ids_list:
        match = re.search(r'C\d+$', occ)
        filename += f"_{match.group(0)}" if match else f"_{occ.replace('/', '_').replace(':', '').replace('.', '')}"
    if source:
        filename += f"_{source}"
    if min_upload_date:
        filename += f"_from{min_upload_date}"
    if max_upload_date:
        filename += f"_to{max_upload_date}"
    filename += ".json"

    file_path = folder / filename
    print(f"🗂️ Cache file path: {file_path}")

    # === ✅ Return cached result if it already exists ===
    if file_path.exists():
        print(f"✅ Cache hit — loading results from '{file_path}' (skipping API call).")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.loads(f.read())

    print(f"🌐 No cache found — running full analysis from API...")

    try:
        # === 1️⃣ Authenticate ===
        print("🔐 Authenticating with Tracker...")
        res = requests.post(
            f"{API_URL}/login",
            json={"username": USERNAME_ENV, "password": PASSWORD_ENV},
            timeout=15
        )
        res.raise_for_status()
        token = res.text.replace('"', "")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        print("✅ Authenticated successfully.")

        # === 2️⃣ Log applied filters ===
        print(f"🏢 Occupation IDs filter: {occ_ids_list if occ_ids_list else '(none)'}")
        print(f"🗂️ Source filter: {source if source else '(none)'}")
        if min_upload_date or max_upload_date:
            print(f"📅 Date range: {min_upload_date or 'any'} → {max_upload_date or 'any'}")

        # === 3️⃣ Helper: build form_data ===
        def build_form_data():
            fd = [("keywords_logic", "or"), ("skill_ids_logic", "or"), ("occupation_ids_logic", "or")]
            for occ in occ_ids_list:
                fd.append(("occupation_ids", occ))
            if source:
                fd.append(("sources", source))
            if min_upload_date:
                fd.append(("min_upload_date", min_upload_date))
            if max_upload_date:
                fd.append(("max_upload_date", max_upload_date))
            return fd

        # === 4️⃣ Constants ===
        page_size = 100
        REQUEST_TIMEOUT = 180
        MAX_RETRIES = 3
        RETRY_BACKOFF = 10

        def fetch_page_with_retry(page_num: int) -> dict:
            url = f"{API_URL}/jobs?page={page_num}&page_size={page_size}"
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"   ↪ Attempt {attempt}/{MAX_RETRIES} for page {page_num} (timeout={REQUEST_TIMEOUT}s)...")
                    r = requests.post(url, headers=headers, data=build_form_data(), timeout=REQUEST_TIMEOUT)
                    if r.status_code != 200:
                        print(f"   ⚠️ HTTP {r.status_code} on page {page_num}: {r.text[:300]}")
                        return {}
                    return r.json()
                except requests.exceptions.ReadTimeout:
                    print(f"   ⏱️ ReadTimeout on page {page_num}, attempt {attempt}/{MAX_RETRIES}.")
                    if attempt < MAX_RETRIES:
                        print(f"   🔄 Retrying in {RETRY_BACKOFF}s...")
                        time.sleep(RETRY_BACKOFF)
                    else:
                        print(f"   ❌ All {MAX_RETRIES} attempts exhausted for page {page_num} — skipping.")
                        return {}
                except requests.exceptions.ConnectionError as conn_err:
                    print(f"   ❌ ConnectionError on page {page_num}: {conn_err}")
                    return {}
                except Exception as ex:
                    print(f"   ❌ Unexpected error on page {page_num}: {type(ex).__name__}: {ex}")
                    return {}

        # === 5️⃣ Probe page 1 to determine total count & pages ===
        print("🔍 Probing API page 1 to determine total record count...")
        probe_data = fetch_page_with_retry(1)

        if not probe_data:
            return {"error": "❌ Probe request (page 1) failed after all retries. Cannot determine total count."}

        total_count = probe_data.get("count", 0)
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
        print(f"📊 Total records available: {total_count} → {total_pages} page(s) to fetch (page_size={page_size})")

        if total_count == 0:
            print("⚠️ No job postings found for the given filters.")
            return {"message": "No job postings found for the given filters."}

        # === 6️⃣ Paginate through ALL available pages ===
        all_jobs = list(probe_data.get("items", []))
        print(f"📦 Page 1/{total_pages}: {len(all_jobs)} jobs loaded from probe.")

        for page in range(2, total_pages + 1):
            print(f"📄 Fetching page {page}/{total_pages}...")
            data = fetch_page_with_retry(page)

            if not data:
                print(f"⚠️ Page {page} returned no data after retries — stopping pagination early.")
                break

            items = data.get("items", [])
            print(f"📦 Page {page}/{total_pages}: {len(items)} jobs (running total: {len(all_jobs) + len(items)})")

            if not items:
                print("✅ No more results — stopping early.")
                break

            all_jobs.extend(items)

            if len(items) < page_size:
                print("✅ Last page reached (fewer results than page_size).")
                break

        print(f"🎯 Total jobs retrieved: {len(all_jobs)} / {total_count} available")

        if not all_jobs:
            return {"message": "No job postings found for the given filters."}

        # === 7️⃣ Collect unique skill URIs found in this job set ===
        unique_skill_ids = set()
        for job in all_jobs:
            for sid in job.get("skills", []):
                if isinstance(sid, str) and sid.startswith("http"):
                    unique_skill_ids.add(sid)

        print(f"📚 Found {len(unique_skill_ids)} unique skill URIs — resolving only these labels in batches...")

        # === 8️⃣ Resolve ONLY the found URIs → labels, in batches of 50 ===
        # We POST the specific IDs we need, not all skills in the tracker.
        id_to_label = {}
        if unique_skill_ids:
            try:
                uri_list = list(unique_skill_ids)
                batch_size = 50
                total_batches = math.ceil(len(uri_list) / batch_size)
                print(f"   Sending {total_batches} batch(es) of up to {batch_size} IDs each...")

                for batch_num, start in enumerate(range(0, len(uri_list), batch_size), 1):
                    batch = uri_list[start:start + batch_size]
                    skill_payload = [("ids", sid) for sid in batch]
                    print(f"   Batch {batch_num}/{total_batches}: resolving {len(batch)} URIs...")
                    skill_res = requests.post(
                        f"{API_URL}/skills",
                        headers={"Authorization": f"Bearer {token}"},
                        data=skill_payload,
                        timeout=60
                    )
                    skill_res.raise_for_status()
                    skill_items = skill_res.json().get("items", [])
                    for s in skill_items:
                        sid = s.get("id", "")
                        if sid:
                            id_to_label[sid] = s.get("label", sid)
                    print(f"   Batch {batch_num}/{total_batches}: got {len(skill_items)} labels back (total so far: {len(id_to_label)})")

                matched = sum(1 for sid in unique_skill_ids if sid in id_to_label)
                unmatched = len(unique_skill_ids) - matched
                print(f"URI matching done — matched: {matched}, unmatched (kept as-is): {unmatched}")

            except Exception as e:
                print(f"Skill label lookup failed: {type(e).__name__}: {e} — using raw URIs as fallback.")
                id_to_label = {sid: sid for sid in unique_skill_ids}
        else:
            print("No ESCO skill URIs detected — skipping label resolution.")

        # === 9️⃣ Replace URIs with labels in jobs ===
        for job in all_jobs:
            job["skills"] = [id_to_label.get(s, s) for s in job.get("skills", [])]

        # === 🔟 Check total jobs in tracker (for context) ===
        try:
            total_tracker_jobs = get_total_jobs_in_tracker()
            print(f"📦 Total jobs in tracker (unfiltered): {total_tracker_jobs}")
        except Exception as e:
            print(f"⚠️ Could not fetch total tracker job count: {e}")
            total_tracker_jobs = None

        # === 1️⃣1️⃣ Warn if low job count ===
        warning_message = None
        if len(all_jobs) < 50:
            warning_message = f"⚠️ Low job count: only {len(all_jobs)} jobs found. Results may not be representative."
            print(warning_message)

        # === 1️⃣2️⃣ Run full Skill Ageing analysis ===
        print("🚀 Running Skill Ageing analysis...")
        analysis_result = run_skill_analysis_from_list(all_jobs)

        # === 1️⃣3️⃣ Enrich result with metadata ===
        analysis_result["filters_used"] = {
            "occupation_ids": occ_ids_list if occ_ids_list else None,
            "source": source,
            "min_upload_date": min_upload_date,
            "max_upload_date": max_upload_date,
        }
        analysis_result["summary"]["Jobs Retrieved"] = len(all_jobs)
        analysis_result["summary"]["Total Jobs Available"] = total_count
        analysis_result["summary"]["Pages Fetched"] = total_pages
        if total_tracker_jobs is not None:
            analysis_result["summary"]["Total Jobs in Tracker"] = total_tracker_jobs
        if warning_message:
            analysis_result["warning"] = warning_message

        # === 1️⃣4️⃣ Save result to cache ===
        print(f"💾 Saving results to cache: '{file_path}'...")
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(analysis_result, json_file, indent=4, ensure_ascii=False)
        print(f"✅ Results cached successfully to '{file_path}'.")

        return analysis_result

    except Exception as e:
        print(f"❌ ERROR in skill-ageing: {type(e).__name__}: {e}")
        return {"error": str(e)}


# ============================================================
#  All other endpoints below are unchanged from the original.
#  Paste them here as-is.
# ============================================================

# ============================================================
#  ✅ /skill-ageing-law-policy
#  Changes vs original:
#    1. Completed_Analyses caching (cache hit returns instantly)
#    2. Batch skill URI resolution (only found URIs, not all tracker skills)
#    3. Paginated law-policies fetch (auto all pages, retry logic)
#    4. Removed duplicate get_token() local redefinition
#    5. Consistent print monitoring
# ============================================================

@app.get("/skill-ageing-law-policy")
def analyze_law_policy_skills(
    keywords: Optional[str] = Query(None, description="Comma-separated keywords, e.g. AI,Data,Education"),
    max_publication_date: Optional[str] = Query(None, description="Max publication date YYYY-MM-DD"),
):
    API_URL = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
    USERNAME_ENV = os.getenv("TRACKER_USERNAME", "skillab_staff")
    PASSWORD_ENV = os.getenv("TRACKER_PASSWORD", "skillroadtrip00")

    # === 📁 Cache folder ===
    folder = Path("Completed_Analyses")
    if not folder.exists():
        folder.mkdir(parents=True)
        print(f"📁 Folder '{folder}' created.")
    else:
        print(f"📁 Folder '{folder}' already exists, moving on.")

    # === 🏷️ Build cache filename ===
    keywords_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
    filename = "completed_analysis_skill_ageing_law_policy"
    for kw in keywords_list:
        filename += f"_{kw}"
    if max_publication_date:
        filename += f"_until{max_publication_date}"
    filename += ".json"

    file_path = folder / filename
    print(f"🗂️ Cache file path: {file_path}")

    if file_path.exists():
        print(f"✅ Cache hit — loading from '{file_path}'.")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.loads(f.read())

    print("🌐 No cache found — running full analysis from API...")

    try:
        # === 1️⃣ Authenticate ===
        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API_URL}/login", json={"username": USERNAME_ENV, "password": PASSWORD_ENV}, timeout=15)
        res.raise_for_status()
        token = res.text.replace('"', "")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        print("✅ Authenticated successfully.")
        print(f"📡 Keywords filter: {keywords_list if keywords_list else '(none)'}")
        print(f"📅 Max publication date: {max_publication_date or '(none)'}")

        # === 2️⃣ Retry helper ===
        page_size = 100
        REQUEST_TIMEOUT = 180
        MAX_RETRIES = 3
        RETRY_BACKOFF = 10

        def fetch_page_with_retry(page_num: int) -> dict:
            url = f"{API_URL}/law-policies?page={page_num}&page_size={page_size}"
            form_data = [("keywords_logic", "or")]
            for kw in keywords_list:
                form_data.append(("keywords", kw))
            if max_publication_date:
                form_data.append(("max_publication_date", max_publication_date))
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"   ↪ Attempt {attempt}/{MAX_RETRIES} for page {page_num}...")
                    r = requests.post(url, headers=headers, data=form_data, timeout=REQUEST_TIMEOUT)
                    if r.status_code != 200:
                        print(f"   ⚠️ HTTP {r.status_code}: {r.text[:300]}")
                        return {}
                    return r.json()
                except requests.exceptions.ReadTimeout:
                    print(f"   ⏱️ Timeout page {page_num}, attempt {attempt}/{MAX_RETRIES}.")
                    if attempt < MAX_RETRIES:
                        print(f"   🔄 Retrying in {RETRY_BACKOFF}s...")
                        time.sleep(RETRY_BACKOFF)
                    else:
                        print(f"   ❌ All retries exhausted for page {page_num}.")
                        return {}
                except Exception as ex:
                    print(f"   ❌ {type(ex).__name__}: {ex}")
                    return {}

        # === 3️⃣ Probe page 1 ===
        print("🔍 Probing page 1 to determine total count...")
        probe_data = fetch_page_with_retry(1)
        if not probe_data:
            return {"error": "❌ Probe request failed after all retries."}

        total_count = probe_data.get("count", 0)
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
        print(f"📊 Total records: {total_count} → {total_pages} page(s)")

        if total_count == 0:
            return {"message": "No law/policy documents found for the given filters."}

        # === 4️⃣ Paginate all pages ===
        all_docs = list(probe_data.get("items", []))
        print(f"📦 Page 1/{total_pages}: {len(all_docs)} docs from probe.")

        for page in range(2, total_pages + 1):
            print(f"📄 Fetching page {page}/{total_pages}...")
            data = fetch_page_with_retry(page)
            if not data:
                print(f"⚠️ Page {page} failed — stopping early.")
                break
            items = data.get("items", [])
            print(f"📦 Page {page}/{total_pages}: {len(items)} docs (running total: {len(all_docs) + len(items)})")
            if not items:
                break
            all_docs.extend(items)
            if len(items) < page_size:
                print("✅ Last page reached.")
                break

        print(f"🎯 Total docs retrieved: {len(all_docs)} / {total_count}")

        # === 5️⃣ Collect unique skill URIs ===
        unique_skill_ids = set()
        for doc in all_docs:
            for sid in (doc.get("skills") or doc.get("skill_ids") or []):
                if isinstance(sid, str) and sid.startswith("http"):
                    unique_skill_ids.add(sid)

        print(f"📚 Found {len(unique_skill_ids)} unique skill URIs — resolving in batches...")

        # === 6️⃣ Batch resolve only found URIs ===
        id_to_label = {}
        if unique_skill_ids:
            try:
                uri_list = list(unique_skill_ids)
                batch_size_skills = 50
                total_batches = math.ceil(len(uri_list) / batch_size_skills)
                for batch_num, start in enumerate(range(0, len(uri_list), batch_size_skills), 1):
                    batch = uri_list[start:start + batch_size_skills]
                    skill_payload = [("ids", sid) for sid in batch]
                    print(f"   Batch {batch_num}/{total_batches}: resolving {len(batch)} URIs...")
                    skill_res = requests.post(
                        f"{API_URL}/skills",
                        headers={"Authorization": f"Bearer {token}"},
                        data=skill_payload,
                        timeout=60
                    )
                    skill_res.raise_for_status()
                    for s in skill_res.json().get("items", []):
                        sid = s.get("id", "")
                        if sid:
                            id_to_label[sid] = s.get("label", sid)
                    print(f"   Batch {batch_num}/{total_batches}: resolved so far: {len(id_to_label)}")
                matched = sum(1 for sid in unique_skill_ids if sid in id_to_label)
                print(f"🔗 Matched: {matched}/{len(unique_skill_ids)} URIs")
            except Exception as e:
                print(f"⚠️ Skill label lookup failed: {type(e).__name__}: {e} — using raw URIs.")
                id_to_label = {sid: sid for sid in unique_skill_ids}

        # === 7️⃣ Build analysis-ready items ===
        all_items = []
        for doc in all_docs:
            pub_date = doc.get("publication_date") or doc.get("date")
            skills = doc.get("skills") or doc.get("skill_ids") or []
            skills = [id_to_label.get(s, s) for s in skills]
            if pub_date and skills:
                all_items.append({"upload_date": str(pub_date).split("T")[0], "skills": skills})

        print(f"🧩 Valid docs with skills: {len(all_items)} / {len(all_docs)}")

        if not all_items:
            return {"warning": "No valid policy records with skills found."}

        if len(all_items) < 50:
            print(f"⚠️ Low doc count: {len(all_items)} — results may not be representative.")

        # === 8️⃣ Run analysis ===
        print("🚀 Running Skill Ageing analysis...")
        result = run_skill_analysis_from_list(all_items)
        result["filters_used"] = {"keywords": keywords_list, "max_publication_date": max_publication_date}
        result["summary"]["Docs Retrieved"] = len(all_docs)
        result["summary"]["Total Docs Available"] = total_count
        result["summary"]["Docs with Skills"] = len(all_items)

        # === 9️⃣ Cache result ===
        print(f"💾 Saving to cache: '{file_path}'...")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"✅ Cached successfully.")
        return result

    except Exception as e:
        print(f"❌ ERROR in skill-ageing-law-policy: {type(e).__name__}: {e}")
        return {"error": str(e)}


# ============================================================
#  ✅ /skill-ageing-courses
#  Changes vs original:
#    1. Completed_Analyses caching
#    2. Batch skill URI resolution (only found URIs)
#    3. Auto-paginate all course pages with retry
#    4. Removed chunked_iterable helper (now uses simple batch loop)
#    5. Consistent print monitoring
# ============================================================

@app.get("/skill-ageing-courses")
def analyze_course_skills(
    keywords: Optional[str] = Query(None, description="Keywords to filter courses"),
    min_creation_date: Optional[str] = Query(None, description="Minimum creation date (YYYY-MM-DD)"),
    max_creation_date: Optional[str] = Query(None, description="Maximum creation date (YYYY-MM-DD)"),
):
    API_URL = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
    USERNAME_ENV = os.getenv("TRACKER_USERNAME", "skillab_staff")
    PASSWORD_ENV = os.getenv("TRACKER_PASSWORD", "skillroadtrip00")

    # === 📁 Cache folder ===
    folder = Path("Completed_Analyses")
    if not folder.exists():
        folder.mkdir(parents=True)
        print(f"📁 Folder '{folder}' created.")
    else:
        print(f"📁 Folder '{folder}' already exists, moving on.")

    # === 🏷️ Build cache filename ===
    filename = "completed_analysis_skill_ageing_courses"
    if keywords:
        for kw in [k.strip() for k in keywords.split(",") if k.strip()]:
            filename += f"_{kw}"
    if min_creation_date:
        filename += f"_from{min_creation_date}"
    if max_creation_date:
        filename += f"_to{max_creation_date}"
    filename += ".json"

    file_path = folder / filename
    print(f"🗂️ Cache file path: {file_path}")

    if file_path.exists():
        print(f"✅ Cache hit — loading from '{file_path}'.")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.loads(f.read())

    print("🌐 No cache found — running full analysis from API...")

    try:
        # === 1️⃣ Authenticate ===
        print("🔐 Authenticating with Tracker...")
        res = requests.post(f"{API_URL}/login", json={"username": USERNAME_ENV, "password": PASSWORD_ENV}, timeout=15)
        res.raise_for_status()
        token = res.text.replace('"', "")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        print("✅ Authenticated successfully.")
        print(f"📡 Keywords filter: {keywords or '(none)'}")

        # === 2️⃣ Retry helper ===
        page_size = 100
        REQUEST_TIMEOUT = 180
        MAX_RETRIES = 3
        RETRY_BACKOFF = 10

        def fetch_page_with_retry(page_num: int) -> dict:
            url = f"{API_URL}/courses?page={page_num}&page_size={page_size}"
            form_data = [("keywords_logic", "or")]
            if keywords:
                for kw in [k.strip() for k in keywords.split(",") if k.strip()]:
                    form_data.append(("keywords", kw))
            if min_creation_date:
                form_data.append(("min_creation_date", min_creation_date))
            if max_creation_date:
                form_data.append(("max_creation_date", max_creation_date))
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"   ↪ Attempt {attempt}/{MAX_RETRIES} for page {page_num}...")
                    r = requests.post(url, headers=headers, data=form_data, timeout=REQUEST_TIMEOUT)
                    if r.status_code != 200:
                        print(f"   ⚠️ HTTP {r.status_code}: {r.text[:300]}")
                        return {}
                    return r.json()
                except requests.exceptions.ReadTimeout:
                    print(f"   ⏱️ Timeout page {page_num}, attempt {attempt}/{MAX_RETRIES}.")
                    if attempt < MAX_RETRIES:
                        print(f"   🔄 Retrying in {RETRY_BACKOFF}s...")
                        time.sleep(RETRY_BACKOFF)
                    else:
                        print(f"   ❌ All retries exhausted for page {page_num}.")
                        return {}
                except Exception as ex:
                    print(f"   ❌ {type(ex).__name__}: {ex}")
                    return {}

        # === 3️⃣ Probe page 1 ===
        print("🔍 Probing page 1 to determine total count...")
        probe_data = fetch_page_with_retry(1)
        if not probe_data:
            return {"error": "❌ Probe request failed after all retries."}

        total_count = probe_data.get("count", 0)
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
        print(f"📊 Total records: {total_count} → {total_pages} page(s)")

        if total_count == 0:
            return {"message": "No courses found for the given filters."}

        # === 4️⃣ Paginate all pages ===
        all_courses = list(probe_data.get("items", []))
        print(f"📦 Page 1/{total_pages}: {len(all_courses)} courses from probe.")

        for page in range(2, total_pages + 1):
            print(f"📄 Fetching page {page}/{total_pages}...")
            data = fetch_page_with_retry(page)
            if not data:
                print(f"⚠️ Page {page} failed — stopping early.")
                break
            items = data.get("items", [])
            print(f"📦 Page {page}/{total_pages}: {len(items)} courses (running total: {len(all_courses) + len(items)})")
            if not items:
                break
            all_courses.extend(items)
            if len(items) < page_size:
                print("✅ Last page reached.")
                break

        print(f"🎯 Total courses retrieved: {len(all_courses)} / {total_count}")

        # === 5️⃣ Collect unique skill URIs ===
        unique_skill_ids = set()
        for c in all_courses:
            for sid in (c.get("skills") or c.get("skill_ids") or []):
                if isinstance(sid, str) and sid.startswith("http"):
                    unique_skill_ids.add(sid)

        print(f"📚 Found {len(unique_skill_ids)} unique skill URIs — resolving in batches...")

        # === 6️⃣ Batch resolve only found URIs ===
        id_to_label = {}
        if unique_skill_ids:
            try:
                uri_list = list(unique_skill_ids)
                batch_size_skills = 50
                total_batches = math.ceil(len(uri_list) / batch_size_skills)
                for batch_num, start in enumerate(range(0, len(uri_list), batch_size_skills), 1):
                    batch = uri_list[start:start + batch_size_skills]
                    skill_payload = [("ids", sid) for sid in batch]
                    print(f"   Batch {batch_num}/{total_batches}: resolving {len(batch)} URIs...")
                    skill_res = requests.post(
                        f"{API_URL}/skills",
                        headers={"Authorization": f"Bearer {token}"},
                        data=skill_payload,
                        timeout=60
                    )
                    skill_res.raise_for_status()
                    for s in skill_res.json().get("items", []):
                        sid = s.get("id", "")
                        if sid:
                            id_to_label[sid] = s.get("label", sid)
                    print(f"   Batch {batch_num}/{total_batches}: resolved so far: {len(id_to_label)}")
                matched = sum(1 for sid in unique_skill_ids if sid in id_to_label)
                print(f"🔗 Matched: {matched}/{len(unique_skill_ids)} URIs")
            except Exception as e:
                print(f"⚠️ Skill label lookup failed: {type(e).__name__}: {e} — using raw URIs.")
                id_to_label = {sid: sid for sid in unique_skill_ids}

        # === 7️⃣ Build analysis-ready items ===
        all_items = []
        for c in all_courses:
            upload_date = (
                c.get("last_updated") or c.get("creation_date")
                or c.get("date") or c.get("created_at")
            )
            if upload_date:
                upload_date = str(upload_date).split("T")[0]
            skills = c.get("skills") or c.get("skill_ids") or []
            skills = [id_to_label.get(s, s) for s in skills if s]
            if upload_date and skills:
                all_items.append({"upload_date": upload_date, "skills": skills})

        print(f"🧩 Valid courses with skills: {len(all_items)} / {len(all_courses)}")

        if not all_items:
            return {"warning": "No valid courses with skills found."}

        if len(all_items) < 50:
            print(f"⚠️ Low course count: {len(all_items)} — results may not be representative.")

        # === 8️⃣ Run analysis ===
        print("🚀 Running Skill Ageing analysis...")
        result = run_skill_analysis_from_list(all_items)
        result["filters_used"] = {
            "keywords": keywords,
            "min_creation_date": min_creation_date,
            "max_creation_date": max_creation_date
        }
        result["summary"]["Courses Retrieved"] = len(all_courses)
        result["summary"]["Total Courses Available"] = total_count
        result["summary"]["Courses with Skills"] = len(all_items)

        # === 9️⃣ Cache result ===
        print(f"💾 Saving to cache: '{file_path}'...")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"✅ Cached successfully.")
        return result

    except Exception as e:
        print(f"❌ ERROR in skill-ageing-courses: {type(e).__name__}: {e}")
        return {"error": str(e)}


# ============================================================
#  ✅ /ku-skill-ageing
#  Changes vs original:
#    1. Completed_Analyses caching
#    2. Removed duplicate endpoint definition (original had two!)
#    3. Kept the better debug version, removed the simpler one
#    4. Consistent print monitoring
#    5. KUs don't have ESCO URIs so no skill resolution needed
# ============================================================

@app.get("/ku-skill-ageing")
def analyze_ku_skills(
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM format"),
    kus: Optional[str] = Query(None, description="Comma-separated KU IDs, e.g. K1,K5,K10"),
    organization: Optional[str] = Query(None, description="Optional organization name to filter by"),
):
    from collections import Counter

    BASE_URL = "https://portal.skillab-project.eu/ku-detection"
    api_url = f"{BASE_URL}/analysis_results"

    # === 📁 Cache folder ===
    folder = Path("Completed_Analyses")
    if not folder.exists():
        folder.mkdir(parents=True)
        print(f"📁 Folder '{folder}' created.")
    else:
        print(f"📁 Folder '{folder}' already exists, moving on.")

    # === 🏷️ Build cache filename ===
    filename = "completed_analysis_ku_skill_ageing"
    if organization:
        filename += f"_{organization.replace(' ', '_')}"
    if kus:
        for ku in [k.strip() for k in kus.split(",") if k.strip()]:
            filename += f"_{ku}"
    if start_date:
        filename += f"_from{start_date}"
    if end_date:
        filename += f"_to{end_date}"
    filename += ".json"

    file_path = folder / filename
    print(f"🗂️ Cache file path: {file_path}")

    if file_path.exists():
        print(f"✅ Cache hit — loading from '{file_path}'.")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.loads(f.read())

    print("🌐 No cache found — running full analysis from API...")

    try:
        # === 1️⃣ Build query params and fetch ===
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if organization:
            params["organization"] = organization

        print(f"🔗 Fetching KU data from: {api_url}")
        print(f"📦 Params: {params}")

        response = requests.get(api_url, params=params, headers={"Accept": "application/json"}, timeout=60)
        print(f"📥 HTTP Status: {response.status_code}")

        try:
            ku_data = response.json()
        except Exception as je:
            print(f"💥 JSON parse error: {je}")
            return {"error": f"Invalid JSON from KU API: {str(je)}"}

        # Handle wrapped response
        if isinstance(ku_data, dict) and "items" in ku_data:
            ku_data = ku_data["items"]

        if not isinstance(ku_data, list) or not ku_data:
            print("⚠️ Empty or invalid KU dataset returned.")
            return {"warning": "No KU analysis data found for the given filters."}

        print(f"✅ Retrieved {len(ku_data)} KU records")

        # === 2️⃣ Transform KU records ===
        selected_kus = set(kus.split(",")) if kus else None
        all_items = []

        for record in ku_data:
            upload_date = record.get("timestamp", "").split("T")[0]
            detected_kus = record.get("detected_kus", {})
            record_org = record.get("organization", "Unknown")

            if organization and record_org.lower() != organization.lower():
                continue

            active_kus = [ku for ku, val in detected_kus.items() if str(val) == "1"]
            if selected_kus:
                active_kus = [ku for ku in active_kus if ku in selected_kus]

            if upload_date and active_kus:
                all_items.append({
                    "upload_date": upload_date,
                    "organization": record_org,
                    "skills": active_kus
                })

        print(f"📊 Valid KU records after filtering: {len(all_items)} / {len(ku_data)}")

        if not all_items:
            return {"warning": "No KU records matched the selected filters."}

        # === 3️⃣ Frequency summary ===
        ku_counter = Counter()
        for item in all_items:
            ku_counter.update(item["skills"])
        print(f"📈 KU frequency (top 10): {ku_counter.most_common(10)}")

        if len(all_items) < 50:
            print(f"⚠️ Low record count: {len(all_items)} — results may not be representative.")

        # === 4️⃣ Run analysis ===
        # Note: KU labels are already human-readable (K1, K2 etc) — no URI resolution needed
        print("🚀 Running Skill Ageing analysis on KU data...")
        result = run_skill_analysis_from_list(all_items)
        result["filters_used"] = {
            "start_date": start_date,
            "end_date": end_date,
            "kus": kus,
            "organization": organization
        }
        result["summary"]["KU Records Retrieved"] = len(all_items)
        result["summary"]["Total KU Records from API"] = len(ku_data)

        # === 5️⃣ Cache result ===
        print(f"💾 Saving to cache: '{file_path}'...")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"✅ Cached successfully.")
        return result

    except requests.exceptions.RequestException as re:
        print(f"🌐 Network error: {re}")
        return {"error": f"Network issue contacting KU API: {str(re)}"}
    except Exception as e:
        print(f"❌ ERROR in ku-skill-ageing: {type(e).__name__}: {e}")
        return {"error": f"KU skill analysis failed: {str(e)}"}


@app.get("/ku-debug")
def ku_debug(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM"),
    organization: Optional[str] = Query(None, description="Filter results by organization name"),
    x_organization: str = Query(..., description="Required: your organization name for the API header (X-User-Organization)"),
):
    from collections import Counter

    BASE_URL = "https://portal.skillab-project.eu/ku-detection"
    api_url = f"{BASE_URL}/analysis_results"

    params = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if organization:
        params["organization"] = organization

    print(f"🔍 [ku-debug] Fetching from {api_url} with params {params}")
    print(f"🔑 X-User-Organization header: {x_organization}")

    try:
        response = requests.get(api_url, params=params, headers={"Accept": "application/json", "X-User-Organization": x_organization}, timeout=60)
        print(f"📥 HTTP Status: {response.status_code}")
        print(f"📥 Response size: {len(response.text)} chars")
        print(f"📥 Raw response: {response.text[:300]}")

        # Return full error detail if not 2xx
        if response.status_code >= 400:
            return {
                "error": f"API returned HTTP {response.status_code}",
                "url_called": str(response.url),
                "params_sent": params,
                "response_body": response.text,
                "hint": "The KU API may require specific params, auth, or a different method (GET vs POST)."
            }

        try:
            raw = response.json()
        except Exception as je:
            return {
                "error": f"JSON parse failed: {je}",
                "raw_text_snippet": response.text[:500]
            }

        # Unwrap if needed
        if isinstance(raw, dict) and "items" in raw:
            records = raw["items"]
            top_level_keys = list(raw.keys())
            top_level_count = raw.get("count", None)
        elif isinstance(raw, list):
            records = raw
            top_level_keys = ["(root is a list)"]
            top_level_count = None
        else:
            return {
                "error": "Unexpected response structure",
                "type": str(type(raw)),
                "snippet": str(raw)[:500]
            }

        total_records = len(records)
        print(f"📊 Total records in response: {total_records}")

        if total_records == 0:
            return {
                "status": "⚠️ API returned 0 records for these filters",
                "params_used": params,
                "top_level_keys": top_level_keys,
                "top_level_count_field": top_level_count,
            }

        # === Inspect first record structure ===
        first_record = records[0]
        first_record_keys = list(first_record.keys())

        # === Organizations ===
        org_counter = Counter(r.get("organization", "Unknown") for r in records)

        # === Timestamps ===
        timestamps = [r.get("timestamp", "") for r in records if r.get("timestamp")]
        timestamps_sorted = sorted(timestamps)
        earliest = timestamps_sorted[0] if timestamps_sorted else "N/A"
        latest = timestamps_sorted[-1] if timestamps_sorted else "N/A"

        # === KU activity ===
        ku_counter = Counter()
        records_with_active_kus = 0
        records_with_no_active_kus = 0

        for r in records:
            detected = r.get("detected_kus", {})
            active = [ku for ku, val in detected.items() if str(val) == "1"]
            if active:
                records_with_active_kus += 1
                ku_counter.update(active)
            else:
                records_with_no_active_kus += 1

        # === Sample records (first 3, trimmed) ===
        sample_records = []
        for r in records[:3]:
            detected = r.get("detected_kus", {})
            active_kus = [ku for ku, val in detected.items() if str(val) == "1"]
            sample_records.append({
                "organization": r.get("organization", "Unknown"),
                "timestamp": r.get("timestamp", "N/A"),
                "total_ku_fields": len(detected),
                "active_kus": active_kus,
                "active_ku_count": len(active_kus),
            })

        result = {
            "status": "✅ KU API reachable and returning data",
            "params_used": params,
            "response_structure": {
                "top_level_keys": top_level_keys,
                "top_level_count_field": top_level_count,
                "first_record_keys": first_record_keys,
            },
            "counts": {
                "total_records": total_records,
                "records_with_active_kus": records_with_active_kus,
                "records_with_no_active_kus": records_with_no_active_kus,
                "unique_organizations": len(org_counter),
                "unique_active_kus": len(ku_counter),
            },
            "date_range": {
                "earliest_timestamp": earliest,
                "latest_timestamp": latest,
            },
            "organizations": dict(org_counter.most_common()),
            "ku_frequency_top20": dict(ku_counter.most_common(20)),
            "sample_records": sample_records,
        }

        print(f"✅ [ku-debug] Done — {total_records} records, {len(org_counter)} orgs, {len(ku_counter)} unique KUs")
        return result

    except requests.exceptions.RequestException as e:
        print(f"🌐 Network error: {e}")
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        print(f"❌ ERROR in ku-debug: {type(e).__name__}: {e}")
        return {"error": str(e)}

