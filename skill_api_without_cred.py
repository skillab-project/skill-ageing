from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import requests
import math
import re
import time
from datetime import datetime
from collections import defaultdict
from itertools import combinations
from sklearn.linear_model import LinearRegression
import json
import uuid
from pathlib import Path
import os
from dotenv import load_dotenv
from typing import Optional
from ageing_forecasting import router as forecasting_router


# === Load environment variables ===
load_dotenv()

app = FastAPI(
    title="Skill Ageing API",
    root_path="/skill-ageing"
)
app.include_router(forecasting_router)

print("🧩 TRACKER_API =", os.getenv("TRACKER_API"))
print("🧩 TRACKER_USERNAME =", os.getenv("TRACKER_USERNAME"))
print("🧩 TRACKER_PASSWORD =", "*****")

API        = os.getenv("TRACKER_API")
USERNAME   = os.getenv("TRACKER_USERNAME")
PASSWORD   = os.getenv("TRACKER_PASSWORD")
KU_API_URL = os.getenv("KU_API_URL")


# ============================================================
#  SHARED HELPERS
# ============================================================

def get_token():
    res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
    res.raise_for_status()
    return res.text.replace('"', "")


def get_total_jobs_in_tracker():
    token = get_token()
    res = requests.post(f"{API}/jobs", headers={"Authorization": f"Bearer {token}"}, data={}, timeout=30)
    res.raise_for_status()
    return res.json().get("count", 0)


def _ensure_cache() -> Path:
    p = Path("Completed_Analyses")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_cache(file_path: Path):
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            print(f"✅ Cache hit — loaded from '{file_path}'.")
            return data
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ Cache corrupted ({e}) — deleting and re-running...")
            file_path.unlink()
    return None


def _save_cache(file_path: Path, result: dict):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"💾 Cached → '{file_path}'.")


# ============================================================
#  SKILL ANALYSIS CORE  ← THE FIXED VERSION
#
#  Key optimisations vs the slow original:
#
#  1. Competing Skills
#     OLD: combinations(ALL filtered_tags, 2)  — up to 125k pairs
#     NEW: only top-50 skills by frequency      — at most 1,225 pairs
#          + early-exit: skip pairs with <5 co-occurrence months
#          + vectorised correlation via np.corrcoef on the full matrix
#
#  2. Inverse Trends
#     OLD: combinations(top-100 skills, 2)      — 4,950 pairs
#     NEW: top-30 skills                        — 435 pairs max
#
#  3. Rapid Obsolescence
#     No change needed — O(n) loop, already fast.
#
#  4. Epidemiological Metrics
#     No change needed — O(n) loop, already fast.
#
#  Everything else (biology_summary, time series) unchanged.
# ============================================================

def run_skill_analysis_from_list(job_list):
    print(f"🔬 run_skill_analysis_from_list: {len(job_list)} items")

    skill_occurrences = defaultdict(list)
    for job in job_list:
        try:
            date_str = job.get("upload_date")
            if not date_str:
                continue
            date = datetime.strptime(str(date_str).split("T")[0], "%Y-%m-%d")
            skills = job.get("skills", [])
            if not isinstance(skills, list):
                continue
            for skill in skills:
                skill_occurrences[skill].append(date)
        except Exception:
            continue

    print(f"📊 Unique skills found: {len(skill_occurrences)}")

    combined_index = pd.date_range(start="2020-01-01", end="2025-12-31", freq="ME")

    def get_slope(ts):
        if len(ts) < 3:
            return 0.0
        X = np.arange(len(ts)).reshape(-1, 1)
        y = ts.values.reshape(-1, 1)
        return LinearRegression().fit(X, y).coef_[0][0]

    # ── Biology summary ──────────────────────────────────────
    print("🧬 Computing skill biology summary...")
    biology_summary = []
    for skill, dates in skill_occurrences.items():
        df_s = pd.DataFrame(dates, columns=["date"])
        df_s["year_month"] = df_s["date"].dt.to_period("M")
        birth     = df_s["date"].min()
        peak      = df_s["year_month"].value_counts().idxmax()
        total_jobs = len(df_s)
        recent_use = df_s["date"].max().year > 2022
        immunity  = "High" if total_jobs > 20 and recent_use else "Low"

        s     = pd.Series(1, index=pd.to_datetime(dates)).resample("ME").sum().reindex(combined_index, fill_value=0)
        slope = get_slope(s)
        trend = "Declining" if slope < -0.01 else ("Rising" if slope > 0.01 else "Stable")

        biology_summary.append({
            "Skill": skill,
            "Date of Birth": birth.strftime("%Y-%m-%d"),
            "Peak Activity Date": str(peak),
            "Total Jobs": total_jobs,
            "Immunity Score": immunity,
            "Trend": trend,
            "Slope": round(slope, 4)
        })
    print(f"   → {len(biology_summary)} skills in biology summary")

    # ── Time series matrix ───────────────────────────────────
    print("📈 Building time series matrix...")
    tag_series = {}
    for skill, dates in skill_occurrences.items():
        s = pd.Series(1, index=pd.to_datetime(dates)).resample("ME").sum().reindex(combined_index, fill_value=0)
        tag_series[skill] = s

    all_tags_df   = pd.DataFrame(tag_series).fillna(0)
    # Keep skills with at least 10 total occurrences
    filtered_tags = [t for t in all_tags_df.columns if all_tags_df[t].sum() >= 10]
    print(f"   → {len(filtered_tags)} skills pass the ≥10 threshold")

    # ── Competing Skills  ────────────────────────────────────
    # FIX: limit to top-50 by total frequency → max 1,225 pairs instead of ~125k
    print("⚔️  Computing competing skills (top-50, max 1,225 pairs)...")
    top50 = sorted(filtered_tags, key=lambda x: all_tags_df[x].sum(), reverse=True)[:50]
    competing_results = []

    if len(top50) >= 2:
        mat   = all_tags_df[top50].values          # shape (n_months, 50)
        corr_matrix = np.corrcoef(mat.T)           # (50, 50) — vectorised, instant

        for i, tag1 in enumerate(top50):
            for j, tag2 in enumerate(top50):
                if j <= i:
                    continue
                s1, s2  = mat[:, i], mat[:, j]
                overlap = (s1 > 0) & (s2 > 0)
                if overlap.sum() < 5:
                    continue
                corr = corr_matrix[i, j]
                if np.isfinite(corr) and corr < -0.5:
                    competing_results.append({
                        "Skill A": tag1, "Skill B": tag2,
                        "Correlation": round(float(corr), 3)
                    })
    print(f"   → {len(competing_results)} competing pairs found")

    # ── Inverse Trends  ─────────────────────────────────────
    # FIX: limit to top-30 → max 435 pairs instead of 4,950
    print("📉 Computing inverse trends (top-30, max 435 pairs)...")
    top30 = sorted(filtered_tags, key=lambda x: all_tags_df[x].sum(), reverse=True)[:30]
    inverse_results = []

    for tag1, tag2 in combinations(top30, 2):
        s1, s2 = all_tags_df[tag1], all_tags_df[tag2]
        mask   = (s1 > 0) & (s2 > 0)
        if mask.sum() < 6:
            continue
        slope1 = get_slope(s1[mask])
        slope2 = get_slope(s2[mask])
        if slope1 < -0.005 and slope2 > 0.005:
            inverse_results.append({
                "Declining Skill": tag1, "Competing Skill": tag2,
                "Slope A": round(slope1, 4), "Slope B": round(slope2, 4),
                "Overlapping Months": int(mask.sum())
            })
    print(f"   → {len(inverse_results)} inverse-trend pairs found")

    # ── Rapid Obsolescence  ─────────────────────────────────
    print("💥 Computing rapid obsolescence...")
    rapid_drops = []
    for tag, series in tag_series.items():
        if (series > 0).sum() < 12:
            continue
        peak_value = series.max()
        if peak_value < 5:
            continue
        peak_idx  = series.idxmax()
        peak_loc  = series.index.get_loc(peak_idx)
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
    print(f"   → {len(rapid_drops)} rapidly obsolete skills")

    # ── Epidemiological Metrics  ────────────────────────────
    print("🦠 Computing epidemiological metrics...")
    epi_metrics = []
    shock_start, shock_end = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")
    old_start,   old_end   = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")

    for tag, series in tag_series.items():
        series      = series.fillna(0)
        total_jobs  = series.sum()
        if total_jobs == 0:
            continue
        incidence     = series.loc[shock_start:shock_end].sum()
        old_incidence = series.loc[old_start:old_end].sum()
        pct_change    = (100 * (incidence - old_incidence) / old_incidence
                         if old_incidence > 0 else (999 if incidence > 0 else 0))
        ip_ratio        = incidence / total_jobs if total_jobs else 0
        recent_activity = series[series.index >= datetime(2023, 7, 1)].sum()
        is_dead         = recent_activity == 0
        revival         = "Yes" if old_incidence < incidence and old_incidence > 0 else "No"
        mortality_ratio = (incidence / (total_jobs - incidence)
                           if (total_jobs - incidence) > 0 else 999)
        was_active  = old_incidence > 0 or incidence > 0
        cfr         = 1.0 if (was_active and is_dead) else 0.0
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
    print(f"   → {len(epi_metrics)} skills with epi metrics")

    output = {
        "skill_biology_summary": biology_summary,
        "competing_skills":      competing_results,
        "inverse_trends":        inverse_results,
        "rapid_obsolescence":    rapid_drops,
        "epidemiological_metrics": epi_metrics,
        "total_jobs_analyzed":   len(job_list)
    }

    Path("results").mkdir(parents=True, exist_ok=True)
    filename = f"results/skill_analysis_{uuid.uuid4().hex[:6]}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Analysis complete — saved to {filename}")
    return {
        "message":    "✅ Skill analysis complete",
        "file_saved": filename,
        "summary": {
            "Total Skills Found":        len(biology_summary),
            "Competing Skill Pairs":     len(competing_results),
            "Inverse Trend Pairs":       len(inverse_results),
            "Rapidly Obsolete Skills":   len(rapid_drops),
            "Epidemiological Metrics":   len(epi_metrics),
        },
        "data": output
    }


# ============================================================
#  /skill-ageing
# ============================================================

@app.get("/skill-ageing-jobs")
def analyze_skills(
    occupation_ids:  Optional[str] = Query(None, description="Comma-separated occupation IDs"),
    source:          Optional[str] = Query(None, description="Optional source filter (e.g. linkedin)"),
    min_upload_date: Optional[str] = Query(None, description="Minimum upload date YYYY-MM-DD"),
    max_upload_date: Optional[str] = Query(None, description="Maximum upload date YYYY-MM-DD"),
):
    folder        = _ensure_cache()
    occ_ids_list  = [o.strip() for o in occupation_ids.split(",") if o.strip()] if occupation_ids else []

    filename = "completed_analysis_skill_ageing"
    for occ in occ_ids_list:
        match = re.search(r'C\d+$', occ)
        filename += f"_{match.group(0)}" if match else f"_{occ.replace('/', '_').replace(':', '').replace('.', '')}"
    if source:          filename += f"_{source}"
    if min_upload_date: filename += f"_from{min_upload_date}"
    if max_upload_date: filename += f"_to{max_upload_date}"
    filename += ".json"

    file_path = folder / filename
    print(f"🗂️ Cache path: {file_path}")
    cached = _load_cache(file_path)
    if cached:
        return cached

    try:
        # 1️⃣ Authenticate
        print("🔐 Authenticating...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token   = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
        print("✅ Authenticated.")
        print(f"🏢 Occupation IDs : {occ_ids_list or '(none)'}")
        print(f"🗂️ Source          : {source or '(none)'}")
        print(f"📅 Date range      : {min_upload_date or '*'} → {max_upload_date or '*'}")

        # 2️⃣ Form builder
        def build_form():
            fd = [("keywords_logic", "or"), ("skill_ids_logic", "or"), ("occupation_ids_logic", "or")]
            for occ in occ_ids_list:
                fd.append(("occupation_ids", occ))
            if source:          fd.append(("sources", source))
            if min_upload_date: fd.append(("min_upload_date", min_upload_date))
            if max_upload_date: fd.append(("max_upload_date", max_upload_date))
            return fd

        # 3️⃣ Retry helper
        page_size = 100
        def fetch_page(page_num: int) -> dict:
            url = f"{API}/jobs?page={page_num}&page_size={page_size}"
            for attempt in range(1, 4):
                try:
                    print(f"   ↪ Attempt {attempt}/3 — page {page_num}...")
                    r = requests.post(url, headers=headers, data=build_form(), timeout=180)
                    if r.status_code != 200:
                        print(f"   ⚠️ HTTP {r.status_code}: {r.text[:200]}")
                        return {}
                    return r.json()
                except requests.exceptions.ReadTimeout:
                    print(f"   ⏱️ Timeout page {page_num}, attempt {attempt}.")
                    if attempt < 3: time.sleep(10)
                    else: return {}
                except Exception as ex:
                    print(f"   ❌ {type(ex).__name__}: {ex}")
                    return {}

        # 4️⃣ Probe + auto-paginate
        print("🔍 Probing page 1...")
        probe = fetch_page(1)
        if not probe:
            return {"error": "❌ Probe failed after all retries."}

        total_count = probe.get("count", 0)
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
        print(f"📊 Total jobs: {total_count} → {total_pages} page(s)")

        if total_count == 0:
            return {"message": "No job postings found for the given filters."}

        all_jobs = list(probe.get("items", []))
        print(f"📦 Page 1/{total_pages}: {len(all_jobs)} jobs")

        for page in range(2, total_pages + 1):
            print(f"📄 Fetching page {page}/{total_pages}...")
            data  = fetch_page(page)
            items = data.get("items", []) if data else []
            print(f"📦 Page {page}/{total_pages}: {len(items)} jobs (running total: {len(all_jobs) + len(items)})")
            if not items: break
            all_jobs.extend(items)
            if len(items) < page_size:
                print("✅ Last page reached.")
                break

        print(f"🎯 Total jobs retrieved: {len(all_jobs)} / {total_count}")

        # 5️⃣ Collect unique skill URIs
        unique_skill_ids = set()
        for job in all_jobs:
            for sid in job.get("skills", []):
                if isinstance(sid, str) and sid.startswith("http"):
                    unique_skill_ids.add(sid)
        print(f"📚 {len(unique_skill_ids)} unique skill URIs found — resolving in batches of 50...")

        # 6️⃣ Batch resolve URIs → labels
        id_to_label = {}
        if unique_skill_ids:
            try:
                uri_list      = list(unique_skill_ids)
                total_batches = math.ceil(len(uri_list) / 50)
                for batch_num, start in enumerate(range(0, len(uri_list), 50), 1):
                    batch        = uri_list[start:start + 50]
                    skill_payload = [("ids", sid) for sid in batch]
                    print(f"   Batch {batch_num}/{total_batches}: {len(batch)} URIs...")
                    skill_res = requests.post(
                        f"{API}/skills",
                        headers={"Authorization": f"Bearer {token}"},
                        data=skill_payload, timeout=60
                    )
                    skill_res.raise_for_status()
                    for s in skill_res.json().get("items", []):
                        sid = s.get("id", "")
                        if sid: id_to_label[sid] = s.get("label", sid)
                    print(f"   Batch {batch_num}/{total_batches}: resolved so far: {len(id_to_label)}")
                matched   = sum(1 for sid in unique_skill_ids if sid in id_to_label)
                unmatched = len(unique_skill_ids) - matched
                print(f"🔗 Matched: {matched} | Unmatched (kept as-is): {unmatched}")
            except Exception as e:
                print(f"⚠️ Skill label lookup failed: {e} — using raw URIs.")
                id_to_label = {sid: sid for sid in unique_skill_ids}

        # 7️⃣ Replace URIs with labels
        for job in all_jobs:
            job["skills"] = [id_to_label.get(s, s) for s in job.get("skills", [])]

        # 8️⃣ Total jobs context
        try:
            total_tracker_jobs = get_total_jobs_in_tracker()
            print(f"📦 Total jobs in tracker (unfiltered): {total_tracker_jobs}")
        except Exception as e:
            print(f"⚠️ Could not fetch tracker total: {e}")
            total_tracker_jobs = None

        warning_message = None
        if len(all_jobs) < 50:
            warning_message = f"⚠️ Low job count: {len(all_jobs)} — results may not be representative."
            print(warning_message)

        # 9️⃣ Run analysis
        print("🚀 Running Skill Ageing analysis...")
        result = run_skill_analysis_from_list(all_jobs)

        result["filters_used"] = {
            "occupation_ids": occ_ids_list or None,
            "source": source,
            "min_upload_date": min_upload_date,
            "max_upload_date": max_upload_date,
        }
        result["summary"]["Jobs Retrieved"]       = len(all_jobs)
        result["summary"]["Total Jobs Available"] = total_count
        result["summary"]["Pages Fetched"]        = total_pages
        if total_tracker_jobs is not None:
            result["summary"]["Total Jobs in Tracker"] = total_tracker_jobs
        if warning_message:
            result["warning"] = warning_message

        _save_cache(file_path, result)
        return result

    except Exception as e:
        print(f"❌ ERROR in skill-ageing: {type(e).__name__}: {e}")
        return {"error": str(e)}


# ============================================================
#  /skill-ageing-law-policy
# ============================================================

@app.get("/skill-ageing-law-policy")
def analyze_law_policy_skills(
    keywords:             Optional[str] = Query(None, description="Comma-separated keywords, e.g. AI,Data"),
    max_publication_date: Optional[str] = Query(None, description="Max publication date YYYY-MM-DD"),
):
    folder           = _ensure_cache()
    keywords_list    = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []

    filename = "completed_analysis_skill_ageing_law_policy"
    for kw in keywords_list: filename += f"_{kw}"
    if max_publication_date: filename += f"_until{max_publication_date}"
    filename += ".json"

    file_path = folder / filename
    print(f"🗂️ Cache path: {file_path}")
    cached = _load_cache(file_path)
    if cached: return cached

    try:
        print("🔐 Authenticating...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token   = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
        print("✅ Authenticated.")
        print(f"📡 Keywords: {keywords_list or '(none)'} | Max date: {max_publication_date or '(none)'}")

        page_size = 100
        def fetch_page(page_num: int) -> dict:
            url       = f"{API}/law-policies?page={page_num}&page_size={page_size}"
            form_data = [("keywords_logic", "or")]
            for kw in keywords_list: form_data.append(("keywords", kw))
            if max_publication_date: form_data.append(("max_publication_date", max_publication_date))
            for attempt in range(1, 4):
                try:
                    print(f"   ↪ Attempt {attempt}/3 — page {page_num}...")
                    r = requests.post(url, headers=headers, data=form_data, timeout=180)
                    if r.status_code != 200: return {}
                    return r.json()
                except requests.exceptions.ReadTimeout:
                    if attempt < 3: time.sleep(10)
                    else: return {}
                except Exception as ex:
                    print(f"   ❌ {type(ex).__name__}: {ex}"); return {}

        print("🔍 Probing page 1...")
        probe = fetch_page(1)
        if not probe: return {"error": "❌ Probe failed."}

        total_count = probe.get("count", 0)
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
        print(f"📊 Total docs: {total_count} → {total_pages} page(s)")
        if total_count == 0: return {"message": "No law/policy documents found."}

        all_docs = list(probe.get("items", []))
        print(f"📦 Page 1/{total_pages}: {len(all_docs)} docs")

        for page in range(2, total_pages + 1):
            print(f"📄 Fetching page {page}/{total_pages}...")
            data  = fetch_page(page)
            items = data.get("items", []) if data else []
            print(f"📦 Page {page}/{total_pages}: {len(items)} docs (running total: {len(all_docs) + len(items)})")
            if not items: break
            all_docs.extend(items)
            if len(items) < page_size:
                print("✅ Last page reached."); break

        print(f"🎯 Total docs: {len(all_docs)} / {total_count}")

        unique_skill_ids = set()
        for doc in all_docs:
            for sid in (doc.get("skills") or doc.get("skill_ids") or []):
                if isinstance(sid, str) and sid.startswith("http"):
                    unique_skill_ids.add(sid)
        print(f"📚 {len(unique_skill_ids)} unique skill URIs — resolving in batches...")

        id_to_label = {}
        if unique_skill_ids:
            try:
                uri_list      = list(unique_skill_ids)
                total_batches = math.ceil(len(uri_list) / 50)
                for batch_num, start in enumerate(range(0, len(uri_list), 50), 1):
                    batch         = uri_list[start:start + 50]
                    skill_payload = [("ids", sid) for sid in batch]
                    print(f"   Batch {batch_num}/{total_batches}: {len(batch)} URIs...")
                    sr = requests.post(f"{API}/skills", headers={"Authorization": f"Bearer {token}"}, data=skill_payload, timeout=60)
                    sr.raise_for_status()
                    for s in sr.json().get("items", []):
                        sid = s.get("id", "")
                        if sid: id_to_label[sid] = s.get("label", sid)
                    print(f"   Resolved so far: {len(id_to_label)}")
            except Exception as e:
                print(f"⚠️ Skill lookup failed: {e}")
                id_to_label = {sid: sid for sid in unique_skill_ids}

        all_items = []
        for doc in all_docs:
            pub_date = doc.get("publication_date") or doc.get("date")
            skills   = [id_to_label.get(s, s) for s in (doc.get("skills") or doc.get("skill_ids") or [])]
            if pub_date and skills:
                all_items.append({"upload_date": str(pub_date).split("T")[0], "skills": skills})

        print(f"🧩 Valid docs with skills: {len(all_items)} / {len(all_docs)}")
        if not all_items: return {"warning": "No valid policy records with skills found."}

        print("🚀 Running Skill Ageing analysis...")
        result = run_skill_analysis_from_list(all_items)
        result["filters_used"] = {"keywords": keywords_list, "max_publication_date": max_publication_date}
        result["summary"]["Docs Retrieved"]       = len(all_docs)
        result["summary"]["Total Docs Available"] = total_count
        result["summary"]["Docs with Skills"]     = len(all_items)

        _save_cache(file_path, result)
        return result

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        return {"error": str(e)}


# ============================================================
#  /skill-ageing-courses
# ============================================================

@app.get("/skill-ageing-courses")
def analyze_course_skills(
    keywords:          Optional[str] = Query(None, description="Keywords to filter courses"),
    min_creation_date: Optional[str] = Query(None, description="Min creation date YYYY-MM-DD"),
    max_creation_date: Optional[str] = Query(None, description="Max creation date YYYY-MM-DD"),
):
    folder = _ensure_cache()

    filename = "completed_analysis_skill_ageing_courses"
    if keywords:
        for kw in [k.strip() for k in keywords.split(",") if k.strip()]: filename += f"_{kw}"
    if min_creation_date: filename += f"_from{min_creation_date}"
    if max_creation_date: filename += f"_to{max_creation_date}"
    filename += ".json"

    file_path = folder / filename
    print(f"🗂️ Cache path: {file_path}")
    cached = _load_cache(file_path)
    if cached: return cached

    try:
        print("🔐 Authenticating...")
        res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=15)
        res.raise_for_status()
        token   = res.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
        print("✅ Authenticated.")
        print(f"📡 Keywords: {keywords or '(none)'} | Dates: {min_creation_date or '*'} → {max_creation_date or '*'}")

        page_size = 100
        def fetch_page(page_num: int) -> dict:
            url       = f"{API}/courses?page={page_num}&page_size={page_size}"
            form_data = [("keywords_logic", "or")]
            if keywords:
                for kw in [k.strip() for k in keywords.split(",") if k.strip()]: form_data.append(("keywords", kw))
            if min_creation_date: form_data.append(("min_creation_date", min_creation_date))
            if max_creation_date: form_data.append(("max_creation_date", max_creation_date))
            for attempt in range(1, 4):
                try:
                    print(f"   ↪ Attempt {attempt}/3 — page {page_num}...")
                    r = requests.post(url, headers=headers, data=form_data, timeout=180)
                    if r.status_code != 200: return {}
                    return r.json()
                except requests.exceptions.ReadTimeout:
                    if attempt < 3: time.sleep(10)
                    else: return {}
                except Exception as ex:
                    print(f"   ❌ {type(ex).__name__}: {ex}"); return {}

        print("🔍 Probing page 1...")
        probe = fetch_page(1)
        if not probe: return {"error": "❌ Probe failed."}

        total_count = probe.get("count", 0)
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
        print(f"📊 Total courses: {total_count} → {total_pages} page(s)")
        if total_count == 0: return {"message": "No courses found."}

        all_courses = list(probe.get("items", []))
        print(f"📦 Page 1/{total_pages}: {len(all_courses)} courses")

        for page in range(2, total_pages + 1):
            print(f"📄 Fetching page {page}/{total_pages}...")
            data  = fetch_page(page)
            items = data.get("items", []) if data else []
            print(f"📦 Page {page}/{total_pages}: {len(items)} courses (running total: {len(all_courses) + len(items)})")
            if not items: break
            all_courses.extend(items)
            if len(items) < page_size:
                print("✅ Last page reached."); break

        print(f"🎯 Total courses: {len(all_courses)} / {total_count}")

        unique_skill_ids = set()
        for c in all_courses:
            for sid in (c.get("skills") or c.get("skill_ids") or []):
                if isinstance(sid, str) and sid.startswith("http"):
                    unique_skill_ids.add(sid)
        print(f"📚 {len(unique_skill_ids)} unique skill URIs — resolving in batches...")

        id_to_label = {}
        if unique_skill_ids:
            try:
                uri_list      = list(unique_skill_ids)
                total_batches = math.ceil(len(uri_list) / 50)
                for batch_num, start in enumerate(range(0, len(uri_list), 50), 1):
                    batch         = uri_list[start:start + 50]
                    skill_payload = [("ids", sid) for sid in batch]
                    print(f"   Batch {batch_num}/{total_batches}: {len(batch)} URIs...")
                    sr = requests.post(f"{API}/skills", headers={"Authorization": f"Bearer {token}"}, data=skill_payload, timeout=60)
                    sr.raise_for_status()
                    for s in sr.json().get("items", []):
                        sid = s.get("id", "")
                        if sid: id_to_label[sid] = s.get("label", sid)
                    print(f"   Resolved so far: {len(id_to_label)}")
            except Exception as e:
                print(f"⚠️ Skill lookup failed: {e}")
                id_to_label = {sid: sid for sid in unique_skill_ids}

        all_items = []
        for c in all_courses:
            upload_date = (c.get("last_updated") or c.get("creation_date") or c.get("date") or c.get("created_at"))
            if upload_date: upload_date = str(upload_date).split("T")[0]
            skills = [id_to_label.get(s, s) for s in (c.get("skills") or c.get("skill_ids") or []) if s]
            if upload_date and skills:
                all_items.append({"upload_date": upload_date, "skills": skills})

        print(f"🧩 Valid courses with skills: {len(all_items)} / {len(all_courses)}")
        if not all_items: return {"warning": "No valid courses with skills found."}

        print("🚀 Running Skill Ageing analysis...")
        result = run_skill_analysis_from_list(all_items)
        result["filters_used"] = {"keywords": keywords, "min_creation_date": min_creation_date, "max_creation_date": max_creation_date}
        result["summary"]["Courses Retrieved"]       = len(all_courses)
        result["summary"]["Total Courses Available"] = total_count
        result["summary"]["Courses with Skills"]     = len(all_items)

        _save_cache(file_path, result)
        return result

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        return {"error": str(e)}


# ============================================================
#  /ku-skill-ageing
# ============================================================

@app.get("/ku-skill-ageing")
def analyze_ku_skills(
    start_date:   Optional[str] = Query(None, description="Start date YYYY-MM"),
    end_date:     Optional[str] = Query(None, description="End date YYYY-MM"),
    kus:          Optional[str] = Query(None, description="Comma-separated KU IDs e.g. K1,K5"),
    organization: Optional[str] = Query(None, description="Filter by organization name"),
):
    from collections import Counter

    folder   = _ensure_cache()
    api_url  = f"{KU_API_URL}/analysis_results"

    filename = "completed_analysis_ku_skill_ageing"
    if organization: filename += f"_{organization.replace(' ', '_')}"
    if kus:
        for ku in [k.strip() for k in kus.split(",") if k.strip()]: filename += f"_{ku}"
    if start_date: filename += f"_from{start_date}"
    if end_date:   filename += f"_to{end_date}"
    filename += ".json"

    file_path = folder / filename
    print(f"🗂️ Cache path: {file_path}")
    cached = _load_cache(file_path)
    if cached: return cached

    try:
        params = {}
        if start_date:   params["start_date"]   = start_date
        if end_date:     params["end_date"]     = end_date
        if organization: params["organization"] = organization

        print(f"🔗 Fetching KU data from: {api_url} | Params: {params}")
        response = requests.get(api_url, params=params, headers={"Accept": "application/json"}, timeout=60)
        print(f"📥 HTTP {response.status_code}")

        ku_data = response.json()
        if isinstance(ku_data, dict) and "items" in ku_data:
            ku_data = ku_data["items"]

        if not isinstance(ku_data, list) or not ku_data:
            return {"warning": "No KU data found for the given filters."}

        print(f"✅ Retrieved {len(ku_data)} KU records")

        selected_kus = set(kus.split(",")) if kus else None
        all_items    = []

        for record in ku_data:
            upload_date  = record.get("timestamp", "").split("T")[0]
            detected_kus = record.get("detected_kus", {})
            record_org   = record.get("organization", "Unknown")
            if organization and record_org.lower() != organization.lower():
                continue
            active_kus = [ku for ku, val in detected_kus.items() if str(val) == "1"]
            if selected_kus:
                active_kus = [ku for ku in active_kus if ku in selected_kus]
            if upload_date and active_kus:
                all_items.append({"upload_date": upload_date, "organization": record_org, "skills": active_kus})

        print(f"📊 Valid KU records after filtering: {len(all_items)} / {len(ku_data)}")

        if not all_items:
            return {"warning": "No KU records matched the selected filters."}

        ku_counter = Counter()
        for item in all_items: ku_counter.update(item["skills"])
        print(f"📈 KU frequency top-10: {ku_counter.most_common(10)}")

        print("🚀 Running Skill Ageing analysis on KU data...")
        result = run_skill_analysis_from_list(all_items)
        result["filters_used"] = {"start_date": start_date, "end_date": end_date, "kus": kus, "organization": organization}
        result["summary"]["KU Records Retrieved"]      = len(all_items)
        result["summary"]["Total KU Records from API"] = len(ku_data)

        _save_cache(file_path, result)
        return result

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        return {"error": f"KU skill analysis failed: {str(e)}"}


# ============================================================
#  /ku-debug
# ============================================================

@app.get("/ku-debug")
def ku_debug(
    start_date:     Optional[str] = Query(None, description="Start date YYYY-MM"),
    end_date:       Optional[str] = Query(None, description="End date YYYY-MM"),
    organization:   Optional[str] = Query(None, description="Filter by organization"),
):
    from collections import Counter

    api_url = f"{KU_API_URL}/analysis_results"
    params  = {}
    if start_date:   params["start_date"]   = start_date
    if end_date:     params["end_date"]     = end_date
    if organization: params["organization"] = organization

    print(f"🔍 [ku-debug] {api_url} | params: {params}")

    try:
        response = requests.get(api_url, params=params, headers={"Accept": "application/json"}, timeout=60)
        print(f"📥 HTTP {response.status_code} | {len(response.text)} chars")

        if response.status_code >= 400:
            return {"error": f"API returned HTTP {response.status_code}", "body": response.text}

        raw = response.json()
        if isinstance(raw, dict) and "items" in raw:
            records = raw["items"]
        elif isinstance(raw, list):
            records = raw
        else:
            return {"error": "Unexpected response structure", "snippet": str(raw)[:500]}

        total_records = len(records)
        if total_records == 0:
            return {"status": "⚠️ API returned 0 records", "params_used": params}

        org_counter = Counter(r.get("organization", "Unknown") for r in records)
        timestamps  = sorted([r.get("timestamp", "") for r in records if r.get("timestamp")])
        ku_counter  = Counter()
        records_with_active = 0

        for r in records:
            active = [ku for ku, val in r.get("detected_kus", {}).items() if str(val) == "1"]
            if active:
                records_with_active += 1
                ku_counter.update(active)

        return {
            "status": "✅ KU API reachable",
            "params_used": params,
            "counts": {
                "total_records": total_records,
                "records_with_active_kus": records_with_active,
                "unique_organizations": len(org_counter),
                "unique_active_kus": len(ku_counter),
            },
            "date_range": {"earliest": timestamps[0] if timestamps else "N/A", "latest": timestamps[-1] if timestamps else "N/A"},
            "organizations": dict(org_counter.most_common()),
            "ku_frequency_top20": dict(ku_counter.most_common(20)),
            "sample_records": [
                {
                    "organization": r.get("organization"),
                    "timestamp":    r.get("timestamp"),
                    "active_kus":   [ku for ku, v in r.get("detected_kus", {}).items() if str(v) == "1"]
                }
                for r in records[:3]
            ]
        }

    except Exception as e:
        print(f"❌ ERROR in ku-debug: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("skill_ageing_fixed:app", host="0.0.0.0", port=8000, reload=True)
