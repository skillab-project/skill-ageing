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



# === API CALL TO SKILLAB ===
def get_token():
    """Authenticate and get API token"""
    res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD})
    res.raise_for_status()
    return res.text.replace('"', "")

def api_extract(payload):
    """Fetch job data from the SKILLAB Tracker API"""
    token = get_token()
    res = requests.post(f"{API}/jobs", headers={"Authorization": f"Bearer {token}"}, data=payload)
    res.raise_for_status()
    return res.json()

def get_total_jobs_in_tracker():
    """Get total job count from the Tracker"""
    token = get_token()
    res = requests.post(f"{API}/jobs", headers={"Authorization": f"Bearer {token}"}, data={})
    res.raise_for_status()
    data = res.json()
    return data.get("count", 0)

    # (rest of your analysis logic)
def run_skill_analysis_from_list(job_list):


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

        # === New: Compute slope (trend) ===
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
            "Trend": trend,                 # 👈 NEW FIELD
            "Slope": round(slope, 4)        # 👈 Optional numeric slope
        })


    # === Time Series Construction ===
    tag_series = {}
    combined_index = pd.date_range(start="2020-01-01", end="2025-12-31", freq="M")
    for skill, dates in skill_occurrences.items():
        s = pd.Series(1, index=pd.to_datetime(dates))
        s = s.resample("M").sum().reindex(combined_index, fill_value=0)
        tag_series[skill] = s

    all_tags_df = pd.DataFrame(tag_series).fillna(0)
    filtered_tags = [tag for tag in all_tags_df.columns if all_tags_df[tag].sum() >= 10]

    # === Competing Skills (Negative Correlation) ===
    competing_results = []
    for tag1, tag2 in combinations(filtered_tags, 2):
        s1, s2 = all_tags_df[tag1], all_tags_df[tag2]
        overlap = (s1 > 0) & (s2 > 0)
        if overlap.sum() < 5:
            continue
        corr = s1[overlap].corr(s2[overlap])
        if pd.notna(corr) and corr < -0.5:
            competing_results.append({
                "Skill A": tag1,
                "Skill B": tag2,
                "Correlation": round(corr, 3)
            })

    # === Inverse Trends ===
    def get_slope(ts):
        X = np.arange(len(ts)).reshape(-1, 1)
        y = ts.values.reshape(-1, 1)
        return LinearRegression().fit(X, y).coef_[0][0]

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
                "Declining Skill": tag1,
                "Competing Skill": tag2,
                "Slope A": round(slope1, 4),
                "Slope B": round(slope2, 4),
                "Overlapping Months": int(mask.sum())
            })

    # === Rapid Obsolescence Detection ===
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
        if old_incidence > 0:
            pct_change = 100 * (incidence - old_incidence) / old_incidence
        elif incidence > 0:
            pct_change = 999
        else:
            pct_change = 0

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

    # === Save results ===
    output = {
        "skill_biology_summary": biology_summary,
        "competing_skills": competing_results,
        "inverse_trends": inverse_results,
        "rapid_obsolescence": rapid_drops,
        "epidemiological_metrics": epi_metrics,
        "total_jobs_analyzed": len(job_list)
    }

    Path("results").mkdir(parents=True, exist_ok=True)
    filename = f"results/ku_skill_analysis_{uuid.uuid4().hex[:6]}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return {
        "message": "✅ KU Skill analysis complete",
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

@app.get("/jobs-with-keywords")
def analyze_jobs_with_keywords(keywords: str = Query(..., description="Comma-separated keywords (e.g. AI, data)"),
                                source: str = Query(None, description="Optional job source"),
                                min_upload_date: str = Query(None),
                                max_upload_date: str = Query(None),
                                max_pages: int = Query(10)):
    try:
        # ============================================
        # 1. AUTHENTICATION
        # ============================================
        load_dotenv()
        API = os.getenv("TRACKER_API", "https://skillab-tracker.csd.auth.gr/api")
        USER = os.getenv("TRACKER_USERNAME", "")
        PASS = os.getenv("TRACKER_PASSWORD", "")

        auth = requests.post(
            f"{API}/login",
            json={"username": USER, "password": PASS},
            timeout=20
        )
        token = auth.text.replace('"', "")
        headers = {"Authorization": f"Bearer {token}"}

        # ============================================
        # 2. FETCH JOBS
        # ============================================
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
        all_items = []

        for page in range(1, max_pages + 1):
            form = [
                ("keywords_logic", "or"),
                ("skill_ids_logic", "or"),
                ("occupation_ids_logic", "or"),
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
                timeout=40
            )

            items = r.json().get("items", [])
            if not items:
                break

            all_items.extend(items)
            if len(items) < 100:
                break

        total_jobs = len(all_items)
        if total_jobs == 0:
            return {"error": "No job postings found."}

        # ============================================
        # 3. EXTRACT ONLY REAL ESCO URI SKILLS
        # ============================================
        raw_uri_occurrences = defaultdict(list)

        for job in all_items:
            try:
                dt = job.get("upload_date")
                if not dt:
                    continue
                date = datetime.strptime(dt, "%Y-%m-%d")

                for skill in job.get("skills", []):

                    # Convert dict skill → URI
                    if isinstance(skill, dict) and "id" in skill:
                        skill = skill["id"]

                    # Keep ONLY ESCO skill URIs
                    if isinstance(skill, str) and skill.startswith("http://data.europa.eu/esco/skill/"):
                        raw_uri_occurrences[skill].append(date)

            except:
                continue

        if not raw_uri_occurrences:
            return {"error": "No ESCO URI skills found in jobs."}

        # ============================================
        # 4. LOAD ESCO SKILLS & MAP URI → label
        # ============================================
        esco_items = []
        page = 1
        while True:
            rr = requests.post(
                f"{API}/skills?page={page}&page_size=100",
                headers=headers,
                timeout=20
            )
            items = rr.json().get("items", [])
            if not items:
                break
            esco_items.extend(items)
            if len(items) < 100:
                break
            page += 1

        id_to_label = {x["id"]: x["label"].lower() for x in esco_items}

        # ===== Convert URI occurrences to DataFrame (like original working version)
        records = []
        for uri, dates in raw_uri_occurrences.items():
            for d in dates:
                records.append({"date": d.strftime("%Y-%m"), "skill_uri": uri})

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        df["skill"] = df["skill_uri"].map(id_to_label)
        df = df[df["skill"].notnull()]

        if df.empty:
            return {"error": "Could not map ESCO URIs to labels."}

        # ===== Rebuild cleaned skill occurrences
        skill_occurrences = defaultdict(list)
        for _, row in df.iterrows():
            skill_occurrences[row["skill"]].append(row["date"])

        # ============================================
        # 5. SKILL BIOLOGY
        # ============================================
        biology_summary = []
        for skill, dates in skill_occurrences.items():
            dd = pd.DataFrame(dates, columns=["date"])
            dd["ym"] = dd["date"].dt.to_period("M")

            birth = dd["date"].min()
            peak = dd["ym"].value_counts().idxmax()
            total_skill_jobs = len(dd)

            recent_use = dd["date"].max().year >= 2023
            immunity = "High" if total_skill_jobs > 20 and recent_use else "Low"

            biology_summary.append({
                "Skill": skill,
                "Date of Birth": birth.strftime("%Y-%m-%d"),
                "Peak Activity Date": str(peak),
                "Total Jobs": total_skill_jobs,
                "Immunity Score": immunity
            })

        # ============================================
        # 6. MONTHLY TIME SERIES
        # ============================================
        combined_index = pd.date_range(start="2020-01-01", end="2025-12-31", freq="ME")
        tag_series = {}

        for skill, dates in skill_occurrences.items():
            s = pd.Series(1, index=pd.to_datetime(dates))
            s = s.resample("ME").sum().fillna(0)
            tag_series[skill] = s.reindex(combined_index, fill_value=0)

        df_ts = pd.DataFrame(tag_series).fillna(0)

        # keep skills with minimum activity
        filtered = [s for s in df_ts.columns if df_ts[s].sum() >= 10]

        # ============================================
        # 7. COMPETING SKILLS (negative correlation)
        # ============================================
        # === COMPETING SKILLS (lighter thresholds) ===
        competing_results = []

        min_overlap_months = 3  # was 5
        negative_corr_threshold = -0.3  # was -0.5

        for a, b in combinations(filtered, 2):
            s1, s2 = df_ts[a], df_ts[b]

            # overlap must still exist but loosen threshold
            mask = (s1 > 0) & (s2 > 0)
            if mask.sum() < min_overlap_months:
                continue

            corr = s1[mask].corr(s2[mask])

            if pd.notna(corr) and corr < negative_corr_threshold:
                competing_results.append({
                    "Skill A": a,
                    "Skill B": b,
                    "Correlation": round(corr, 3)
                })

        # competing_results = []
        # for a, b in combinations(filtered, 2):
        #     s1, s2 = df_ts[a], df_ts[b]
        #     mask = (s1 > 0) & (s2 > 0)
        #     if mask.sum() < 5:
        #         continue
        #
        #     corr = s1[mask].corr(s2[mask])
        #     if pd.notna(corr) and corr < -0.5:
        #         competing_results.append({
        #             "Skill A": a,
        #             "Skill B": b,
        #             "Correlation": round(corr, 3)
        #         })

        # ============================================
        # 8. INVERSE TRENDS
        # ============================================
        def slope(ts):
            X = np.arange(len(ts)).reshape(-1, 1)
            y = ts.values.reshape(-1, 1)
            m = LinearRegression().fit(X, y)
            return m.coef_[0][0]

        inverse_results = []
        top_skills = sorted(filtered, key=lambda x: df_ts[x].sum(), reverse=True)[:100]

        for a, b in combinations(top_skills, 2):
            s1, s2 = df_ts[a], df_ts[b]
            mask = (s1 > 0) & (s2 > 0)
            if mask.sum() < 6:
                continue

            s1_slope = slope(s1[mask])
            s2_slope = slope(s2[mask])

            if s1_slope < -0.005 and s2_slope > 0.005:
                inverse_results.append({
                    "Declining Skill": a,
                    "Rising Skill": b,
                    "Slope A": round(s1_slope, 4),
                    "Slope B": round(s2_slope, 4),
                    "Overlapping Months": int(mask.sum())
                })

        # ============================================
        # 9. RAPID OBSOLESCENCE
        # ============================================
        rapid_drops = []
        min_peak_value = 5
        drop_window = 6
        drop_threshold = 0.3

        for skill, s in tag_series.items():
            s = s.fillna(0)

            if (s > 0).sum() < 12:
                continue

            peak = s.max()
            if peak < min_peak_value:
                continue

            peak_idx = s.idxmax()
            i = s.index.get_loc(peak_idx)
            end_i = i + drop_window
            if end_i >= len(s):
                continue

            after = s.iloc[i:end_i + 1]
            min_val = after.min()

            if (peak - min_val) / peak >= drop_threshold:
                rapid_drops.append({
                    "Skill": skill,
                    "Peak Month": peak_idx.strftime("%Y-%m"),
                    "Peak Value": int(peak),
                    "Min After Peak": int(min_val),
                    "Drop %": round((peak - min_val) * 100 / peak, 2)
                })

        # ============================================
        # 10. EXTERNAL SHOCK (±6 months around cutoff)
        # ============================================
        cutoff = pd.Timestamp("2024-12-31")

        pre_start = cutoff - pd.DateOffset(months=6) + pd.DateOffset(days=1)
        pre_end = cutoff

        post_start = cutoff + pd.DateOffset(days=1)
        post_end = cutoff + pd.DateOffset(months=6)

        shock_results = []

        for skill, s in tag_series.items():
            s = s.fillna(0)

            pre = s.loc[pre_start:pre_end]
            post = s.loc[post_start:post_end]

            if len(pre) == 0 or len(post) == 0:
                continue

            pre_avg = pre.mean()
            post_avg = post.mean()

            if pre_avg == 0 and post_avg == 0:
                continue

            change = 999 if pre_avg == 0 else 100 * (post_avg - pre_avg) / pre_avg

            if abs(change) >= 20:
                shock_results.append({
                    "Skill": skill,
                    "Pre Avg": round(pre_avg, 2),
                    "Post Avg": round(post_avg, 2),
                    "Change %": round(change, 2)
                })

        # ============================================
        # 11. EPIDEMIOLOGICAL METRICS
        # ============================================
        epi_metrics = []

        for skill, s in tag_series.items():
            s = s.fillna(0)
            total = s.sum()
            if total == 0:
                continue

            inc_1 = s.loc["2024-01-01":"2024-12-31"].sum()
            inc_2 = s.loc["2023-01-01":"2023-12-31"].sum()

            pct_change = (
                999 if inc_2 == 0 and inc_1 > 0
                else 0 if inc_2 == 0
                else 100 * (inc_1 - inc_2) / inc_2
            )

            active_months = (s > 0).sum()
            attack_rate = active_months / len(s)

            epi_metrics.append({
                "Skill": skill,
                "Total Jobs": int(total),
                "Incidence 2024": int(inc_1),
                "Incidence 2023": int(inc_2),
                "% Change": round(pct_change, 2),
                "Attack Rate": round(attack_rate, 4),
            })

        # ============================================
        # 12. SKILL R0 AND HERD IMMUNITY THRESHOLD
        # ============================================

        # ============================================
        # 12. CORRECT R0 AND HIT CALCULATION
        # ============================================

        r0_results = []
        herd_immunity_results = []

        # Presence matrix: 1 if skill is visible in that month
        presence_matrix = (df_ts > 0).astype(int)
        skills_list = list(presence_matrix.columns)
        n_months = len(presence_matrix)

        for skill in skills_list:

            # find all months where this skill is active
            active_months = np.where(presence_matrix[skill] == 1)[0]
            if len(active_months) == 0:
                continue

            total_new_after_skill = 0
            total_following_windows = 0

            for idx in active_months:

                # skills present in the CURRENT month (co-skills)
                baseline_set = set(presence_matrix.columns[presence_matrix.iloc[idx] == 1])

                # look ahead 3 months AFTER the skill appears
                for j in range(idx + 1, min(idx + 4, n_months)):
                    following_set = set(presence_matrix.columns[presence_matrix.iloc[j] == 1])

                    # NEW skills that appear AFTER this skill
                    new_skills = following_set - baseline_set

                    total_new_after_skill += len(new_skills)
                    total_following_windows += 1

            if total_following_windows == 0:
                continue

            r0 = total_new_after_skill / total_following_windows

            # Herd Immunity Threshold
            hit = 0 if r0 < 1 else (1 - (1 / r0))

            r0_results.append({
                "Skill": skill,
                "R0": round(r0, 3)
            })

            herd_immunity_results.append({
                "Skill": skill,
                "R0": round(r0, 3),
                "HIT": round(hit, 4)
            })

        # ============================================
        # SKILL CONTACT RATE (Exposure Rate)
        # ============================================

        contact_rate_results = []

        # We need job-level skill lists, not time series.
        # So reconstruct: for each job, the set of skills (labels)
        job_skill_sets = []

        for job in all_items:
            skills = []
            for s in job.get("skills", []):

                # Normalize skill format
                if isinstance(s, dict) and "id" in s:
                    s = s["id"]

                # Keep only ESCO URIs
                if isinstance(s, str) and s in id_to_label:
                    skills.append(id_to_label[s].lower())

            if skills:
                job_skill_sets.append(set(skills))

        # Contact Rate: How many other skills appear with each skill
        skill_co_counts = defaultdict(int)
        skill_occ_counts = defaultdict(int)

        for skill_set in job_skill_sets:
            for skill in skill_set:
                skill_occ_counts[skill] += 1
                skill_co_counts[skill] += (len(skill_set) - 1)  # co-skills = all others

        contact_rate_results = []
        for skill in skill_occ_counts:
            rate = skill_co_counts[skill] / skill_occ_counts[skill]
            contact_rate_results.append({
                "Skill": skill,
                "Contact Rate": round(rate, 3),
                "Occurrences": skill_occ_counts[skill]
            })

        # ============================================
        # 12. SAVE RESULTS
        # ============================================
        Path("results").mkdir(exist_ok=True)

        fname = f"results/skill_analysis_{uuid.uuid4().hex[:6]}.json"

        output = {
            "skill_biology_summary": biology_summary,
            "competing_skills": competing_results,
            "inverse_trends": inverse_results,
            "rapid_obsolescence": rapid_drops,
            "external_shock_skills": shock_results,
            "epidemiological_metrics": epi_metrics,
            "total_jobs": total_jobs,
            "r0": r0_results,
            "herd_immunity_threshold": herd_immunity_results,
            "contact_rate": contact_rate_results,

        }

        with open(fname, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return {
            "message": "Skill analysis complete",
            "file": fname,
            "summary": {
                "skills": len(skill_occurrences),
                "competing": len(competing_results),
                "inverse": len(inverse_results),
                "obsolescence": len(rapid_drops),
                "shock": len(shock_results),
                "epidemiology": len(epi_metrics),
                "jobs": total_jobs,
                "HIT": len(herd_immunity_results),
                "contact_rate_skills": len(contact_rate_results),
            },
            "data": output
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/jobs")
def analyze_skills(occupation: str = Query(...), source: str = Query(...)):
    try:
        # === Fetch data ===
        source_data = api_extract({
            "occupation_ids": [occupation],
            "sources": source,
            "limit": 100000
        })

        total_tracker_jobs = get_total_jobs_in_tracker()
        print(f"📦 Total jobs in tracker (unfiltered): {total_tracker_jobs}")

        all_items = source_data.get("items", [])
        if not all_items:
            return {"error": "No jobs found for given occupation and source."}

        jobs_with_skills = [job for job in all_items if job.get("skills")]
        jobs_without_skills = [job for job in all_items if not job.get("skills")]

        print(f"✅ Jobs with skills: {len(jobs_with_skills)}")
        print(f"❌ Jobs without skills: {len(jobs_without_skills)}")

        # === Step 1: Collect unique skill IDs ===
        unique_skill_ids = set()
        for job in jobs_with_skills:
            for sid in job.get("skills", []):
                if isinstance(sid, str) and sid.startswith("http"):
                    unique_skill_ids.add(sid)

        id_to_label = {}

        if unique_skill_ids:
            print(f"📚 Fetching {len(unique_skill_ids)} unique skill labels from /api/skills ...")

            # === Authenticate with Tracker ===
            API = os.getenv("TRACKER_API_URL", "https://skillab-tracker.csd.auth.gr/api")
            USERNAME = os.getenv("TRACKER_USERNAME")
            PASSWORD = os.getenv("TRACKER_PASSWORD")

            res = requests.post(f"{API}/login", json={"username": USERNAME, "password": PASSWORD})
            res.raise_for_status()
            token = res.text.replace('"', "")

            # === Bulk POST to /api/skills ===
            skill_payload = {"ids": list(unique_skill_ids)}
            try:
                skill_res = requests.post(
                    f"{API}/skills",
                    headers={"Authorization": f"Bearer {token}"},
                    data=skill_payload,
                    timeout=60
                )
                skill_res.raise_for_status()
                skill_data = skill_res.json().get("items", [])
                id_to_label = {s["id"]: s.get("label", s["id"]) for s in skill_data}
                print(f"✅ Retrieved {len(id_to_label)} skill labels")

            except Exception as e:
                print(f"⚠️ Skill label lookup failed: {e}")
                id_to_label = {sid: sid for sid in unique_skill_ids}
        else:
            print("ℹ️ No ESCO skill IDs detected — skipping label mapping.")

        # === Step 2: Replace skill IDs with labels ===
        for job in jobs_with_skills:
            skills = job.get("skills", [])
            job["skills"] = [id_to_label.get(s, s) for s in skills]

        # === Proceed with analysis ===
        skill_occurrences = defaultdict(list)
        for job in all_items:
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

        warning_message = None
        total_jobs = len(all_items)

        if total_jobs < 50:
            warning_message = f"⚠️ Low job count: only {total_jobs} jobs found. Results may not be representative."

        # === Skill Biology ===
        biology_summary = []
        for skill, dates in skill_occurrences.items():
            df = pd.DataFrame(dates, columns=["date"])
            df["year_month"] = df["date"].dt.to_period("M")
            birth = df["date"].min()
            peak = df["year_month"].value_counts().idxmax()
            total_jobs = len(df)
            recent_use = df["date"].max().year > 2022
            immunity = "High" if total_jobs > 20 and recent_use else "Low"

            biology_summary.append({
                "Skill": skill,
                "Date of Birth": birth.strftime("%Y-%m-%d"),
                "Peak Activity Date": str(peak),
                "Total Jobs": total_jobs,
                "Immunity Score": immunity
            })

        # === Time Series Setup ===
        tag_series = {}
        combined_index = pd.date_range(start="2020-01-01", end="2025-12-31", freq="M")
        for skill, dates in skill_occurrences.items():
            s = pd.Series(1, index=pd.to_datetime(dates))
            s = s.resample("M").sum().fillna(0)
            tag_series[skill] = s.reindex(combined_index, fill_value=0)

        all_tags_df = pd.DataFrame(tag_series).fillna(0)
        min_total_jobs = 10
        filtered_tags = [tag for tag in all_tags_df.columns if all_tags_df[tag].sum() >= min_total_jobs]

        # === Competing Skills ===
        competing_results = []
        min_overlap_months = 5
        for tag1, tag2 in combinations(filtered_tags, 2):
            s1 = all_tags_df[tag1]
            s2 = all_tags_df[tag2]
            overlap = (s1 > 0) & (s2 > 0)
            if overlap.sum() < min_overlap_months:
                continue
            corr = s1[overlap].corr(s2[overlap])
            if pd.notna(corr) and corr < -0.5:
                competing_results.append({
                    "Skill A": tag1,
                    "Skill B": tag2,
                    "Correlation": round(corr, 3)
                })

        # === Inverse Trends ===
        def get_slope(ts):
            X = np.arange(len(ts)).reshape(-1, 1)
            y = ts.values.reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            return model.coef_[0][0]

        inverse_results = []
        top_100_skills = sorted(filtered_tags, key=lambda x: all_tags_df[x].sum(), reverse=True)[:100]
        for tag1, tag2 in combinations(top_100_skills, 2):
            s1 = all_tags_df[tag1]
            s2 = all_tags_df[tag2]
            mask = (s1 > 0) & (s2 > 0)
            if mask.sum() < 6:
                continue
            slope1 = get_slope(s1[mask])
            slope2 = get_slope(s2[mask])
            if slope1 < -0.005 and slope2 > 0.005:
                inverse_results.append({
                    "Declining Skill": tag1,
                    "Competing Skill": tag2,
                    "Slope A": round(slope1, 4),
                    "Slope B": round(slope2, 4),
                    "Overlapping Months": int(mask.sum())
                })

                # === Rapid Obsolescence Detection ===
            drop_threshold = 0.3
            drop_window = 6
            min_peak_value = 5
            min_total_months = 12

            rapid_drops = []
            for tag, series in tag_series.items():
                series = series.fillna(0)
                if (series > 0).sum() < min_total_months:
                    continue
                peak_value = series.max()
                if peak_value < min_peak_value:
                    continue
                peak_idx = series.idxmax()
                peak_loc = series.index.get_loc(peak_idx)
                end_loc = peak_loc + drop_window
                if end_loc >= len(series):
                    continue
                post_peak = series.iloc[peak_loc:end_loc + 1]
                min_val = post_peak.min()
                drop_ratio = (peak_value - min_val) / peak_value
                if drop_ratio >= drop_threshold:
                    rapid_drops.append({
                        "Skill": tag,
                        "Peak Month": peak_idx.strftime("%Y-%m"),
                        "Peak Value": int(peak_value),
                        "Min Value After Peak": int(min_val),
                        "Drop Window (Months)": drop_window,
                        "Drop %": round(drop_ratio * 100, 2)
                    })

            # === External Shock Detection ===
            # === External Shock Detection ===
            shock_start = pd.Timestamp("2023-01-01")
            shock_end = pd.Timestamp("2023-12-31")
            pre_start = pd.Timestamp("2022-01-01")
            pre_end = pd.Timestamp("2022-12-31")

            shock_results = []
            for tag, series in tag_series.items():
                series = series.fillna(0)

                pre_period = series.loc[pre_start:pre_end]
                post_period = series.loc[shock_start:shock_end]

                if len(pre_period) == 0 or len(post_period) == 0:
                    continue

                pre_avg = pre_period.mean()
                post_avg = post_period.mean()

                if pre_avg == 0 and post_avg == 0:
                    continue

                change_pct = 0
                if pre_avg == 0:
                    change_pct = 999  # treat as infinite growth
                else:
                    change_pct = 100 * (post_avg - pre_avg) / pre_avg

                if abs(change_pct) < 20:  # 🧪 new relaxed threshold
                    continue

                shock_results.append({
                    "Skill": tag,
                    "Pre-Shock Avg": round(pre_avg, 2),
                    "Post-Shock Avg": round(post_avg, 2),
                    "Change (%)": round(change_pct, 2)
                })

            # Optional print for debug (top 5 skills)
            print("🔍 Checking pre vs post shock for top 5 skills:")
            for tag in list(tag_series.keys())[:5]:
                s = tag_series[tag].fillna(0)
                pre_avg = s.loc[pre_start:pre_end].mean()
                post_avg = s.loc[shock_start:shock_end].mean()
                print(f"{tag}: pre={pre_avg:.2f}, post={post_avg:.2f}")

            # === Epidemiological Metrics ===
            # With this:
            # Time windows
            shock_start = pd.Timestamp("2023-01-01")
            shock_end = pd.Timestamp("2023-12-31")  # full year
            old_start = pd.Timestamp("2022-01-01")
            old_end = pd.Timestamp("2022-12-31")

            epi_metrics = []

            for tag, series in tag_series.items():
                series = series.fillna(0)
                total_jobs = series.sum()
                if total_jobs == 0:
                    continue

                # Use range-based slicing
                incidence = series.loc[shock_start:shock_end].sum()
                old_incidence = series.loc[old_start:old_end].sum()

                if old_incidence > 0:
                    pct_change = 100 * (incidence - old_incidence) / old_incidence
                elif incidence > 0:
                    pct_change = 999
                else:
                    pct_change = 0

                incidence_prevalence_ratio = incidence / total_jobs if total_jobs > 0 else 0

                recent_activity = series[series.index >= datetime(2023, 7, 1)].sum()
                is_dead = recent_activity == 0

                revival = "Yes" if old_incidence < incidence and old_incidence > 0 else "No"

                mortality_ratio = incidence / (total_jobs - incidence) if (total_jobs - incidence) > 0 else 999

                was_active = old_incidence > 0 or incidence > 0
                cfr = 1.0 if (was_active and is_dead) else 0.0

                active_months = (series > 0).sum()
                total_months = len(series)
                attack_rate = active_months / total_months if total_months > 0 else 0.0

                epi_metrics.append({
                    "Skill": tag,
                    "Total Jobs": int(total_jobs),
                    "Incidence (2023)": int(incidence),
                    "Incidence (2022)": int(old_incidence),
                    "% Change in Incidence": round(pct_change, 2),
                    "Incidence : Prevalence": round(incidence_prevalence_ratio, 4),
                    "Mortality Risk": "☠️" if is_dead else "🟢",
                    "Revived?": revival,
                    "Incidence : Mortality Ratio": round(mortality_ratio, 2),
                    "CFR": round(cfr, 2),
                    "Attack Rate": round(attack_rate, 4)
                })

            # === Save all results ===
            output = {
                "skill_biology_summary": biology_summary,
                "competing_skills": competing_results,
                "inverse_trends": inverse_results,
                "rapid_obsolescence": rapid_drops,
                "external_shock_skills": shock_results,
                # ... inside final return dictionary
                "total_jobs_in_tracker": total_tracker_jobs,
                "epidemiological_metrics": epi_metrics,
                "warning": warning_message,

            }

            Path("results").mkdir(parents=True, exist_ok=True)

            def sanitize(s):
                return s.replace("/", "_").replace(":", "_")

            filename = f"results/skill_analysis_{sanitize(source)}_{sanitize(occupation)}_{uuid.uuid4().hex[:6]}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            return {
                "message": "✅ Skill analysis complete",
                "file_saved": filename,
                "summary": {
                    "Total Skills Found": len(biology_summary),
                    "Competing Skills Pairs": len(competing_results),
                    "Inverse Trend Pairs": len(inverse_results),
                    "Rapid Obsolescence Skills": len(rapid_drops),
                    "External Shock Skills (2023)": len(shock_results),
                    "total_epidemiological_skills": len(epi_metrics),
                },
                "data": output
            }

    except Exception as e:
        return {"error": str(e)}

@app.get("/law-policy")
def analyze_law_policy_skills(
    keywords: str = Query(None, description="Comma-separated keywords to filter law/policies, e.g. AI,Data,Education"),
    max_publication_date: str = Query(None, description="Max publication date in YYYY-MM-DD format")
):
    """
    Runs Skill Ageing analysis on law & policy documents from the Skillab Tracker.
    Fetches data from /api/law-policies using optional keyword and publication date filters,
    matches ESCO skill IDs with their labels via a single POST to /api/skills,
    and performs the full Skill Ageing analysis.
    """
    try:
        # === Step 1: Read environment variables ===
        API = os.getenv("TRACKER_API_URL", "https://skillab-tracker.csd.auth.gr/api")
        USERNAME = os.getenv("TRACKER_USERNAME")
        PASSWORD = os.getenv("TRACKER_PASSWORD")

        if not USERNAME or not PASSWORD:
            raise ValueError("❌ Missing TRACKER_USERNAME or TRACKER_PASSWORD in .env")

        print("🔧 Loaded from .env:")
        print("TRACKER_API =", API)
        print("TRACKER_USERNAME =", USERNAME)
        print("TRACKER_PASSWORD =", "********")  # hidden for safety

        # === Step 2: Authenticate ===
        def get_token():
            try:
                res = requests.post(
                    f"{API}/login",
                    json={"username": USERNAME, "password": PASSWORD},
                    timeout=10
                )
                res.raise_for_status()
                return res.text.replace('"', "")
            except Exception as e:
                raise RuntimeError(f"Login failed: {e}")

        token = get_token()

        # === Step 3: Build payload for /api/law-policies ===
        payload = {}
        if keywords:
            payload["keywords"] = [k.strip() for k in keywords.split(",") if k.strip()]
        if max_publication_date:
            payload["max_publication_date"] = max_publication_date

        print(f"📡 Fetching law/policy data with payload: {payload}")

        # === Step 4: Fetch law/policy data ===
        res = requests.post(
            f"{API}/law-policies",
            headers={"Authorization": f"Bearer {token}"},
            data=payload,
            timeout=60
        )
        res.raise_for_status()
        data = res.json()

        items = data.get("items", data)
        if not items:
            return {"error": "No law or policy documents found for the given filters."}

        print(f"✅ Retrieved {len(items)} law/policy documents")

        # === Step 5: Collect all skill IDs ===
        unique_skill_ids = set()
        for p in items:
            skill_list = p.get("skills") or p.get("skill_ids") or []
            for sid in skill_list:
                if isinstance(sid, str) and sid.startswith("http"):
                    unique_skill_ids.add(sid)

        # === Step 6: Fetch skill labels ===
        if unique_skill_ids:
            print(f"📚 Fetching {len(unique_skill_ids)} unique skill labels from /api/skills ...")
            skill_payload = {"ids": list(unique_skill_ids)}

            try:
                skill_res = requests.post(
                    f"{API}/skills",
                    headers={"Authorization": f"Bearer {token}"},
                    data=skill_payload,
                    timeout=60
                )
                skill_res.raise_for_status()
                skill_data = skill_res.json().get("items", [])
                id_to_label = {s["id"]: s.get("label", s["id"]) for s in skill_data}
                print(f"✅ Retrieved {len(id_to_label)} skill labels")
            except Exception as e:
                print(f"⚠️ Skill label lookup failed: {e}")
                id_to_label = {sid: sid for sid in unique_skill_ids}
        else:
            id_to_label = {}

        # === Step 7: Transform to analysis-ready format ===
        all_items = []
        for p in items:
            pub_date = p.get("publication_date") or p.get("date")
            skills = p.get("skills") or p.get("skill_ids") or []
            skills = [id_to_label.get(s, s) for s in skills]  # Convert IDs → labels
            if pub_date and isinstance(skills, list) and len(skills) > 0:
                all_items.append({
                    "upload_date": pub_date,
                    "skills": skills
                })

        if not all_items:
            return {"warning": "No valid policy records with skills found."}

        print(f"🧩 Total valid policies with skills: {len(all_items)}")
        print("🚀 Running Skill Ageing analysis...")

        # === Step 8: Run analysis ===
        result = run_skill_analysis_from_list(all_items)

        # === Step 9: Save output ===
        Path("results").mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filter_tag = f"{(keywords or 'all').replace(',', '-')}_{(max_publication_date or 'latest')}"
        filename = f"results/law_policy_skill_analysis_{filter_tag}_{uuid.uuid4().hex[:6]}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        result["file_saved"] = filename
        result["filters_used"] = {
            "keywords": keywords,
            "max_publication_date": max_publication_date
        }

        return result

    except Exception as e:
        return {"error": f"Law/Policy skill ageing analysis failed: {str(e)}"}


@app.get("/ku")
def analyze_ku_skills(
    start_date: str = Query(None, description="Start date in YYYY-MM format"),
    end_date: str = Query(None, description="End date in YYYY-MM format"),
    kus: str = Query(None, description="Comma-separated list of KU IDs to include, e.g., K1,K5,K10"),
    organization: str = Query(None, description="Optional organization name to filter KU results by")
):
    import requests, json
    from datetime import datetime
    from collections import Counter

    BASE_URL = "https://portal.skillab-project.eu/ku-detection"
    ENDPOINT = "/analysis_results"
    api_url = f"{BASE_URL}{ENDPOINT}"

    # === 1. Build query parameters ===
    params = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if organization:
        params["organization"] = organization

    try:
        print(f"🔍 Fetching KU analysis data from: {api_url} with params {params}")
        response = requests.get(api_url, params=params, timeout=60)
        response.raise_for_status()
        ku_data = response.json()

        # === Debug: List unique organizations ===
        orgs = sorted({r.get("organization", "Unknown") for r in ku_data})
        print(f"🏢 Found {len(orgs)} unique organizations:")
        print(orgs)

        if not ku_data:
            return {"error": "No KU analysis data found for the given filters."}

        print(f"✅ Retrieved {len(ku_data)} KU records")

        # === 2. Transform KU data ===
        selected_kus = set(kus.split(",")) if kus else None
        all_items = []

        for record in ku_data:
            upload_date = record.get("timestamp", "").split("T")[0]
            detected_kus = record.get("detected_kus", {})
            record_org = record.get("organization", "Unknown")

            # ✅ Apply organization filter (if provided)
            if organization and record_org.lower() != organization.lower():
                continue

            # ✅ Keep only active KUs (value == "1")
            active_kus = [ku for ku, val in detected_kus.items() if str(val) == "1"]

            # ✅ Optional filter for specific KUs
            if selected_kus:
                active_kus = [ku for ku in active_kus if ku in selected_kus]

            if upload_date and active_kus:
                all_items.append({
                    "upload_date": upload_date,
                    "organization": record_org,
                    "skills": active_kus
                })

        print(f"📊 Total valid items with skills: {len(all_items)}")

        # === 3. Frequency check ===
        ku_counter = Counter()
        for job in all_items:
            ku_counter.update(job["skills"])
        print("📊 KU frequency counts (top 10):")
        for ku, count in ku_counter.most_common(10):
            print(f"{ku}: {count}")

        if not all_items:
            return {"warning": "No KU records matched the selected filters."}

        # === 4. Run skill ageing analysis ===
        result = run_skill_analysis_from_list(all_items)
        result["filters_used"] = {
            "start_date": start_date,
            "end_date": end_date,
            "kus": kus,
            "organization": organization
        }

        return result

    except Exception as e:
        return {"error": f"KU skill analysis failed: {str(e)}"}


@app.get("/courses")
def analyze_course_skills(
    keywords: str = Query(None, description="Keywords to filter courses"),
    min_creation_date: str = Query(None, description="Minimum creation date (YYYY-MM-DD)"),
    max_creation_date: str = Query(None, description="Maximum creation date (YYYY-MM-DD)")
):
    try:
        # === Step 1: Read environment variables ===
        API = os.getenv("TRACKER_API")
        USERNAME = os.getenv("TRACKER_USERNAME")
        PASSWORD = os.getenv("TRACKER_PASSWORD")

        print("🔧 Loaded from .env:")
        print("TRACKER_API =", API)
        print("TRACKER_USERNAME =", USERNAME)
        print("TRACKER_PASSWORD =", "********")  # hidden

        # === Step 2: Authenticate ===
        def get_token():
            try:
                res = requests.post(
                    f"{API}/login",
                    json={"username": USERNAME, "password": PASSWORD},
                    timeout=10
                )
                res.raise_for_status()
                return res.text.replace('"', "")
            except Exception as e:
                raise RuntimeError(f"Login failed: {e}")

        token = get_token()

        # === Step 3: Prepare payload for /api/courses ===
        payload = {}
        if keywords:
            payload["keywords"] = [keywords]
        if min_creation_date:
            payload["min_creation_date"] = min_creation_date
        if max_creation_date:
            payload["max_creation_date"] = max_creation_date

        print(f"📡 Fetching course data with payload: {payload}")

        # === Step 4: Fetch courses ===
        res = requests.post(
            f"{API}/courses",
            headers={"Authorization": f"Bearer {token}"},
            data=payload,
            timeout=60
        )
        res.raise_for_status()
        course_data = res.json()
        items = course_data.get("items", [])
        if not items:
            return {"warning": "No valid courses found in tracker response."}

        print(f"✅ Retrieved {len(items)} courses")

        # === Step 5: Extract unique skill IDs ===
        unique_skill_ids = set()
        for c in items:
            skills = c.get("skills") or c.get("skill_ids") or []
            unique_skill_ids.update(skills)

        print(f"📚 Fetching {len(unique_skill_ids)} unique skill labels from /api/skills ...")

        # === Step 6: Fetch skills in batches ===
        def chunked_iterable(iterable, size):
            it = iter(iterable)
            while True:
                chunk = list(islice(it, size))
                if not chunk:
                    break
                yield chunk

        all_skill_data = []
        batch_size = 100
        for batch in chunked_iterable(list(unique_skill_ids), batch_size):
            skill_payload = [("ids", sid) for sid in batch]
            print(f"🧾 Sending batch of {len(batch)} skill IDs to /api/skills ...")
            skill_res = requests.post(
                f"{API}/skills",
                headers={"Authorization": f"Bearer {token}"},
                data=skill_payload,
                timeout=60
            )
            skill_res.raise_for_status()
            skill_batch = skill_res.json().get("items", [])
            all_skill_data.extend(skill_batch)

        print(f"✅ Retrieved {len(all_skill_data)} total skill labels across batches")

        # === Step 7: Map skill IDs to labels ===
        id_to_label = {}
        for s in all_skill_data:
            skill_id = s["id"].strip().rstrip("/")
            label = s.get("label", s["id"])
            id_to_label[skill_id] = label
            id_to_label[skill_id.split("/")[-1]] = label  # UUID version

        print("🧠 Example skill mappings:")
        for k, v in list(id_to_label.items())[:5]:
            print(f"  {k} → {v}")

        # === Step 8: Convert to analysis-ready format ===
        all_items = []
        for c in items:
            upload_date = (
                c.get("last_updated")
                or c.get("creation_date")
                or c.get("date")
                or c.get("created_at")
            )
            if upload_date:
                upload_date = str(upload_date).split("T")[0]

            skills = c.get("skills") or c.get("skill_ids") or []
            skills = [id_to_label.get(s.strip().rstrip("/"), s) for s in skills if s]

            if upload_date and skills:
                all_items.append({
                    "upload_date": upload_date,
                    "skills": skills
                })

        print("🧩 Preview of first course with skills:")
        print(json.dumps(all_items[:2], indent=2, ensure_ascii=False))

        if not all_items:
            return {"warning": "No valid courses with skills found."}

        # === Step 9: Run analysis ===
        result = run_skill_analysis_from_list(all_items)

        # === Step 10: Save output ===
        Path("results").mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filter_tag = f"{(keywords or 'all').replace(',', '-')}"
        filename = f"results/course_skill_analysis_{filter_tag}_{timestamp}_{uuid.uuid4().hex[:6]}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        result["file_saved"] = filename
        result["filters_used"] = {
            "keywords": keywords,
            "min_creation_date": min_creation_date,
            "max_creation_date": max_creation_date
        }

        return result

    except Exception as e:
        return {"error": f"Course skill ageing analysis failed: {str(e)}"}
