from fastapi import FastAPI, APIRouter, Query
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests, os, json, uuid
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------
# Initialize FastAPI + Router
# -----------------------------
app = FastAPI(title="SKILLAB Skill Intelligence API")
analysis_router = APIRouter(prefix="/api/analysis", tags=["SKILL Analysis"])
# forecasting_router = APIRouter()
@analysis_router.get("/jobs_skill_analysis-DITI")
def jobs_skill_analysis(
    keywords: str = Query(..., description="Comma-separated keywords (e.g. AI, data)"),
    source: str = Query(None, description="Optional job source"),
    min_upload_date: str = Query(None),
    max_upload_date: str = Query(None),
    max_pages: int = Query(10)
):
    """
    CLEAN VERSION:
    - Fetch jobs
    - Extract ESCO URI-based skills only
    - Map ESCO URIs → labels (simple + working)
    - Analyze (biology, competing, trends, epidemiology)
    """

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
            "biology": biology_summary,
            "competing": competing_results,
            "inverse_trends": inverse_results,
            "rapid_obsolescence": rapid_drops,
            "external_shock": shock_results,
            "epidemiology": epi_metrics,
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
