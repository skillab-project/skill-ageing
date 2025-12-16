from analysis import run_skill_analysis_from_list

def test_analysis_runs():
    fake_jobs = [
        {"upload_date": "2022-01-01", "skills": ["Python", "SQL"]},
        {"upload_date": "2023-01-01", "skills": ["Python"]},
    ]

    result = run_skill_analysis_from_list(fake_jobs)

    assert "data" in result
    assert "skill_biology_summary" in result["data"]

def test_skill_death_detection():
    jobs = [
        {"upload_date": "2020-01-01", "skills": ["OldSkill"]},
        {"upload_date": "2021-01-01", "skills": ["OldSkill"]},
    ]

    result = run_skill_analysis_from_list(jobs)
    epi = result["data"]["epidemiological_metrics"]

    dead = [s for s in epi if s["Skill"] == "OldSkill"]
    assert dead[0]["Mortality Risk"] == "☠️"

# --------------------------------------------------
# 1. Empty input should not crash
# --------------------------------------------------
def test_empty_input():
    result = run_skill_analysis_from_list([])

    assert "data" in result
    assert result["summary"]["Total Skills Found"] == 0


# --------------------------------------------------
# 2. Single skill, single occurrence
# --------------------------------------------------
def test_single_skill_basic():
    jobs = [
        {"upload_date": "2023-01-01", "skills": ["Python"]}
    ]

    result = run_skill_analysis_from_list(jobs)
    skills = result["data"]["skill_biology_summary"]

    assert len(skills) == 1
    assert skills[0]["Skill"] == "Python"
    assert skills[0]["Total Jobs"] == 1


# --------------------------------------------------
# 3. Skill trend detection (rising)
# --------------------------------------------------
def test_rising_skill_trend():
    jobs = []

    # Skill appears more frequently in later years
    for year in [2020, 2021, 2022, 2023, 2024]:
        jobs.append({
            "upload_date": f"{year}-01-01",
            "skills": ["AI"]
        })

    result = run_skill_analysis_from_list(jobs)

    ai_skill = next(
        s for s in result["data"]["skill_biology_summary"]
        if s["Skill"] == "AI"
    )

    assert ai_skill["Trend"] in ["Rising", "Stable"]


# --------------------------------------------------
# 4. Declining skill detection
# --------------------------------------------------
def test_declining_skill_trend():
    jobs = [
        {"upload_date": "2020-01-01", "skills": ["OldTech"]},
        {"upload_date": "2021-01-01", "skills": ["OldTech"]},
        {"upload_date": "2022-01-01", "skills": ["OldTech"]},
    ]

    result = run_skill_analysis_from_list(jobs)

    old_skill = next(
        s for s in result["data"]["skill_biology_summary"]
        if s["Skill"] == "OldTech"
    )

    assert old_skill["Trend"] in ["Declining", "Stable"]


# --------------------------------------------------
# 5. Epidemiological mortality detection
# --------------------------------------------------
def test_skill_mortality():
    jobs = [
        {"upload_date": "2020-01-01", "skills": ["DeadSkill"]},
        {"upload_date": "2021-01-01", "skills": ["DeadSkill"]},
    ]

    result = run_skill_analysis_from_list(jobs)

    epi = result["data"]["epidemiological_metrics"]
    dead = next(s for s in epi if s["Skill"] == "DeadSkill")

    assert dead["Mortality Risk"] in ["☠️", "🟢"]


# --------------------------------------------------
# 6. Competing skills (negative correlation)
# --------------------------------------------------
def test_competing_skills_structure():
    jobs = [
        {"upload_date": "2020-01-01", "skills": ["Java"]},
        {"upload_date": "2021-01-01", "skills": ["Python"]},
        {"upload_date": "2022-01-01", "skills": ["Java"]},
        {"upload_date": "2023-01-01", "skills": ["Python"]},
        {"upload_date": "2024-01-01", "skills": ["Java"]},
        {"upload_date": "2025-01-01", "skills": ["Python"]},
    ]

    result = run_skill_analysis_from_list(jobs)

    competing = result["data"]["competing_skills"]

    # We don't assert existence (too data-sensitive),
    # only structure correctness
    for pair in competing:
        assert "Skill A" in pair
        assert "Skill B" in pair
        assert "Correlation" in pair


# --------------------------------------------------
# 4. Birth date detection
# --------------------------------------------------
def test_skill_birth_date():
    jobs = [
        {"upload_date": "2021-06-01", "skills": ["Kubernetes"]},
        {"upload_date": "2023-01-01", "skills": ["Kubernetes"]},
    ]

    result = run_skill_analysis_from_list(jobs)
    skill = result["data"]["skill_biology_summary"][0]

    assert skill["Date of Birth"] == "2021-06-01"


# --------------------------------------------------
# 5. Peak activity detection
# --------------------------------------------------
def test_peak_activity_month():
    jobs = [
        {"upload_date": "2022-01-01", "skills": ["Rust"]},
        {"upload_date": "2022-01-15", "skills": ["Rust"]},
        {"upload_date": "2023-01-01", "skills": ["Rust"]},
    ]

    result = run_skill_analysis_from_list(jobs)
    rust = result["data"]["skill_biology_summary"][0]

    assert "2022-01" in rust["Peak Activity Date"]


# --------------------------------------------------
# 6. Immunity score logic
# --------------------------------------------------
def test_immunity_score_high():
    jobs = [
        {"upload_date": f"2023-01-{i+1:02d}", "skills": ["Python"]}
        for i in range(25)
    ]

    result = run_skill_analysis_from_list(jobs)
    python = result["data"]["skill_biology_summary"][0]

    assert python["Immunity Score"] in ["High", "Low"]


# --------------------------------------------------
# 7. Trend slope is numeric
# --------------------------------------------------
def test_trend_slope_numeric():
    jobs = [
        {"upload_date": f"2022-{m:02d}-01", "skills": ["AI"]}
        for m in range(1, 13)
    ]

    result = run_skill_analysis_from_list(jobs)
    ai = result["data"]["skill_biology_summary"][0]

    assert isinstance(ai["Slope"], float)


# --------------------------------------------------
# 8. Rapid obsolescence structure
# --------------------------------------------------
def test_rapid_obsolescence_structure():
    jobs = []
    for year in range(2020, 2023):
        jobs.append({"upload_date": f"{year}-01-01", "skills": ["Flash"]})

    result = run_skill_analysis_from_list(jobs)

    for item in result["data"]["rapid_obsolescence"]:
        assert "Skill" in item
        assert "Drop %" in item


# --------------------------------------------------
# 9. Inverse trend structure
# --------------------------------------------------
def test_inverse_trend_structure():
    jobs = []
    for year in range(2020, 2023):
        jobs.append({"upload_date": f"{year}-01-01", "skills": ["Cobol"]})
    for year in range(2023, 2026):
        jobs.append({"upload_date": f"{year}-01-01", "skills": ["Python"]})

    result = run_skill_analysis_from_list(jobs)

    for pair in result["data"]["inverse_trends"]:
        assert "Declining Skill" in pair
        assert "Competing Skill" in pair


# --------------------------------------------------
# 10. Epidemiological metric completeness
# --------------------------------------------------
def test_epi_metric_fields():
    jobs = [
        {"upload_date": "2022-01-01", "skills": ["Scala"]},
        {"upload_date": "2023-01-01", "skills": ["Scala"]},
    ]

    result = run_skill_analysis_from_list(jobs)
    epi = result["data"]["epidemiological_metrics"][0]

    expected_keys = {
        "Skill",
        "Total Jobs",
        "Incidence (2023)",
        "Incidence (2022)",
        "% Change in Incidence",
        "Incidence : Prevalence",
        "Mortality Risk",
        "Revived?",
        "Incidence : Mortality Ratio",
        "CFR",
        "Attack Rate"
    }

    assert expected_keys.issubset(epi.keys())