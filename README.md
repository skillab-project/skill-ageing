# Skill Ageing Back-End

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/skillab-project/skill-ageing)

## Description

This project implements the backend API for **Skill Ageing**, an open-source framework designed to monitor the dynamics of skill demand over time — tracking which skills are growing, plateauing, or declining in the European labour market.

It is built with FastAPI (Python) and exposes forecast endpoints for:

- Fetching ESCO-tagged job postings, EU law/policy documents, and KU (Knowledge Unit) detection results from the SkillLab APIs.
- Building monthly time-series of skill or KU occurrence counts using pandas pivot tables.
- Forecasting future skill demand over a configurable horizon (3, 6, or 12 months) using a three-tier fallback model: **ARIMA → Linear Trend → Moving Average**.
- Normalising predicted counts to relative market shares across all forecasted skills per month.
- Caching completed forecasts to `Completed_Analyses/` to avoid recomputation on repeated requests.

The service is part of the [SkillLab](https://github.com/skillab-project) EU Horizon Europe project.

---

## Getting Started Guide

### Prerequisites

- **Python 3.11** or newer ([Download Python](https://www.python.org/downloads/))
- **Git** ([Download Git](https://git-scm.com/downloads))
- **Access to the SkillLab Tracker API** — credentials for `TRACKER_API`, `TRACKER_USERNAME`, and `TRACKER_PASSWORD`.
- **Optional:** Access to the SkillLab KU API (`KU_API_URL`) for KU forecasting.

---

### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/skillab-project/skill-ageing.git
   cd skill-ageing
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your `.env` file:**

   ```env
   TRACKER_API=https://skillab-tracker.csd.auth.gr/api
   TRACKER_USERNAME=your_username
   TRACKER_PASSWORD=your_password
   KU_API_URL=https://your-ku-api-host/api   # Optional, for KU forecasting
   ```

---

## Running the Application

### Locally

```bash
uvicorn ageing_forecasting:app --host 0.0.0.0 --port 8000 --reload
```

The API will be accessible at `http://localhost:8000`. Swagger UI is at `http://localhost:8000/docs`.

### With Docker

```bash
docker-compose up --build
```

Or manually:

```bash
docker build -t skill-ageing .
docker run -p 8008:8000 --env-file .env skill-ageing
```

---

## API Endpoints

All forecast endpoints are prefixed with `/forecast`. Each response includes a `summary`, per-skill `results` (with history and predictions), and a `skipped` list of skills that had insufficient data.

Each skill's result contains:

```json
{
  "method": "arima | trend | moving_average",
  "history_total": 42,
  "history": [{"date": "2023-01", "count": 3}, ...],
  "prediction": [{"date": "2024-07", "absolute": 4.2, "share": 0.031}, ...]
}
```

### `GET /forecast/jobs_skill_forecast_NEWONE`

Forecast ESCO skill demand from job postings.

| Parameter         | Default | Description                                         |
|-------------------|---------|-----------------------------------------------------|
| `keywords`        | —       | Comma-separated keywords                            |
| `occupation_ids`  | —       | Comma-separated ESCO occupation IDs                 |
| `source`          | —       | Optional job source filter (e.g. `linkedin`)        |
| `min_upload_date` | —       | `YYYY-MM-DD`                                        |
| `max_upload_date` | —       | `YYYY-MM-DD`                                        |
| `horizon`         | `6`     | Forecast horizon in months (3, 6, or 12)            |

### `GET /forecast/policy_skill_forecast`

Forecast ESCO skill demand from EU law/policy documents (auto-paginates `eur_lex`).

| Parameter   | Default | Description                              |
|-------------|---------|------------------------------------------|
| `keywords`  | required| Comma-separated keywords                 |
| `horizon`   | `6`     | Forecast horizon in months               |

### `GET /forecast/ku_forecast_arima`

Forecast Knowledge Unit (KU) activity over time.

| Parameter      | Default | Description                              |
|----------------|---------|------------------------------------------|
| `horizon`      | `6`     | Forecast horizon in months               |
| `start_date`   | —       | `YYYY-MM` start filter                   |
| `end_date`     | —       | `YYYY-MM` end filter                     |
| `organization` | —       | Filter by organization name              |

---

## Forecasting Model

The model applies a three-tier strategy per skill series:

1. **ARIMA(1,1,1)** via `statsmodels` — used if the series has ≥ 5 data points and the resulting forecast has meaningful variance (range ≥ 0.3).
2. **Linear Trend** — estimated from the last 3 months' mean difference; used if ARIMA is rejected.
3. **Moving Average** — mean of the last 3 months; used as fallback if the series is flat.

Skills with fewer than 2 historical observations or no activity in the last 4 months are skipped.

---

## Running the Tests

```bash
pytest tests/
```

---

## Project Structure

```
skill-ageing/
├── ageing_forecasting.py      # FastAPI router, forecasting logic, all endpoints
├── skill_api_without_cred.py  # Utility script for credential-free API exploration
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container image definition
├── docker-compose.yml         # Compose configuration
├── .env                       # Environment variables (fill in before running)
├── Completed_Analyses/        # Auto-created cache folder for completed forecasts
├── jenkins/                   # CI/CD pipeline configuration
└── tests/                     # Test suite
```

---

## Technologies

- **Python 3.11**
- **FastAPI** — REST API framework
- **Uvicorn** — ASGI server
- **statsmodels** — ARIMA time series forecasting
- **pandas / NumPy** — Time series construction and processing
- **python-dotenv** — Environment variable management
- **Docker / Docker Compose** — Containerised deployment

---

## License

This project is licensed under the **Eclipse Public License 2.0 (EPL-2.0)**. See the [LICENSE](LICENSE) file for details.
