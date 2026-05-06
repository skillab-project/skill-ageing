# Contributing to Skill Ageing

Thank you for your interest in contributing! This document outlines the process for reporting issues, proposing changes, and submitting code.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Submitting Code Changes](#submitting-code-changes)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project is part of the [SkillLab](https://github.com/skillab-project) EU Horizon Europe research initiative. All contributors are expected to engage respectfully and constructively. Harassment or disruptive behaviour of any kind will not be tolerated.

---

## Getting Started

Before contributing, please:

1. Read the [README](README.md) to understand the forecasting pipeline and the ARIMA → Trend → Moving Average fallback strategy.
2. Check the [open issues](https://github.com/skillab-project/skill-ageing/issues) to see if your bug or idea has already been raised.
3. For significant changes — especially to the forecasting model, series selection criteria, or normalisation logic — open an issue first to discuss your approach.

---

## How to Contribute

### Reporting Bugs

Open an issue and include:

- A clear title describing the problem.
- Steps to reproduce, including query parameters used.
- Expected vs. actual behaviour.
- Environment details: OS, Python version, Docker version if applicable.
- Any error messages or stack traces.

### Suggesting Enhancements

Describe the use case, your proposed solution, and any alternatives. Particularly welcome contributions include: additional forecasting models (e.g. Prophet, ETS, LSTM), new data source endpoints (e.g. profiles, courses), and improvements to the series selection or outlier handling.

### Submitting Code Changes

All contributions are made via **Pull Requests**. See [Pull Request Process](#pull-request-process) below.

---

## Development Setup

1. **Fork and clone your fork:**

   ```bash
   git clone https://github.com/<your-username>/skill-ageing.git
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

4. **Configure `.env`:**

   ```env
   TRACKER_API=https://skillab-tracker.csd.auth.gr/api
   TRACKER_USERNAME=your_username
   TRACKER_PASSWORD=your_password
   KU_API_URL=https://your-ku-api-host/api
   ```

5. **Start the development server:**

   ```bash
   uvicorn ageing_forecasting:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) for all Python code.
- Add docstrings to new functions and classes.
- Suppress `statsmodels` warnings in production code with `warnings.filterwarnings("ignore")` — this is already done in the existing codebase.
- Do not commit credentials or secrets. Use `.env` for all configuration.
- Do not commit `Completed_Analyses/` cache files.

---

## Testing

```bash
pytest tests/
```

When contributing:

- Add or update tests for any new or changed behaviour.
- Ensure all existing tests pass before opening a PR.
- For new forecasting models, include tests that verify the output structure (`method`, `history`, `prediction`, `share` fields) and that the fallback chain works as expected.

---

## Commit Message Guidelines

```
<type>: <short summary>
```

| Type       | When to use                                      |
|------------|--------------------------------------------------|
| `feat`     | A new endpoint or forecasting model              |
| `fix`      | A bug fix                                        |
| `refactor` | Code restructuring without behaviour change      |
| `test`     | Adding or updating tests                         |
| `docs`     | Documentation changes only                      |
| `chore`    | Dependency updates, CI config, tooling           |

Examples:

```
feat: add Prophet as optional forecasting backend
fix: handle empty series after date filtering
docs: add forecasting model description to README
```

---

## Pull Request Process

1. **Branch naming:** e.g. `feat/prophet-forecaster` or `fix/empty-series-crash`.
2. **Keep PRs focused:** One logical change per PR.
3. **Fill in the PR description** with what changed, why, and how it was tested.
4. **Link related issues** using `Closes #<issue-number>`.
5. **Review:** At least one maintainer review is required.
6. **CI:** All automated checks must pass.

---

## Questions

Open a [discussion or issue](https://github.com/skillab-project/skill-ageing/issues) if you have questions not covered here.
