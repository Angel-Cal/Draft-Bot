# Fantasy Football Draft Bot - Project Plan

## Project Overview
An ML-powered fantasy football draft assistant that recommends optimal picks based on player projections, draft position, and league settings.

---

## Phase 1: Foundation & Data Collection

### 1.1 Project Setup
- [ ] Initialize Git repository
- [ ] Set up Python virtual environment
- [ ] Create project structure
- [ ] Set up dependency management (requirements.txt or pyproject.toml)

### 1.2 Data Collection
- [ ] Identify data sources:
  - Historical player statistics (NFL API, Pro Football Reference, ESPN)
  - Fantasy points scoring (by league type: PPR, Standard, Half-PPR)
  - ADP (Average Draft Position) data
  - Injury history
  - Team/schedule data
- [ ] Build data scraping/API integration scripts
- [ ] Create data storage solution (SQLite for simplicity, or PostgreSQL)
- [ ] Document data dictionary

### 1.3 Data Pipeline
- [ ] ETL pipeline for historical data (3-5 seasons)
- [ ] Data cleaning and normalization
- [ ] Feature engineering pipeline

---

## Phase 2: Exploratory Analysis & Feature Engineering

### 2.1 Exploratory Data Analysis
- [ ] Analyze player performance trends
- [ ] Identify key predictive features
- [ ] Visualize correlations between stats and fantasy output
- [ ] Document insights in Jupyter notebooks

### 2.2 Feature Engineering
- [ ] Create derived features:
  - Rolling averages (3-game, 5-game, season)
  - Year-over-year trends
  - Age curves
  - Target share / snap counts
  - Red zone opportunities
  - Strength of schedule
- [ ] Encode categorical variables (position, team)
- [ ] Handle missing data strategies

---

## Phase 3: Machine Learning Models

### 3.1 Player Projection Model
- [ ] Define target variable (fantasy points per game)
- [ ] Train/test split (by season to avoid leakage)
- [ ] Baseline models:
  - Linear Regression
  - Random Forest
  - Gradient Boosting (XGBoost/LightGBM)
- [ ] Hyperparameter tuning
- [ ] Model evaluation metrics (MAE, RMSE, R²)
- [ ] Position-specific models vs unified model comparison

### 3.2 Uncertainty Quantification
- [ ] Implement prediction intervals
- [ ] Model player "boom/bust" potential
- [ ] Risk-adjusted projections

### 3.3 Model Persistence
- [ ] Save trained models (pickle/joblib)
- [ ] Version control for models
- [ ] Model retraining pipeline

---

## Phase 4: Draft Strategy Engine

### 4.1 Value-Based Drafting (VBD)
- [ ] Implement VBD algorithm
- [ ] Calculate position scarcity
- [ ] Dynamic value updates as draft progresses

### 4.2 Draft Simulation
- [ ] Build draft simulator
- [ ] Implement opponent modeling (ADP-based picks)
- [ ] Monte Carlo simulations for pick optimization

### 4.3 Roster Construction
- [ ] League settings configuration (roster spots, scoring)
- [ ] Positional requirements enforcement
- [ ] Bye week optimization
- [ ] Team stacking/anti-correlation strategies

---

## Phase 5: Application Development

### 5.1 Core Application
- [ ] CLI interface for draft assistance
- [ ] Real-time pick tracking
- [ ] Recommendation engine integration

### 5.2 Web Interface (Optional Enhancement)
- [ ] FastAPI/Flask backend
- [ ] React/simple HTML frontend
- [ ] Real-time draft board visualization
- [ ] Pick recommendations with explanations

### 5.3 Integration Options
- [ ] ESPN/Yahoo/Sleeper API integration for live drafts
- [ ] Export/import draft results

---

## Phase 6: Testing & Documentation

### 6.1 Testing
- [ ] Unit tests for core functions
- [ ] Integration tests for pipelines
- [ ] Model validation tests
- [ ] Backtesting on historical drafts

### 6.2 Documentation
- [ ] README with project overview
- [ ] Setup and installation guide
- [ ] API documentation
- [ ] Model methodology write-up
- [ ] Jupyter notebooks showcasing analysis

---

## Phase 7: Portfolio Presentation

### 7.1 Resume-Ready Deliverables
- [ ] Clean GitHub repository with good commit history
- [ ] Comprehensive README with visuals
- [ ] Demo video or GIF
- [ ] Blog post explaining approach
- [ ] Performance metrics and results

---

## Suggested Project Structure

```
DraftBot/
├── data/
│   ├── raw/              # Original data files
│   ├── processed/        # Cleaned/transformed data
│   └── external/         # Third-party data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_draft_simulation.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── scraper.py
│   │   └── pipeline.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── draft/
│   │   ├── __init__.py
│   │   ├── vbd.py
│   │   ├── simulator.py
│   │   └── recommender.py
│   └── app/
│       ├── __init__.py
│       ├── cli.py
│       └── api.py
├── models/               # Saved model files
├── tests/
├── config/
│   └── settings.yaml
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| ML Framework | Scikit-learn, XGBoost/LightGBM |
| Deep Learning (optional) | PyTorch/TensorFlow |
| Database | SQLite (dev), PostgreSQL (prod) |
| API | FastAPI |
| Visualization | Matplotlib, Seaborn, Plotly |
| Testing | Pytest |
| Notebooks | Jupyter |

---

## Key ML Concepts Demonstrated

1. **Supervised Learning** - Player projection models
2. **Feature Engineering** - Creating predictive features from raw stats
3. **Time Series Considerations** - Proper train/test splits to avoid leakage
4. **Ensemble Methods** - Combining multiple models
5. **Hyperparameter Optimization** - Grid search, cross-validation
6. **Uncertainty Quantification** - Prediction intervals
7. **Simulation/Monte Carlo** - Draft optimization
8. **Reinforcement Learning (stretch goal)** - Learning draft strategy through simulation

---

## Timeline Suggestion

| Phase | Duration |
|-------|----------|
| Phase 1: Foundation | 1-2 weeks |
| Phase 2: EDA & Features | 1-2 weeks |
| Phase 3: ML Models | 2-3 weeks |
| Phase 4: Draft Engine | 1-2 weeks |
| Phase 5: Application | 1-2 weeks |
| Phase 6: Testing & Docs | 1 week |
| Phase 7: Portfolio Polish | 1 week |

---

## Getting Started - First Steps

1. Initialize the repository and project structure
2. Find and document your data sources
3. Start with a simple data pipeline for one season
4. Build a baseline model before optimizing
5. Iterate and improve incrementally

---

## Stretch Goals

- [ ] Reinforcement learning agent that learns draft strategy
- [ ] Dynasty/keeper league support
- [ ] Auction draft optimization
- [ ] Trade value calculator
- [ ] In-season lineup optimization
- [ ] Natural language explanations for picks (LLM integration)
