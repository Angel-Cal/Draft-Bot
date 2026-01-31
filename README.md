# Fantasy Football Draft Bot

An ML-powered fantasy football draft assistant that recommends optimal picks based on player projections, value-based drafting, and roster construction strategies.

## Features

- **Player Projections**: Machine learning models trained on historical NFL data to predict fantasy performance
- **Value-Based Drafting (VBD)**: Compare players across positions using value over replacement
- **Draft Simulation**: Monte Carlo simulations to optimize draft strategy
- **Interactive CLI**: Real-time draft assistance with smart recommendations
- **Configurable Scoring**: Support for PPR, Half-PPR, and Standard leagues

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DraftBot.git
cd DraftBot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# 1. Collect historical NFL data
python -m src.app.cli collect --seasons 2022 2023 2024

# 2. Process and clean the data
python -m src.app.cli process

# 3. Train projection models
python -m src.app.cli train

# 4. View player rankings
python -m src.app.cli rankings --top 20

# 5. Start draft assistant
python -m src.app.cli draft --pick 5 --teams 12
```

## Project Structure

```
DraftBot/
├── data/
│   ├── raw/              # Original scraped data
│   ├── processed/        # Cleaned/transformed data
│   └── external/         # Third-party data sources
├── notebooks/            # Jupyter notebooks for analysis
├── src/
│   ├── data/            # Data collection and processing
│   │   ├── scraper.py   # Web scraping utilities
│   │   └── pipeline.py  # ETL pipeline
│   ├── features/        # Feature engineering
│   │   └── build_features.py
│   ├── models/          # ML models
│   │   ├── train.py     # Model training
│   │   └── predict.py   # Prediction generation
│   ├── draft/           # Draft strategy
│   │   ├── vbd.py       # Value-based drafting
│   │   ├── simulator.py # Draft simulation
│   │   └── recommender.py
│   └── app/             # Application interfaces
│       └── cli.py       # Command-line interface
├── models/              # Saved model files
├── tests/               # Unit tests
├── config/
│   └── settings.yaml    # Configuration
├── requirements.txt
└── README.md
```

## Configuration

Edit `config/settings.yaml` to customize:

- League settings (team count, roster positions)
- Scoring rules (PPR, standard, custom)
- Model parameters
- Draft settings

## Machine Learning Approach

### Data Pipeline
1. Scrape historical player statistics from Pro Football Reference
2. Clean and normalize data across seasons
3. Engineer predictive features (rolling averages, efficiency metrics, age curves)

### Models
- **Baseline**: Ridge/Lasso Regression
- **Ensemble**: Random Forest, Gradient Boosting
- **Evaluation**: MAE, RMSE, R² with time-series cross-validation

### Draft Strategy
- Value-Based Drafting (VBD) for cross-position comparison
- Positional scarcity adjustments
- Monte Carlo simulation for pick optimization
- Real-time roster need analysis

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Adding New Features

1. Add feature logic to `src/features/build_features.py`
2. Update the pipeline in `src/data/pipeline.py`
3. Retrain models with `python -m src.app.cli train`

## Roadmap

- [ ] Web interface with React frontend
- [ ] ESPN/Yahoo/Sleeper API integration
- [ ] Auction draft support
- [ ] Dynasty/keeper league features
- [ ] In-season lineup optimization
- [ ] Reinforcement learning for draft strategy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

---

Built as a machine learning portfolio project demonstrating:
- Data engineering and ETL pipelines
- Feature engineering for predictive modeling
- Ensemble machine learning methods
- Simulation and optimization algorithms
- Production-ready Python application structure
