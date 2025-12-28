Bina.az Rent Price Prediction

Machine Learning Internship Project – This project collects rental apartment listings from Bina.az, cleans and preprocesses the data, and predicts monthly rental prices based on apartment features using Machine Learning regression models.

Project Structure
```text
bina_az_ml/
├── data/
│   ├── raw/bina_az_raw.csv         # Raw scraped data
│   ├── processed/bina_az_clean.csv # Cleaned & preprocessed data
├── api/main.py                     # FastAPI endpoint
├── model/
│   ├── model.joblib                # Trained ML model pipeline
│   └── train.ipynb                 # Model training notebook
├── notebooks/analysis.ipynb        # EDA & feature engineering
├── scraping/scrape_bina.ipynb      # Web scraping notebook
├── venv/
├── README.md                       # Project documentation
└── requirements.txt                # Required Python packages
```

Project Overview

This project predicts monthly rental prices of apartments in Azerbaijan based on location, area, number of rooms, floor, and building type.

Data Collection

Scraped using Selenium & BeautifulSoup (scraping/scrape_bina.ipynb)

7500+ rental listings collected

Raw data: data/raw/bina_az_raw.csv

Columns: location, rooms, area_m2, floor, total_floor, price, is_new_building

Data Cleaning & Features

Missing values handled, outliers removed

place encoded with One-Hot Encoding

Features: rooms, area_m2, floor, total_floor, is_new_building, place

Clean data: data/processed/bina_az_clean.csv

EDA: price distribution, area vs price, rooms vs price, correlation heatmap

Model Training

Train/Test split: 80/20

Models: Linear Regression, Random Forest, XGBoost, MLPRegressor

Best performance: XGBoost & Random Forest (R² ≈ 0.73)

Model saved: model/model.joblib

Place columns: model/place_columns.joblib

API Usage

FastAPI endpoint /predict (api/main.py)

{
  "rooms": 2,
  "area_m2": 70,
  "floor": 3,
  "total_floor": 10,
  "is_new_building": 1,
  "place": "Nərimanov q.",
  "model_name": "xgboost"
}


Run API:

uvicorn api.main:app --reload


Access: http://127.0.0.1:8000

Docs: http://127.0.0.1:8000/docs

How to Run
pip install -r requirements.txt
uvicorn api.main:app --reload
