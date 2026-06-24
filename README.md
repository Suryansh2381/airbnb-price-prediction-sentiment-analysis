# Airbnb Price Prediction & Review Sentiment Analysis

An end-to-end machine learning project that predicts Airbnb listing prices in New York City and analyzes guest review sentiment through an interactive Streamlit web application.

---

## Results

| Model           | MAE    | RMSE   | R²     |
|----------------|--------|--------|--------|
| Random Forest   | 0.2750 | 0.3620 | 0.6477 |
| LightGBM        | 0.2756 | 0.3596 | 0.6525 | 
| XGBoost (Tuned) | **0.2707** | **0.3554** | **0.6605** |

> Metrics reported on log-transformed price. XGBoost selected as final model.

---

## Tech Stack
Python, Pandas, Scikit-Learn, XGBoost, LightGBM, SHAP, VADER, Streamlit

---

## Run Locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Dataset

- [NYC Airbnb Open Data (Listings)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
-  [New York City Airbnb Reviews](https://www.kaggle.com/datasets/thedevastator/discovering-new-york-city-through-airbnb-user-re)
---

## Project Structure
```
├── 01_Airbnb_Price_Prediction.ipynb
├── 02_Sentiment.ipynb
├── app.py
├── requirements.txt
├── airbnb_price_model.pkl
├── neighbourhood_map.pkl
└── neighbourhood_coords.pkl
```

## Screenshots

### Price Prediction

![Price Prediction](<screenshots/Demo_price.png>)

### Sentiment Analysis

![Sentiment Analysis](<screenshots/Demo_sentiment.png>)



##  Demo Video

Watch the project demo here:

[Watch Demo Video](https://youtu.be/Ck1-vlmfqII)
