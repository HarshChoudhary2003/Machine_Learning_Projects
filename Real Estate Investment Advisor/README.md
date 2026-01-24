# Real Estate Investment Advisor 🏠

An AI-powered tool that analyzes property features to classify investment potential and predict future price appreciation.

## Features
- **Investment Classification:** Uses Random Forest/XGBoost to determine if a property is a "Good" or "Not Recommended" investment.
- **Price Prediction:** Predicts 5-year price appreciation using historical trends.
- **Interactive Input:** Sidebar for detailed property specifications (BHK, Size, Location, Amenities).
- **Visual Analytics:** ROI projections and feature importance charts.

## Key Files
- `app.py`: Main Streamlit application.
- `models/`: Pre-trained classification and regression models.
- `artifacts/`: Scalers, encoders, and feature metadata.

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset Information
The models are trained on real estate market data, incorporating features like density scores (schools/hospitals), price efficiency, and property age.
