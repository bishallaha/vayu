# VAYU – Air Quality Forecasting and Health Risk Assessment

VAYU aims to provide air quality and pollution-related information to help assess environmental conditions and their potential health impacts across several Indian cities. The data is used to predict future Air Quality Index (AQI) levels and interpret the risk levels of the AQI for people in several Indian cities. The project uses historical AQI data, weather data, machine learning, time-series forecasting, and Explainable Artificial Intelligence (XAI).

The system employs two approaches to forecast the air quality index (AQI) for the the next 24 hours, namely Facebook Prophet and XGBoost Regressor. An automatic model selection algorithm evaluates the accuracy of these approaches for each city and selects the most accurate one for subsequent use. In addition to numerical forecasts, an XGBoost Classifier is used to predict the class of AQI, which can be Good, Fair, Moderate, Poor, or Hazardous. The contribution of each factor to the accuracy of forecasts is explained using SHAP (SHapley Additive Explanations) values.

In addition to the prediction feature, the proposed system will provide a method for calculating health risk based on World Health Organization (WHO) air quality guidelines. It means that the pollution levels detected by the system will be automatically converted into a special index ranging from 0 to 100 (Health Risk Index). The index will display health-related recommendations for kids, elderly people, and individuals with respiratory diseases. It will also offer the calculation of the health risk based on WHO recommendation. It means that the pollution data detected by the system will be automatically converted into a special index, varying from 0 to 100 (Health Risk Index). This index will also provide information about health advice for children, elderly people, and people with respiratory diseases.


## Features

- Automated AQI and weather data collection
- Data cleaning and preprocessing pipeline
- Feature engineering using lag and rolling statistics
- AQI forecasting using Prophet
- AQI forecasting using XGBoost Regressor
- Automatic best-model selection for each city
- AQI category prediction using XGBoost Classifier
- SHAP-based model explainability
- WHO-aligned Health Risk Index (0–100)
- Demographic-specific health advisories
- SQLite database integration
- Interactive dashboard (under development)


## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Prophet
- SHAP
- SQLite
- Matplotlib


## Project Status

The backend data processing and machine learning pipeline have been completed successfully. Current work focuses on developing an interactive dashboard to visualize forecasts, health risk scores, model explanations, and air quality trends in a user-friendly interface.