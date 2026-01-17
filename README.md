# Air Quality Health Impact XAI Dashboard

A role-based Explainable AI (XAI) dashboard for interpreting air quality health impact predictions using XGBoost and SHAP.

## Installation

### Prerequisites
- Python 3.8+
- Required packages (install via pip):

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost shap dice-ml joblib
```

### Required Files
Ensure these files are in the same directory as `streamlit_xai_dashboard.py`:
- `xgboost_model.pkl` - Trained XGBoost model
- `scaler.pkl` - Fitted StandardScaler
- `air_quality_health_impact_data.csv` - Dataset

## Usage

### Running the Dashboard

1. Navigate to the project directory:
```bash
cd "workspace/"
```

2. Run the Streamlit app:
```bash
streamlit run streamlit_xai_dashboard.py
```

3. The dashboard will open in your default web browser (typically at `http://localhost:8501`)

### Using the Dashboard

1. **Select Your Role**: Use the sidebar to choose your role (Scientist, Regulator, Public Health Officer, Clinician, or Public User)

2. **Interact with Visualizations**:
   - Each role has customized views and controls
   - Use dropdowns, sliders, and buttons to interact with the data
   - SHAP plots are generated dynamically based on your selections

3. **Interpret Results**:
   - Red/Pink colors in SHAP plots indicate factors that increase health risk
   - Blue/Green colors indicate factors that decrease health risk
   - Larger absolute SHAP values mean stronger feature influence


## Features Analyzed

- **AQI**: Air Quality Index
- **PM10**: Particulate Matter (10 micrometers)
- **PM2_5**: Particulate Matter (2.5 micrometers)
- **NO2**: Nitrogen Dioxide
- **SO2**: Sulfur Dioxide
- **O3**: Ozone
- **Temperature**: Ambient temperature
- **Humidity**: Relative humidity
- **WindSpeed**: Wind speed
- **RespiratoryCases**: Number of respiratory cases
- **CardiovascularCases**: Number of cardiovascular cases
- **HospitalAdmissions**: Number of hospital admissions

## Technical Details

### Model
- **Algorithm**: XGBoost Classifier
- **Preprocessing**: StandardScaler normalization
- **Classes**: 5 (0-4, representing health impact severity)

### XAI Methods
- **SHAP (SHapley Additive exPlanations)**: 
  - TreeExplainer for XGBoost
  - Global explanations (beeswarm, bar plots)
  - Local explanations (waterfall, force plots)
  - Aggregated local explanations
  
- **DiCE (Diverse Counterfactual Explanations)**:
  - Genetic algorithm method
  - Shows minimal changes needed to alter predictions



