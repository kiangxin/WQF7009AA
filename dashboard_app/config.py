from __future__ import annotations

from typing import Dict


# Role definitions (Clinician removed as requested)
ROLES: Dict[str, Dict[str, str]] = {
    "Scientist": {
        "icon": "🔬",
        "description": "Research and analysis of all health impact classes",
    },
    "Regulator": {
        "icon": "⚖️",
        "description": "Policy-making insights for safe air quality standards",
    },
    "Public Health Officer": {
        "icon": "🏥",
        "description": "Population-level health risk assessment",
    },
    "Public User": {
        "icon": "👤",
        "description": "Personal health risk awareness",
    },
}


# Health impact class descriptions (kept for alerts/text; dropdown labels are now "Class X")
CLASS_DESCRIPTIONS = {
    0: "Very High Health Impact (Score ≥ 80) - Emergency conditions, entire population affected",
    1: "High Health Impact (Score 60-79) - Health warnings, everyone may experience effects",
    2: "Moderate Health Impact (Score 40-59) - General population may experience health effects",
    3: "Low Health Impact (Score 20-39) - Sensitive groups may experience minor effects",
    4: "Very Low Health Impact (Score < 20) - Air quality is good, minimal risk",
}

# Short class labels for UI where you don't want long descriptions
CLASS_LABELS_SHORT = {
    0: "Very High Health Impact (Score ≥ 80)",
    1: "High Health Impact (Score 60-79)",
    2: "Moderate Health Impact (Score 40-59)",
    3: "Low Health Impact (Score 20-39)",
    4: "Very Low Health Impact (Score < 20)",
}


FEATURE_COLS = [
    "AQI",
    "PM10",
    "PM2_5",
    "NO2",
    "SO2",
    "O3",
    "Temperature",
    "Humidity",
    "WindSpeed",
    "RespiratoryCases",
    "CardiovascularCases",
    "HospitalAdmissions",
]


FEATURE_DESCRIPTIONS = {
    "AQI": "Air Quality Index - Overall measure of air pollution (higher = worse)",
    "PM10": "Particulate Matter < 10μm diameter (μg/m³)",
    "PM2_5": "Particulate Matter < 2.5μm diameter (μg/m³)",
    "NO2": "Nitrogen Dioxide concentration (ppb)",
    "SO2": "Sulfur Dioxide concentration (ppb)",
    "O3": "Ozone concentration (ppb)",
    "Temperature": "Ambient temperature (°C)",
    "Humidity": "Relative humidity (%)",
    "WindSpeed": "Wind speed (m/s)",
    "RespiratoryCases": "Number of respiratory cases reported",
    "CardiovascularCases": "Number of cardiovascular cases reported",
    "HospitalAdmissions": "Number of hospital admissions reported",
}

FEATURE_HIGH_EXPLANATION = {
    "AQI": {
        "reason": "air quality is poor, indicating high overall pollution levels",
        "advice": "consider staying indoors and limiting outdoor activities",
    },
    "PM10": {
        "reason": "coarse particulate matter levels are high, which can irritate the airways",
        "advice": "avoid prolonged outdoor exposure, especially near traffic or construction areas",
    },
    "PM2_5": {
        "reason": "fine particulate matter levels are high and can penetrate deep into the lungs",
        "advice": "wear a mask outdoors and reduce outdoor physical activity",
    },
    "NO2": {
        "reason": "nitrogen dioxide levels are high, often associated with traffic pollution",
        "advice": "avoid busy roads and limit time spent outdoors",
    },
    "O3": {
        "reason": "ground-level ozone concentration is high, which can worsen respiratory stress",
        "advice": "reduce outdoor activities, especially during peak daylight hours",
    },
    "SO2": {
        "reason": "sulfur dioxide levels are elevated and may cause airway irritation",
        "advice": "limit outdoor exposure and monitor air quality alerts",
    },
    "Temperature": {
        "reason": "ambient temperature is high, which may intensify the effects of air pollution",
        "advice": "stay hydrated and avoid outdoor activities during the hottest periods",
    },
    "Humidity": {
        "reason": "humidity levels are high, which can affect breathing comfort and pollutant dispersion",
        "advice": "reduce strenuous outdoor activities and stay in well-ventilated areas",
    },
    "WindSpeed": {
        "reason": "wind conditions may influence how pollutants accumulate in the air",
        "advice": "monitor air quality conditions before spending time outdoors",
    },
    "RespiratoryCases": {
        "reason": "the surrounding area shows a higher number of respiratory cases, indicating increased health burden",
        "advice": "take extra precautions and follow public health recommendations",
    },
    "CardiovascularCases": {
        "reason": "cardiovascular cases are elevated in the surrounding area, indicating increased population health risk",
        "advice": "avoid strenuous activities and follow public health guidance",
    },
    "HospitalAdmissions": {
        "reason": "hospital admissions are elevated, suggesting increased overall health stress in the area",
        "advice": "stay alert to health advisories and limit exposure to outdoor risk factors",
    },
}
