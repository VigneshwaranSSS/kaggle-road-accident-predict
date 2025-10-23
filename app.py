# app.py (English Version)
import streamlit as st
import pandas as pd
import lightgbm as lgb
import joblib

# --- Load Model and Mappings ---
# Use @st.cache_resource to ensure the model is loaded only once for efficiency
@st.cache_resource
def load_model():
    model = joblib.load('deployment_lgbm.joblib')
    mappings = joblib.load('category_mappings.joblib')
    return model, mappings

model, mappings = load_model()

# --- Page Configuration ---
st.set_page_config(page_title="Accident Risk Predictor", layout="wide")
st.title("🚗 Road Safety Comparator")
st.write("Adjust the parameters below to compare the accident risk of two different routes. Our model will tell you which path is safer!")

# --- Define Input Options ---
road_type_options = list(mappings['road_type'].keys())
lighting_options = list(mappings['lighting'].keys())
weather_options = list(mappings['weather'].keys())
time_of_day_options = list(mappings['time_of_day'].keys())

# --- Create a two-column layout ---
col1, col2 = st.columns(2)

def create_ui(column, title):
    """Creates all UI components in a single column."""
    with column:
        st.header(title)
        road_type = st.selectbox("Road Type", road_type_options, key=f"road_{title}")
        lighting = st.selectbox("Lighting Condition", lighting_options, key=f"light_{title}")
        weather = st.selectbox("Weather Condition", weather_options, key=f"weather_{title}")
        time_of_day = st.selectbox("Time of Day", time_of_day_options, key=f"time_{title}")
        
        speed_limit = st.slider("Speed Limit (km/h)", 30, 120, 60, 5, key=f"speed_{title}")
        curvature = st.slider("Road Curvature", 0.0, 1.0, 0.5, 0.01, key=f"curve_{title}")
        num_lanes = st.slider("Number of Lanes", 1, 8, 2, 1, key=f"lanes_{title}")
        num_reported_accidents = st.slider("Number of Reported Accidents", 0, 50, 5, 1, key=f"acc_{title}")

        st.write("") # Add some space
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            road_signs_present = st.checkbox("Road Signs Present", value=True, key=f"signs_{title}")
            public_road = st.checkbox("Is a Public Road", value=True, key=f"public_{title}")
        with sub_col2:
            holiday = st.checkbox("Is a Holiday", value=False, key=f"holiday_{title}")
            school_season = st.checkbox("Is School Season", value=False, key=f"school_{title}")
        
        # Convert all 12 inputs to the numbers required by the model
        input_data = {
            'road_type': mappings['road_type'][road_type],
            'num_lanes': num_lanes,
            'curvature': curvature,
            'speed_limit': speed_limit,
            'lighting': mappings['lighting'][lighting],
            'weather': mappings['weather'][weather],
            'road_signs_present': int(road_signs_present), # Convert True/False to 1/0
            'public_road': int(public_road),
            'holiday': int(holiday),
            'school_season': int(school_season),
            'time_of_day': mappings['time_of_day'][time_of_day],
            'num_reported_accidents': num_reported_accidents
        }
        
        # Ensure the feature order is exactly the same as during training
        feature_order = [
            'road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting', 
            'weather', 'road_signs_present', 'public_road', 'holiday', 
            'school_season', 'time_of_day', 'num_reported_accidents'
        ]
        return pd.DataFrame([input_data])[feature_order]

# --- Generate and Display Predictions ---
df1 = create_ui(col1, "Route A")
df2 = create_ui(col2, "Route B")

pred1 = model.predict(df1)[0]
pred2 = model.predict(df2)[0]

st.divider() # Add a separator line

res_col1, res_col2 = st.columns(2)
with res_col1:
    st.metric(label="Route A Accident Risk", value=f"{pred1:.4f}")
    if pred1 < pred2:
        st.success("✅ This route is relatively safer.")
with res_col2:
    st.metric(label="Route B Accident Risk", value=f"{pred2:.4f}")
    if pred2 < pred1:
        st.success("✅ This route is relatively safer.")

if abs(pred1 - pred2) < 0.01:
    st.info("The risk levels of both routes are very similar.")