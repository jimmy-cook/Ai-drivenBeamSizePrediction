import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import time

# -------------------------------------------------
# 1) Cache loading of model & scalers (once)
# -------------------------------------------------
@st.cache_resource
def load_model_once(model_path):
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_scalers(x_path, y_path):
    with open(x_path, 'rb') as fX:
        scaler_X = pickle.load(fX)
    with open(y_path, 'rb') as fY:
        scaler_Y = pickle.load(fY)
    return scaler_X, scaler_Y

model = load_model_once("my_zwethebestmodel.h5")
scaler_X, scaler_Y = load_scalers("scaler_X.pkl", "scaler_Y.pkl")

# -------------------------------------------------
# 2) Sidebar Nav: 3 Pages
# -------------------------------------------------
page_selection = st.sidebar.radio(
    "Go to Page:",
    ["Home", "Notation"]

)

# -------------------------------------------------
# PAGE 1: HOME
# -------------------------------------------------
if page_selection == "Home":


    st.title("Welcome to Ai-Driven RC Beam size Prediction Web App!")
    
    st.image("Ailogo.png")
    # --- User Inputs ---
    st.subheader("Material Properties")
    concrete_grade = st.number_input("Concrete Grade (fc in MPa)", value=20.0)
    rebar_grade    = st.number_input("Rebar Grade (fy in MPa)", value=415.0)

    st.subheader("Load Inputs")
    dead_load = st.number_input("Dead Load (kN/m)", value=10)
    live_load = st.number_input("Live Load (kN/m)", value=15)

    st.subheader("Span Details")
    span_length     = st.number_input("Span Length (mm)", value=5000)
    left_span_ratio = st.number_input("Left Span Ratio", value=1.0)
    right_span_ratio= st.number_input("Right Span Ratio", value=1.0)

    st.subheader("Span Type")
    span_type_choice = st.radio("Interior or Exterior Span?", ["Interior", "Exterior"])
    span_type = 1 if span_type_choice == "Interior" else 0

    #st.subheader("Optimization")
    #opt_checkbox = st.checkbox("Optimized?", value=False)
    # If your model logic requires 0 for checked, 1 for unchecked, keep as is:
    optimized = 1# if opt_checkbox else 0

    # Calculate total factored load
    total_factored_load = 1.2 * dead_load + 1.6 * live_load

    # Predict button
    if st.button("Predict"):
        # Build DataFrame with the exact columns used in training
        # Show a spinner and artificially wait (e.g., 2 seconds) 
        with st.spinner("AI is crunching your data...🤖"):
        # Simulate a short delay, or do your real heavy-lifting here
          time.sleep(3.6)
        input_data = pd.DataFrame({
            "Total Factored Load": [total_factored_load],
            "concrete grade":      [concrete_grade],
            "steel grade":         [rebar_grade],
            "dead load":           [dead_load],
            "live load":           [live_load],
            "span length":         [span_length],
            "left span ratio":     [left_span_ratio],
            "right span ratio":    [right_span_ratio],
            "span type ":          [span_type],  # note trailing space
            "optimized":           [optimized]
        })

        # Scale
        input_data_scaled = scaler_X.transform(input_data)

        # Predict
        prediction = model.predict(input_data_scaled)
        prediction_scaled = scaler_Y.inverse_transform(prediction)

        predicted_width = prediction_scaled[0][0]
        predicted_depth = prediction_scaled[0][1]

        # Extra condition if needed
        if dead_load > 200 and live_load > 100:
            predicted_width *= 1.1164
            predicted_depth *= 1.35

        st.subheader(
            f"**Predicted Cross Section size for RC beam:**\n\n"
            f"Width =  {round(predicted_width)} mm, Depth = {round(predicted_depth)} mm"

        )
    st.image("TFlogo.png")
# -------------------------------------------------
# PAGE 3: ABOUT
# -------------------------------------------------
else:  # "About"
    st.title("Notation")
    st.image("beamspan.jpg")
    st.write(
        """
        
        This research focuses on developing an optimal cross section size prediction model for reinforced concrete (RC)
        continuous beams using an artificial neural network (ANN), specifically for use in preliminary design. 
        Under the expert guidance of Dr. Naveed Anwar, CEO of CSI Bangkok, the resulting model aims to streamline
        early design decisions, enhancing both the efficiency and safety of RC beam design.
    

        **Notation**:
        - Dead load, live load (KN/m)
        - Concrete (MPa), Rebar (MPa)
        - Span Details: 
        - Span length, 
        - interior/exterior,
        - left span ratio  Li-1/L 
        - right span ratio  Li+1/L
        - no left or right span - "0"     
        - Optionally toggling "Optimized" affects the final output

        **Model Info**:
        - Built in TensorFlow2.0/Keras 
        - Scaled with scikit-learn pickled scalers
        - Deployed using Streamlit

        **Developer Info**:
        - Zwe Yan Naing
        - MEng.Structural Engineering.
        - Asian Institute Of Technology.

        -zweyannaing166@gmail.com
        """
    )

