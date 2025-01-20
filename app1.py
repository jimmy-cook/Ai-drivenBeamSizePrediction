import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
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
page_selection = st.sidebar.radio("Go to Page:", ["Home", "Chatbot", "About"])

# -------------------------------------------------
# PAGE 1: HOME
# -------------------------------------------------
if page_selection == "Home":
    st.title("Welcome to AI-Driven RC Beam Size Prediction Web App!")
    st.image("Ailogo.png")
    
    # --- User Inputs ---
    st.subheader("Material Properties")
    concrete_grade = st.number_input("Concrete Grade (fc in MPa)", value=25.0)
    rebar_grade = st.number_input("Rebar Grade (fy in MPa)", value=415.0)

    st.subheader("Load Inputs")
    dead_load = st.number_input("Dead Load (kN/m)", value=10.0)
    live_load = st.number_input("Live Load (kN/m)", value=15.0)

    st.subheader("Span Details")
    span_length = st.number_input("Span Length (mm)", value=6000)
    left_span_ratio = st.number_input("Left Span Ratio", value=1.0)
    right_span_ratio = st.number_input("Right Span Ratio", value=1.0)

    st.subheader("Span Type")
    span_type_choice = st.radio("Interior or Exterior Span?", ["Interior", "Exterior"])
    span_type = 1 if span_type_choice == "Interior" else 0

    optimized = 1  # Default optimization

    # Calculate total factored load
    total_factored_load = 1.2 * dead_load + 1.6 * live_load

    # Predict button
    if st.button("Predict"):
        # Build DataFrame
        with st.spinner("AI is crunching your data...🤖"):
            time.sleep(3.6)  # Simulate a delay
            input_data = pd.DataFrame({
                "Total Factored Load": [total_factored_load],
                "concrete grade": [concrete_grade],
                "steel grade": [rebar_grade],
                "dead load": [dead_load],
                "live load": [live_load],
                "span length": [span_length],
                "left span ratio": [left_span_ratio],
                "right span ratio": [right_span_ratio],
                "span type ": [span_type],
                "optimized": [optimized],
            })

            # Scale and predict
            input_data_scaled = scaler_X.transform(input_data)
            prediction = model.predict(input_data_scaled)
            prediction_scaled = scaler_Y.inverse_transform(prediction)

            # Extra condition if needed
          
            predicted_width = round(prediction_scaled[0][0])
            predicted_depth = round(prediction_scaled[0][1])

            
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
## -------------------------------------------------
# PAGE 2: CHATBOT
# -------------------------------------------------

elif page_selection == "Chatbot":
    st.title("Welcome to AI-Driven RC Beam Size Prediction Chat bot!🤖")
    if "step" not in st.session_state:
        st.session_state.step = 0
        st.session_state.inputs = {
            "concrete_grade": None,
            "rebar_grade": None,
            "dead_load": None,
            "live_load": None,
            "span_length": None,
            "left_span_ratio": None,
            "right_span_ratio": None,
            "span_type": None,
            "optimized": 1
        }
        st.session_state.predicted = False

    questions = [
        "What is the concrete grade (fc in MPa)?",
        "What is the rebar grade (fy in MPa)?",
        "What is the dead load (in kN/m)?",
        "What is the live load (in kN/m)?",
        "What is the span length (in mm)?",
        "What is the left span ratio?",
        "What is the right span ratio?",
        "Is the span type interior or exterior? (Type interior or exterior without quotes)"
    ]

    input_keys = [
        "concrete_grade",
        "rebar_grade",
        "dead_load",
        "live_load",
        "span_length",
        "left_span_ratio",
        "right_span_ratio",
        "span_type"
    ]

    def process_input():
        key = input_keys[st.session_state.step]
        user_input = st.session_state.get(f"input_{st.session_state.step}", "").strip()

        if key == "span_type":
            cleaned_input = user_input.lower()
            if cleaned_input in ["interior", "exterior"]:
                st.session_state.inputs[key] = 1 if cleaned_input == "interior" else 0
            else:
                st.error("Please enter 'interior' or 'exterior' (without quotes).")
                return
        else:
            try:
                st.session_state.inputs[key] = float(user_input)
            except ValueError:
                st.error("Please enter a valid numeric value.")
                return

        st.session_state.step += 1
        st.experimental_rerun()

    if st.session_state.step < len(questions):
        question = questions[st.session_state.step]
        st.markdown(f"<h3>{question}</h3>", unsafe_allow_html=True)
        st.text_input("Your Answer:", key=f"input_{st.session_state.step}", on_change=process_input)

    else:
        if not st.session_state.predicted:
            total_factored_load = (
                1.2 * st.session_state.inputs["dead_load"]
                + 1.6 * st.session_state.inputs["live_load"]
            )
            input_data = pd.DataFrame({
                "Total Factored Load": [total_factored_load],
                "concrete grade": [st.session_state.inputs["concrete_grade"]],
                "steel grade": [st.session_state.inputs["rebar_grade"]],
                "dead load": [st.session_state.inputs["dead_load"]],
                "live load": [st.session_state.inputs["live_load"]],
                "span length": [st.session_state.inputs["span_length"]],
                "left span ratio": [st.session_state.inputs["left_span_ratio"]],
                "right span ratio": [st.session_state.inputs["right_span_ratio"]],
                "span type ": [st.session_state.inputs["span_type"]],
                "optimized": [st.session_state.inputs["optimized"]],
            })

            input_data_scaled = scaler_X.transform(input_data)
            prediction = model.predict(input_data_scaled)
            prediction_scaled = scaler_Y.inverse_transform(prediction)

            predicted_width = round(prediction_scaled[0][0])
            predicted_depth = round(prediction_scaled[0][1])

            # Check condition for dead load and live load
            if (
                st.session_state.inputs["dead_load"] > 200
                and st.session_state.inputs["live_load"] > 100
            ):
                predicted_width = round(predicted_width * 1.1164)
                predicted_depth = round(predicted_depth * 1.35)

            st.session_state.prediction_result = (
                f"**Predicted Cross Section size for RC beam:**\n\n"
                f"Width = {predicted_width} mm, Depth = {predicted_depth} mm"
            )
            st.session_state.predicted = True
            st.experimental_rerun()

        else:
            st.title("AI Powered Prediction")
            st.subheader(st.session_state.prediction_result)
            if st.button("Reset"):
                st.session_state.step = 0
                st.session_state.inputs = {k: None for k in st.session_state.inputs}
                st.session_state.predicted = False
                st.experimental_rerun()


# -------------------------------------------------
# PAGE 3: ABOUT
# -------------------------------------------------
else:
    st.title("About")
    st.image("beamspan.jpg")
    st.write(
        """
        
        This research focuses on developing an optimal cross section size prediction model for reinforced concrete (RC)
        continuous beams using an artificial neural network (ANN), specifically for using in preliminary design. 
        Under the expert guidance of **Dr. Naveed Anwar**, CEO of CSI Bangkok, the resulting model aims to streamline
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


        **Model Info**:
        - Built in TensorFlow2.0/Keras 
        - Scaled with scikit-learn pickled scalers
        - Deployed using Streamlit

        **Developer Info**:
        - **Zwe Yan Naing**
        - MEng.Structural Engineering.
        - Asian Institute Of Technology.

        -📧zweyannaing166@gmail.com
        """
    )

