import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import pandas as pd
from PIL import Image

# -----------------------------
# Load Model and Files
# -----------------------------
model = tf.keras.models.load_model("model/food_model.keras")

with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

nutrition_df = pd.read_csv("nutrition/nutrition.csv")

with open("diet/diet_recommendations.json", "r") as f:
    diet_data = json.load(f)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_food(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(prediction[0])
    predicted_food = class_names[predicted_index]
    confidence = float(np.max(prediction[0]) * 100)

    return predicted_food, confidence


def calculate_bmi(height_cm, weight_kg):
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 2)

    if bmi < 18.5:
        status = "Underweight"
    elif bmi < 25:
        status = "Normal Weight"
    elif bmi < 30:
        status = "Overweight"
    else:
        status = "Obese"

    return bmi, status


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Indian Food Recognition", layout="wide")

st.title("🍛 Indian Food Recognition System")
st.write("Upload a food image to predict the food and view nutrition details.")

uploaded_file = st.file_uploader("Upload Food Image", type=["jpg", "jpeg", "png"])

diet_goal = st.selectbox(
    "Select Diet Goal",
    ["weight_loss", "weight_gain", "muscle_gain", "diabetes", "heart_healthy"]
)

st.subheader("BMI Calculator")
height = st.number_input("Height (cm)", min_value=1.0)
weight = st.number_input("Weight (kg)", min_value=1.0)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, width=300)

    if st.button("Predict Food"):
        predicted_food, confidence = predict_food(img)

        st.success(f"Predicted Food: {predicted_food}")
        st.info(f"Confidence: {confidence:.2f}%")

        # -------------------------
        # Nutrition lookup
        # -------------------------
        nutrition_data = nutrition_df[
            nutrition_df["Food"]
            .str.lower()
            .str.strip()
            .str.replace("-", "", regex=False)
            == predicted_food.lower().strip().replace("-", "")
        ]

        if nutrition_data.empty:
            st.error("Nutrition data not available.")
        else:
            nutrition = nutrition_data.iloc[0]

            st.subheader("🥗 Nutrition")
            st.write("Calories :", nutrition["Calories"])
            st.write("Protein :", nutrition["Protein"])
            st.write("Carbohydrates :", nutrition["Carbohydrates"])
            st.write("Fat :", nutrition["Fat"])

            if "Ingredients" in nutrition_df.columns:
                st.subheader("🧂 Ingredients")
                st.write(nutrition["Ingredients"])

        # -------------------------
        # Diet Recommendation
        # -------------------------
        recommendation = "No recommendation available."
        food_key = predicted_food.strip().lower()

        for key in diet_data:
            if key.strip().lower() == food_key:
                if diet_goal in diet_data[key]:
                    recommendation = diet_data[key][diet_goal]
                break

        st.subheader("🥗 Diet Recommendation")
        st.write(recommendation)

        # -------------------------
        # BMI
        # -------------------------
        if height > 0 and weight > 0:
            bmi, status = calculate_bmi(height, weight)

            st.subheader("💪 BMI Calculator")
            st.write("BMI :", bmi)
            st.write("Status :", status)