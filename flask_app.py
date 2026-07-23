from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import json
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
  
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
model = tf.keras.models.load_model("model/food_model.keras")
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)
nutrition_df = pd.read_csv("nutrition/nutrition.csv")
with open("diet/diet_recommendations.json", "r") as f:
    diet_data = json.load(f)

def predict_food(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    

    prediction = model.predict(img_array,verbose=0)
    print(prediction[0])
    print("Predicted Index:", np.argmax(prediction[0]))
    print("Predicted Food:", class_names[np.argmax(prediction[0])])

    predicted_index = np.argmax(prediction)
    print("Predicted Index:", predicted_index)
    print("Class Name:", class_names[predicted_index])

    predicted_food = class_names[predicted_index]
    confidence = float(np.max(prediction[0]) * 100)


    return predicted_food, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]

        if file:
            upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(upload_path)
            diet_goal = request.form.get("diet_goal", "")
            height = request.form.get("height", "")
            weight = request.form.get("weight", "")
            predicted_food, confidence = predict_food(upload_path)
            bmi = None
            bmi_status = ""

            if height and weight:
                height = float(height) / 100   # Convert cm to meters
                weight = float(weight)

                bmi = round(weight / (height * height), 2)

                if bmi < 18.5:
                    bmi_status = "Underweight"
                elif bmi < 25:
                    bmi_status = "Normal Weight"
                elif bmi < 30:
                    bmi_status = "Overweight"
                else:
                    bmi_status = "Obese"
            print("Predicted Food =", repr(predicted_food))
            print("Diet Goal =", repr(diet_goal))
            print("JSON Keys =", list(diet_data.keys()))

            print("Selected Diet Goal:", diet_goal)
        
            recommendation = "No recommendation available."

            food_key = predicted_food.strip().lower()

            for key in diet_data:
                if key.strip().lower() == food_key:
                   if diet_goal in diet_data[key]:
                      recommendation = diet_data[key][diet_goal]
                   break
            print("Predicted Food:", predicted_food)
            print("Foods in CSV:", nutrition_df["Food"].tolist())
            nutrition_data = nutrition_df[
                nutrition_df["Food"]
                .str.lower()
                .str.strip()
                .str.replace("-", "", regex=False)
             ==
            predicted_food.lower().strip().replace("-", "")
            ]
            if nutrition_data.empty:
                calories = protein = carbohydrates = fat = "Not Available"
                ingredients = "Not Available"
            else:
                 nutrition = nutrition_data.iloc[0]
                 calories = nutrition["Calories"]
                 protein = nutrition["Protein"]
                 carbohydrates = nutrition["Carbohydrates"]
                 fat = nutrition["Fat"]
                 ingredients = nutrition["Ingredients"]
            print("Predicted:", predicted_food)
            print("Confidence:", confidence)

            return render_template("index.html",
               image=file.filename,
               food=predicted_food,
               confidence=round(confidence, 2),
               calories=calories,
               protein=protein,
               carbohydrates=carbohydrates,
               fat=fat,
               ingredients=ingredients,
               recommendation=recommendation,
               bmi=bmi,
               bmi_status=bmi_status
               
            )

    return render_template("index.html")

@app.route("/bmi", methods=["GET", "POST"])
def bmi():
    if request.method == "POST":
        height = float(request.form["height"]) / 100
        weight = float(request.form["weight"])

        bmi = round(weight / (height * height), 2)

        if bmi < 18.5:
            status = "Underweight"
            recommendation = "Increase calorie and protein intake. Eat foods like Kadai Paneer, Rajma Chawal, Dal Curry, eggs, milk, and nuts."

        elif bmi < 25:
            status = "Normal Weight"
            recommendation = "Maintain a balanced diet. Continue eating healthy foods like Idly, Chapathi, Poha, Dal Curry, fruits, and vegetables."

        elif bmi < 30:
            status = "Overweight"
            recommendation = "Reduce fried and high-calorie foods. Choose Idly, Poha, Chapathi, salads, and exercise regularly."

        else:
            status = "Obese"
            recommendation = "Consult a healthcare professional. Focus on a low-calorie, high-fiber diet and regular physical activity."

        return render_template(
            "bmi.html",
            bmi=bmi,
            status=status,
            recommendation=recommendation
        )

    return render_template("bmi.html")

if __name__ == "__main__":
    app.run(debug=True)
