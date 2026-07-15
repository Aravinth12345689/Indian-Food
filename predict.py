import os
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("model/food_model.keras")
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)


image_folder = "test_images"

image_file = None

for file in os.listdir(image_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_file = os.path.join(image_folder, file)
        break
if image_file is None:
    print("No image found in test_images folder!")
    exit()


img = image.load_img(image_file, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)
predicted_food = class_names[predicted_index]
confidence = np.max(prediction) * 100


print("Predicted Food:", predicted_food)
print("Confidence:", round(confidence, 2), "%")
#print(train_dataset.class_names)

