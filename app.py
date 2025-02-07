from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import joblib  # For loading scaler
import os
from tensorflow.keras.preprocessing import image
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

# Ensure model directories exist
os.makedirs("static/uploads", exist_ok=True)

# Load models
xray_model = tf.keras.models.load_model("models/xray_model.h5")
clinical_model = tf.keras.models.load_model("models/multiclass_oa_model.h5")
meta_classifier = joblib.load("models/meta_classifier.pkl")  # Load the ensemble model

# Dynamic weight selection function
def get_dynamic_weights(pred1, pred2):
    confidence1 = np.max(pred1)
    confidence2 = np.max(pred2)
    total_confidence = confidence1 + confidence2
    w1 = confidence1 / total_confidence
    w2 = confidence2 / total_confidence
    return w1, w2

# Load the scaler
scaler = joblib.load("models/scaler.pkl")  # Ensure you have a scaler.pkl file

# Define the **15 required features**
CLINICAL_FEATURES = [
    "AGE", "HEIGHT", "WEIGHT", "BMI", "SXKOA", "SWELLING", 
    "BENDING FULLY", "SYMPTOMATIC", "CREPITUS", "KOOS PAIN SCORE",
    "osteophytes_y", "jsn_y", "osfl", "scfl", "cyfl"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/xray")
def xray_page():
    return render_template("xray.html")

@app.route("/clinical")
def clinical_page():
    return render_template("clinical.html")

@app.route("/predict_xray", methods=["POST"])
def predict_xray():
    # Check if the uploaded file is provided under the key "xray"
    if "xray" not in request.files:
        return jsonify({"error": "No X-ray file uploaded"}), 400

    # Retrieve the X-ray file from the request
    xray_file = request.files["xray"]
    # Build the full path where the X-ray image will be stored
    filename = os.path.join("static/uploads", xray_file.filename)
    xray_file.save(filename)

    # Load and preprocess the X-ray image
    # Here we use the Keras image module to load the file and resize it to (224, 224)
    xray_image = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
    xray_array = tf.keras.preprocessing.image.img_to_array(xray_image)
    xray_array = np.expand_dims(xray_array, axis=0)

    # Run the X-ray model prediction on the preprocessed X-ray image
    prediction = xray_model.predict(xray_array)
    predicted_class = np.argmax(prediction, axis=-1)

    # Return the predicted class as a JSON response
    return jsonify({"prediction": int(predicted_class[0])})
@app.route("/predict_clinical", methods=["POST"])
def predict_clinical():
    print("\n📥 Received Clinical Form Data:", request.form)
    try:
        # Normalize input keys to uppercase for consistency
        clinical_form_data = {key.upper(): value for key, value in request.form.items()}
        print("\n📥 Normalized Clinical Data:", clinical_form_data)

        # Define the required clinical features for prediction
        required_clinical_features = [
            "AGE", "HEIGHT", "WEIGHT", "BMI", "FREQUENT_PAIN", "SURGERY",
            "SXKOA", "SWELLING", "BENDING_FULLY", "SYMPTOMATIC", "CREPITUS",
            "KOOS_PAIN_SCORE", "OSTEOPHYTES_Y", "JSN_Y", "OSFL"
        ]

        clinical_inputs = []
        for feature in required_clinical_features:
            if feature not in clinical_form_data:
                print(f"❌ ERROR: Missing required clinical feature: {feature}")
                return jsonify({"error": f"Missing required clinical feature: {feature}"}), 400
            try:
                value = float(clinical_form_data[feature])
            except ValueError:
                print(f"❌ ERROR: Invalid value for {feature}: {clinical_form_data[feature]}")
                return jsonify({"error": f"Invalid value for {feature}. Must be a number."}), 400
            clinical_inputs.append(value)

        # Convert clinical input features into a NumPy array
        clinical_inputs_np = np.array([clinical_inputs])
        print(f"🔍 Clinical Data Shape BEFORE Scaling: {clinical_inputs_np.shape}")

        # Standardize the clinical data using the pre-loaded scaler
        clinical_inputs_np = scaler.transform(clinical_inputs_np)
        print(f"🔍 Clinical Data Shape AFTER Scaling: {clinical_inputs_np.shape}")

        # Use the clinical model to make a prediction
        clinical_prediction = clinical_model.predict(clinical_inputs_np)
        predicted_clinical_class = np.argmax(clinical_prediction, axis=-1)[0]
        print(f"✅ Clinical Prediction: {predicted_clinical_class}")

        return jsonify({"prediction": int(predicted_clinical_class)})

    except Exception as e:
        print(f"❌ Clinical Prediction Server Error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/fusion")
def fusion_page():
    return render_template("fusion.html")

# Corrected: Only one /predict_fusion endpoint is defined below.
@app.route("/predict_fusion", methods=["POST"])
def predict_fusion():
    try:
        # Get X-ray image
        if "xray" not in request.files:
            return jsonify({"error": "No X-ray image uploaded"}), 400
        file = request.files["xray"]
        filename = os.path.join("static/uploads", file.filename)
        file.save(filename)

        # Load and preprocess the image
        img = image.load_img(filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Get X-ray model prediction
        pred_xray = xray_model.predict(img_array)
        pred_xray_class = np.argmax(pred_xray, axis=-1)[0]
        pred_xray_probs = pred_xray[0]

        # Get Clinical data
        required_features = [
            "AGE", "HEIGHT", "WEIGHT", "BMI", "FREQUENT_PAIN", "SURGERY", "RISK",
            "SXKOA", "SWELLING", "BENDING_FULLY", "SYMPTOMATIC", "CREPITUS",
            "KOOS_PAIN_SCORE", "OSTEOPHYTES_Y", "JSN_Y"
        ]
        clinical_data = []
        for feature in required_features:
            value = request.form.get(feature)
            if value is None:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400
            clinical_data.append(float(value))

        clinical_data_np = np.array([clinical_data])
        pred_clinical = clinical_model.predict(clinical_data_np)
        pred_clinical_class = np.argmax(pred_clinical, axis=-1)[0]
        pred_clinical_probs = pred_clinical[0]

        # Get dynamic weights and fuse predictions
        w_xray, w_clinical = get_dynamic_weights(pred_xray_probs, pred_clinical_probs)
        fused_probs = (w_xray * pred_xray_probs) + (w_clinical * pred_clinical_probs)
        fused_class = np.argmax(fused_probs)

        # Prepare meta-classifier input and get final ensemble prediction
        meta_input = np.hstack([pred_xray_probs, pred_clinical_probs, [w_xray, w_clinical]])
        final_pred = meta_classifier.predict([meta_input])[0]

        return jsonify({
            "xray_prediction": int(pred_xray_class),
            "clinical_prediction": int(pred_clinical_class),
            "fused_prediction": int(fused_class),
            "final_ensemble_prediction": int(final_pred)
        })
    
    except Exception as e:
        print(f"❌ SERVER ERROR: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/late_fusion")
def late_fusion_page():
    return render_template("late_fusion.html")

if __name__ == "__main__":
    app.run(debug=True, port=8080)