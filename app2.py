from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import joblib  # For loading scaler
import os
from tensorflow.keras.preprocessing import image
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import openai
import faiss

# -------------------- AI Proxy Configuration -------------------- #
# Use the AI proxy token (e.g., from IITM login) instead of a standard OpenAI key.
openai.api_key = os.getenv("AIPROXY_TOKEN")
openai.api_base = "https://aiproxy.sanand.workers.dev/openai"

# -------------------- Flask App Setup -------------------- #
app = Flask(__name__)

# Ensure necessary directories exist
os.makedirs("static/uploads", exist_ok=True)

# -------------------- Load Models and Scaler -------------------- #
xray_model = tf.keras.models.load_model("models/xray_model.h5")
clinical_model = tf.keras.models.load_model("models/multiclass_oa_model.h5")
meta_classifier = joblib.load("models/meta_classifier.pkl")  # Ensemble model
scaler = joblib.load("models/scaler.pkl")  # Ensure scaler.pkl exists

# -------------------- Utility Functions -------------------- #
def get_dynamic_weights(pred1, pred2):
    confidence1 = np.max(pred1)
    confidence2 = np.max(pred2)
    total_confidence = confidence1 + confidence2
    w1 = confidence1 / total_confidence
    w2 = confidence2 / total_confidence
    return w1, w2

# -------------------- RAG Retrieval Functions -------------------- #
def load_faiss_index():
    """
    Load the FAISS index and the corresponding persistent documents.
    Expects 'persistent_index.index' and 'persistent_docs.txt' in the project root.
    """
    index = faiss.read_index("persistent_index.index")
    with open("persistent_docs.txt", "r") as f:
        docs = [line.strip() for line in f]
    return index, docs

# Load the FAISS index and persistent documents at startup
faiss_index, persistent_docs = load_faiss_index()

def get_embedding(text):
    """
    Get an embedding for the given text using the OpenAI embedding model via the AI proxy.
    """
    response = openai.Embedding.create(
        model="text-embedding-3-small",  # Use the proxy-supported embedding model
        input=[text]
    )
    embedding = response["data"][0]["embedding"]
    return np.array(embedding, dtype=np.float32)

def retrieve_relevant_context(query, k=1):
    """
    Retrieve the most relevant context for the query from the persistent documents using FAISS.
    """
    embedding = get_embedding(query)
    embedding = np.expand_dims(embedding, axis=0)
    distances, indices = faiss_index.search(embedding, k)
    retrieved_docs = [persistent_docs[i] for i in indices[0]]
    return " ".join(retrieved_docs)

# -------------------- Flask Endpoints -------------------- #
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/xray")
def xray_page():
    return render_template("xray.html")

@app.route("/clinical")
def clinical_page():
    return render_template("clinical.html")

@app.route("/fusion")
def fusion_page():
    return render_template("fusion.html")

@app.route("/late_fusion")
def late_fusion_page():
    return render_template("late_fusion.html")

@app.route("/predict_xray", methods=["POST"])
def predict_xray():
    if "xray" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["xray"]
    filename = os.path.join("static/uploads", file.filename)
    file.save(filename)

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = xray_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=-1)
    return jsonify({"prediction": int(predicted_class[0])})

@app.route("/predict_clinical", methods=["POST"])
def predict_clinical():
    print("\n📥 Received Form Data:", request.form)
    try:
        # Normalize input keys to uppercase for consistency
        form_data = {key.upper(): value for key, value in request.form.items()}
        print("\n📥 Normalized Form Data:", form_data)

        # Define the required features for clinical prediction
        required_features = [
            "AGE", "HEIGHT", "WEIGHT", "BMI", "FREQUENT_PAIN", "SURGERY",
            "SXKOA", "SWELLING", "BENDING_FULLY", "SYMPTOMATIC", "CREPITUS",
            "KOOS_PAIN_SCORE", "OSTEOPHYTES_Y", "JSN_Y", "OSFL"
        ]

        clinical_data = []
        for feature in required_features:
            if feature not in form_data:
                print(f"❌ ERROR: Missing required feature: {feature}")
                return jsonify({"error": f"Missing required feature: {feature}"}), 400
            try:
                value = float(form_data[feature])
            except ValueError:
                print(f"❌ ERROR: Invalid value for {feature}: {form_data[feature]}")
                return jsonify({"error": f"Invalid value for {feature}. Must be a number."}), 400
            clinical_data.append(value)

        clinical_data_np = np.array([clinical_data])
        print(f"🔍 Clinical Data Shape BEFORE Scaling: {clinical_data_np.shape}")
        clinical_data_np = scaler.transform(clinical_data_np)
        print(f"🔍 Clinical Data Shape AFTER Scaling: {clinical_data_np.shape}")

        prediction = clinical_model.predict(clinical_data_np)
        predicted_class = np.argmax(prediction, axis=-1)[0]
        print(f"✅ Prediction: {predicted_class}")
        return jsonify({"prediction": int(predicted_class)})

    except Exception as e:
        print(f"❌ SERVER ERROR: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/predict_fusion", methods=["POST"])
def predict_fusion():
    try:
        # Process X-ray image
        if "xray" not in request.files:
            return jsonify({"error": "No X-ray image uploaded"}), 400
        file = request.files["xray"]
        filename = os.path.join("static/uploads", file.filename)
        file.save(filename)

        img = image.load_img(filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        pred_xray = xray_model.predict(img_array)
        pred_xray_class = np.argmax(pred_xray, axis=-1)[0]
        pred_xray_probs = pred_xray[0]

        # Process clinical data
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

        w_xray, w_clinical = get_dynamic_weights(pred_xray_probs, pred_clinical_probs)
        fused_probs = (w_xray * pred_xray_probs) + (w_clinical * pred_clinical_probs)
        fused_class = np.argmax(fused_probs)

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

# -------------------- LLM Endpoints -------------------- #
# Basic LLM endpoint (without RAG)
@app.route("/llm", methods=["POST"])
def llm():
    """
    Endpoint to interact with the LLM directly (without RAG).
    Expects a JSON payload with a "prompt" key.
    """
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        llm_output = response.choices[0].message["content"]
        return jsonify({"response": llm_output})
    except Exception as e:
        print(f"❌ LLM Integration Error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# Virtual Bot endpoint using RAG
@app.route("/virtual_bot", methods=["POST"])
def virtual_bot():
    """
    Endpoint to interact with the LLM augmented by Retrieval-Augmented Generation (RAG).
    It retrieves relevant context from a vector store and uses it to enhance the LLM prompt.
    Expects a JSON payload with a "prompt" key.
    """
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Retrieve relevant context using FAISS
        context = retrieve_relevant_context(prompt, k=1)
        augmented_prompt = f"Context: {context}\n\nUser Prompt: {prompt}"

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": augmented_prompt}],
        )
        llm_output = response.choices[0].message["content"]
        return jsonify({"response": llm_output})
    except Exception as e:
        print(f"❌ Virtual Bot Error: {str(e)}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# -------------------- Main -------------------- #
if __name__ == "__main__":
    app.run(debug=True, port=8080)