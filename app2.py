from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import joblib
import os
from tensorflow.keras.preprocessing import image
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import openai
import faiss

# -------------------- AI Proxy Configuration -------------------- #
openai.api_key = os.getenv("AIPROXY_TOKEN")
openai.api_base = "https://aiproxy.sanand.workers.dev/openai"

# -------------------- Flask App Setup -------------------- #
app = Flask(__name__)
os.makedirs("static/uploads", exist_ok=True)

# -------------------- Load Models and Scaler -------------------- #
xray_model = tf.keras.models.load_model("models/xray_model.h5")
clinical_model = tf.keras.models.load_model("models/multiclass_oa_model.h5")
meta_classifier = joblib.load("models/meta_classifier.pkl")
scaler = joblib.load("models/scaler.pkl")

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
    index = faiss.read_index("persistent_index.index")
    with open("persistent_docs.txt", "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    return index, docs

faiss_index, persistent_docs = load_faiss_index()

def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=[text]
    )
    embedding = response["data"][0]["embedding"]
    return np.array(embedding, dtype=np.float32)

def retrieve_relevant_context(query, k=1):
    embedding = get_embedding(query)
    embedding = np.expand_dims(embedding, axis=0)
    distances, indices = faiss_index.search(embedding, k)
    retrieved_docs = [persistent_docs[i] for i in indices[0]]
    return " ".join(retrieved_docs)

def load_default_prompt():
    with open("data/oa_kl_document.txt", "r", encoding="utf-8") as f:
        return f.read()

default_prompt = load_default_prompt()

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
        return jsonify({"error": "No X-ray image uploaded"}), 400
    xray_file = request.files["xray"]
    xray_filename = os.path.join("static/uploads", xray_file.filename)
    xray_file.save(xray_filename)
    xray_img = image.load_img(xray_filename, target_size=(224, 224))
    xray_array = image.img_to_array(xray_img)
    xray_array = np.expand_dims(xray_array, axis=0)
    prediction = xray_model.predict(xray_array)
    predicted_class = int(np.argmax(prediction, axis=-1)[0])
    return jsonify({"prediction": predicted_class})

@app.route("/predict_clinical", methods=["POST"])
def predict_clinical():
    try:
        form_data = {key.upper(): value for key, value in request.form.items()}
        required_features = [
            "AGE", "HEIGHT", "WEIGHT", "BMI", "FREQUENT_PAIN", "SURGERY",
            "SXKOA", "SWELLING", "BENDING_FULLY", "SYMPTOMATIC", "CREPITUS",
            "KOOS_PAIN_SCORE", "OSTEOPHYTES_Y", "JSN_Y", "OSFL"
        ]
        clinical_data = []
        for feature in required_features:
            if feature not in form_data:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400
            try:
                clinical_data.append(float(form_data[feature]))
            except ValueError:
                return jsonify({"error": f"Invalid value for {feature}. Must be a number."}), 400
        clinical_data_np = np.array([clinical_data])
        clinical_data_np = scaler.transform(clinical_data_np)
        prediction = clinical_model.predict(clinical_data_np)
        predicted_class = int(np.argmax(prediction, axis=-1)[0])
        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/predict_fusion", methods=["POST"])
def predict_fusion():
    try:
        # Process X-ray image
        if "xray" not in request.files:
            return jsonify({"error": "No X-ray image uploaded"}), 400
        xray_file = request.files["xray"]
        xray_filename = os.path.join("static/uploads", xray_file.filename)
        xray_file.save(xray_filename)
        xray_img = image.load_img(xray_filename, target_size=(224, 224))
        xray_array = image.img_to_array(xray_img)
        xray_array = np.expand_dims(xray_array, axis=0)
        pred_xray = xray_model.predict(xray_array)
        pred_xray_class = int(np.argmax(pred_xray, axis=-1)[0])
        pred_xray_probs = pred_xray[0]

        # Process clinical data
        required_clinical_features = [
            "AGE", "HEIGHT", "WEIGHT", "BMI", "FREQUENT_PAIN", "SURGERY", "RISK",
            "SXKOA", "SWELLING", "BENDING_FULLY", "SYMPTOMATIC", "CREPITUS",
            "KOOS_PAIN_SCORE", "OSTEOPHYTES_Y", "JSN_Y"
        ]
        clinical_inputs = {}
        clinical_data_list = []
        for feature in required_clinical_features:
            value = request.form.get(feature)
            if value is None:
                return jsonify({"error": f"Missing required clinical feature: {feature}"}), 400
            try:
                clinical_data_list.append(float(value))
            except ValueError:
                return jsonify({"error": f"Invalid value for {feature}. Must be a number."}), 400
            clinical_inputs[feature] = value
        clinical_data_np = np.array([clinical_data_list])
        pred_clinical = clinical_model.predict(clinical_data_np)
        pred_clinical_class = int(np.argmax(pred_clinical, axis=-1)[0])
        pred_clinical_probs = pred_clinical[0]

        # Compute dynamic weights and fusion
        w_xray, w_clinical = get_dynamic_weights(pred_xray_probs, pred_clinical_probs)
        fused_probs = (w_xray * pred_xray_probs) + (w_clinical * pred_clinical_probs)
        fused_class = int(np.argmax(fused_probs))
        meta_input = np.hstack([pred_xray_probs, pred_clinical_probs, [w_xray, w_clinical]])
        final_pred = int(meta_classifier.predict([meta_input])[0])

        # Build a text summary of clinical parameters for the prompt
        clinical_params_text = "\n".join(
            [f"{key}: {clinical_inputs[key]}" for key in required_clinical_features]
        )
        # Build the comprehensive prompt for the LLM.
        prompt_text = f"""
Please provide a detailed analysis and commentary based on the data below.
Your response must be formatted into two clearly labeled sections separated by the delimiters.

[AI Bot Commentary]
Please provide a thorough clinical analysis of the following clinical parameters:
{clinical_params_text}

[KL Summary]
Based on the X-ray analysis, the model predicted class: {pred_xray_class}. The final late fusion KL score is: {final_pred}.
Please provide a detailed radiographic analysis, explain how these clinical parameters correlate with the radiographic findings, and offer recommendations for further investigation or treatment.

Ensure your answer is well-structured, with the first section dedicated to clinical analysis and the second section to radiographic analysis and summary.
        """
        llm_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_text}],
        )
        ai_commentary = llm_response.choices[0].message["content"]

        # Split the commentary based on the delimiters
        clinical_commentary = ""
        kl_summary = ""
        if "[KL Summary]" in ai_commentary:
            parts = ai_commentary.split("[KL Summary]")
            clinical_commentary = parts[0].replace("[AI Bot Commentary]", "").strip()
            kl_summary = parts[1].strip()
        else:
            clinical_commentary = ai_commentary
            kl_summary = ai_commentary

        return jsonify({
            "xray_prediction": pred_xray_class,
            "clinical_prediction": pred_clinical_class,
            "fused_prediction": fused_class,
            "final_ensemble_prediction": final_pred,
            "ai_bot_commentary": clinical_commentary,
            "kl_summary": kl_summary
        })
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/llm", methods=["POST"])
def llm():
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
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/virtual_bot", methods=["POST"])
def virtual_bot():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        context = retrieve_relevant_context(prompt, k=1)
        if not context.strip():
            context = default_prompt
        augmented_prompt = f"Context: {context}\n\nUser Prompt: {prompt}"
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": augmented_prompt}],
        )
        llm_output = response.choices[0].message["content"]
        return jsonify({"response": llm_output})
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8080)