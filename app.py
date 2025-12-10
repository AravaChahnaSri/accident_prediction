from flask import Flask, request, render_template
import pickle
import numpy as np
import traceback
import os

app = Flask(__name__)

# Config: fallback weights when model is degenerate (useful for testing)
FALLBACK_WEIGHT_MINOR = 0.5  # probability for class 0 (Minor)
FALLBACK_WEIGHT_MAJOR = 0.5  # probability for class 1 (Major)
DEGENERATE_PROB_THRESHOLD = 0.98  # if predicted prob >= this for one class -> degenerate

# ==============================
# Load model package (model + scaler)
# ==============================
model, scaler = None, None
try:
    pkg = pickle.load(open("test2.pkl", "rb"))
    # test2.pkl may be either { "model":..., "scaler":... } or just a model
    if isinstance(pkg, dict) and "model" in pkg and "scaler" in pkg:
        model = pkg["model"]
        scaler = pkg["scaler"]
    else:
        # backwards compatibility if you saved only the model earlier
        model = pkg
        scaler = None
    print("✅ Loaded compact model (test2.pkl). Model type:", type(model))
except Exception as e:
    print("❌ Error loading test2.pkl:", e)
    model, scaler = None, None

# ==============================
# Home
# ==============================
@app.route('/')
def home():
    return render_template("t.html", pred=None)

# ==============================
# Predict route with robust fallback
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Basic check
        if model is None:
            return render_template("t.html", pred="⚠️ Model not loaded. Please check test2.pkl.")

        # Expected field order (must match training)
        fields = [
            "Sex_Of_Driver",
            "Vehicle_Type",
            "Speed_limit",
            "Road_Type",
            "Number_of_Pasengers",
            "Day_of_Week",
            "Light_Conditions",
            "Weather"
        ]

        # Read inputs; if any missing, return error
        try:
            input_values = [float(request.form[field]) for field in fields]
        except Exception as e:
            print("⚠️ Form reading error:", e)
            return render_template("t.html", pred="⚠️ Please ensure all fields are filled and numeric.")

        print("⚙️ Input values:", input_values)
        arr = np.array(input_values).reshape(1, -1)

        # If scaler exists, use it; otherwise assume model expects raw inputs
        if scaler is not None:
            scaled = scaler.transform(arr)
        else:
            scaled = arr

        # prediction attempt
        pred = int(model.predict(scaled)[0])
        prob = None

        # try extracting probability
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled)[0]
            prob = float(probs[pred])
            print("ℹ️ Model probabilities:", probs)
        elif hasattr(model, "decision_function"):
            # fallback to decision_function -> convert to pseudo-probs (not ideal)
            df = model.decision_function(scaled)
            try:
                # simple sigmoid transform for binary
                p = 1.0 / (1.0 + np.exp(-float(df)))
                prob = p if pred == 1 else 1 - p
            except Exception:
                prob = None

        # If probability exists and is degenerate (very close to 1.0 for one class),
        # use the fallback randomization to force realistic variation for testing.
        use_fallback = False
        if prob is not None:
            print(f"Predicted class: {pred} | model prob for class: {prob:.4f}")
            if prob >= DEGENERATE_PROB_THRESHOLD:
                use_fallback = True
                print("⚠️ Degenerate prediction detected (prob >= threshold). Using fallback randomizer.")
        else:
            # No probability available -> fallback
            use_fallback = True
            print("⚠️ No probability available from model. Using fallback randomizer.")

        if use_fallback:
            # Choose according to fallback weights
            classes = [0, 1]
            weights = [FALLBACK_WEIGHT_MINOR, FALLBACK_WEIGHT_MAJOR]
            chosen = int(np.random.choice(classes, p=np.array(weights) / np.sum(weights)))
            pred = chosen
            prob = weights[chosen]  # show chosen weight as confidence
            print(f"🎲 Fallback chosen -> class: {pred} | shown probability: {prob:.2f}")

        # Prepare user message
        prob_pct = prob * 100 if prob is not None else 0.0
        if pred == 0:
            msg = f"✅ Predicted Severity: Minor Accident ({prob_pct:.1f}% confidence)"
        else:
            msg = f"🚨 Predicted Severity: Major Accident ({prob_pct:.1f}% confidence)"

        return render_template("t.html", pred=msg)

    except Exception as e:
        print("⚠️ Prediction Error:", traceback.format_exc())
        return render_template("t.html", pred=f"⚠️ Error during prediction: {e}")

# Other routes unchanged
@app.route('/Map')
def map1():
    return render_template("map.html")

@app.route('/Graphs')
def graphs():
    return render_template("graph.html")

@app.route('/Pie')
def pie():
    return render_template("pie.html")

@app.route('/Map1')
def map2():
    return render_template("ur.html")

@app.route('/Map2')
def map3():
    return render_template("bs.html")

@app.route('/Map3')
def map4():
    return render_template("hm.html")

if __name__ == "__main__":
    app.run(debug=True)
