# src/app/routes.py

from flask import Blueprint, render_template, request, jsonify
from src.inference.predict import predict_hardness, predict_oxidation

app_bp = Blueprint("app_bp", __name__)


@app_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def _convert(value):
    """Convert form values to numeric when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value  # Keep as string for Material


@app_bp.route("/predict", methods=["POST"])
def predict():
    # Read form inputs
    material = request.form.get("Material")
    current = _convert(request.form.get("Current"))
    heat_input = _convert(request.form.get("Heat_Input"))
    soaking_time = _convert(request.form.get("Soaking_Time"))
    carbon = _convert(request.form.get("Carbon"))
    manganese = _convert(request.form.get("Manganese"))

    # Build payloads
    hardness_payload = {
        "Material": material,
        "Current": current,
        "Heat_Input": heat_input,
        "Carbon": carbon,
        "Manganese": manganese,
    }

    oxidation_payload = {
        "Material": material,
        "Current": current,
        "Heat_Input": heat_input,
        "Soaking_Time": soaking_time,
        "Carbon": carbon,
        "Manganese": manganese,
    }

    # Get predictions
    hardness_result = predict_hardness(hardness_payload)
    oxidation_result = predict_oxidation(oxidation_payload)

    return render_template(
        "index.html",
        hardness=hardness_result.get("prediction"),
        oxidation=oxidation_result.get("prediction"),
        hardness_error=hardness_result.get("error"),
        oxidation_error=oxidation_result.get("error"),
        selected_material=material,
        form_data=request.form,
    )


# JSON API for async UI
@app_bp.route("/api/v1/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    # Convert numeric fields here also
    for key in ["Current", "Heat_Input", "Soaking_Time", "Carbon", "Manganese"]:
        if key in payload:
            try:
                payload[key] = float(payload[key])
            except Exception:
                pass  # Ignore; inference layer will validate

    hardness_result = predict_hardness(payload)
    oxidation_result = predict_oxidation(payload)

    return jsonify({
        "hardness": hardness_result.get("prediction"),
        "oxidation": oxidation_result.get("prediction"),
        "hardness_error": hardness_result.get("error"),
        "oxidation_error": oxidation_result.get("error"),
    }), 200
