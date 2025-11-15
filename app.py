import numpy as np
import pandas as pd
import pickle
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="AI Placement Predictor (Regression)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

model = None
scaler = None

FEATURE_ORDER = [
    "IQ",
    "CGPA",
    "Academic_Performance",
    "Internship_Experience",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed"
]


class InputData(BaseModel):
    College_ID: Optional[str] = None
    IQ: float
    CGPA: float
    Academic_Performance: float
    Internship_Experience: str
    Extra_Curricular_Score: float
    Communication_Skills: float
    Projects_Completed: float


def generate_suggestions(row: pd.Series):
    """Generate student improvement tips"""
    tips = []
    if row["CGPA"] < 7:
        tips.append("📚 Focus on improving CGPA (academic performance).")
    elif row["CGPA"] >= 9:
        tips.append("✅ Excellent CGPA — keep it up!")

    if row["Communication_Skills"] < 6:
        tips.append("💬 Work on communication and interview skills.")
    else:
        tips.append("💪 Strong communication skills!")

    if row["Projects_Completed"] < 3:
        tips.append("🧑‍💻 Complete more projects to strengthen portfolio.")
    else:
        tips.append("🚀 Good project experience!")

    if row["Internship_Experience"] == 0:
        tips.append("🧾 Try to gain internship experience.")
    else:
        tips.append("🏆 Internship experience is a strong advantage!")

    if row["Extra_Curricular_Score"] < 5:
        tips.append("🎭 Engage in extra-curricular activities to improve your overall profile.")

    return " | ".join(tips)


async def load_resources():
    """Loads model and scaler asynchronously when first needed"""
    global model, scaler
    if model is None or scaler is None:
        import tensorflow
        from tensorflow import keras
        from keras.models import load_model
        import tensorflow as tf
        model = load_model("placement_predict_model.keras")
        scaler = pickle.load(open("scaler.pkl", "rb"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve HTML frontend"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(data: InputData):
    await load_resources()  

    input_dict = data.dict()
    internship_mapped = {"Yes": 1, "No": 0, "yes": 1, "no": 0}.get(
        input_dict.get("Internship_Experience", ""), 0
    )

    row = pd.DataFrame(
        [[
            input_dict["IQ"],
            input_dict["CGPA"],
            input_dict["Academic_Performance"],
            internship_mapped,
            input_dict["Extra_Curricular_Score"],
            input_dict["Communication_Skills"],
            input_dict["Projects_Completed"]
        ]],
        columns=FEATURE_ORDER
    )

    X_scaled = scaler.transform(row)
    prob = float(model.predict(X_scaled, verbose=0)[0][0])

    if prob < 0.2:
        placement_status = "Very Low Chance 😔"
        color = "red"
    elif prob < 0.4:
        placement_status = "Low Chance 🙁"
        color = "orange"
    elif prob < 0.6:
        placement_status = "Mid Chance 😬"
        color = "yellow"
    elif prob < 0.8:
        placement_status = "Good Chance 🙂"
        color = "blue"
    else:
        placement_status = "Very Good Chance 😎"
        color = "green"

    placement_percent = round(prob * 100, 2)
    suggestions = generate_suggestions(row.iloc[0])

    return JSONResponse({
        "College_ID": input_dict.get("College_ID", "N/A"),
        "placement_status": placement_status,
        "placement_score": round(prob, 3),
        "placement_probability_percent": placement_percent,
        "color": color,
        "suggestions": suggestions
    })


@app.get("/health")
async def health():
    return {"status": "ok"}
