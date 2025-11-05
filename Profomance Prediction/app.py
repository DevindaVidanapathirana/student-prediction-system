from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path


# Initialize FastAPI app.
app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting student performance scores and grades",
    version="1.0.0"
)

# Add CORS middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and scaler
try:
    lr_model = joblib.load("performance_linear_regression.pkl")
    perf_scaler = joblib.load("performance_scaler.pkl")
except FileNotFoundError:
    print("Warning: Model files not found. Please ensure the notebook has been run first.")

# Define request/response models
class StudentData(BaseModel):
    quiz_avg: float
    assignment_avg: float
    exam_score: float
    login_frequency: float
    time_spent_hours: float
    course_progress: float
    historical_gpa: float
    eti_score: float

class PredictionResponse(BaseModel):
    predicted_score: float
    predicted_grade: str
    input_data: StudentData

# Helper function to convert score to grade
def score_to_grade(score):
    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 55:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    print(f"Warning: Could not mount static directory: {e}")

# API Routes
@app.get("/")
async def read_root():
    return FileResponse("static/index.html", media_type="text/html")

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_performance(student: StudentData):
    """
    Predict student performance score and grade based on their metrics.
    
    - **quiz_avg**: Average quiz score (0-100)
    - **assignment_avg**: Average assignment score (0-100)
    - **exam_score**: Final exam score (0-100)
    - **login_frequency**: Number of logins per week (0-10)
    - **time_spent_hours**: Total hours spent on course per week (1-40)
    - **course_progress**: Course progress percentage (0-100)
    - **historical_gpa**: Previous GPA (0-4)
    - **eti_score**: Engagement and Time Investment score (0-100)
    """
    try:
        # Prepare data
        student_df = pd.DataFrame([{
            "quiz_avg": student.quiz_avg,
            "assignment_avg": student.assignment_avg,
            "exam_score": student.exam_score,
            "login_frequency": student.login_frequency,
            "time_spent_hours": student.time_spent_hours,
            "course_progress": student.course_progress,
            "historical_gpa": student.historical_gpa,
            "eti_score": student.eti_score
        }])
        
        # Scale features
        X_scaled = perf_scaler.transform(student_df)
        
        # Make prediction
        predicted_score = float(lr_model.predict(X_scaled)[0])
        
        # Clamp score to valid range
        predicted_score = max(0, min(100, predicted_score))
        
        # Get grade
        predicted_grade = score_to_grade(predicted_score)
        
        return PredictionResponse(
            predicted_score=round(predicted_score, 2),
            predicted_grade=predicted_grade,
            input_data=student
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Student Performance Prediction API"}

@app.get("/api/sample")
async def get_sample_data():
    """Get sample student data for testing"""
    return {
        "sample_student": {
            "quiz_avg": 72,
            "assignment_avg": 78,
            "exam_score": 70,
            "login_frequency": 5,
            "time_spent_hours": 18,
            "course_progress": 80,
            "historical_gpa": 3.0,
            "eti_score": 74
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
