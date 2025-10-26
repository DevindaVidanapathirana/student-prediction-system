# Student Performance Prediction API Setup Guide

## Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- The trained model files from your Jupyter notebook

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Model Files
Make sure your notebook has been executed completely to generate:
- `performance_linear_regression.pkl`
- `performance_scaler.pkl`

Both files should be in the same directory as `app.py`

### 4. Run the Server
```bash
python app.py
```

You should see output like:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 5. Access the Web UI
Open your browser and navigate to:
```
http://localhost:8000
```

### 6. Test the API
- Use the web UI form to enter student data and see predictions
- Click "Load Sample" button to populate test data
- View API documentation at: http://localhost:8000/docs

## Project Structure
```
Component 2/
├── app.py                              # FastAPI application
├── requirements.txt                    # Python dependencies
├── API.md                             # API documentation
├── README.md                          # This file
├── model_test.ipynb                   # Jupyter notebook (training)
├── performance_linear_regression.pkl  # Trained model (generated)
├── performance_scaler.pkl             # Feature scaler (generated)
└── static/
    └── index.html                     # Web UI interface
```

## API Endpoints

### Main Endpoints
- `POST /api/predict` - Make a prediction
- `GET /api/health` - Health check
- `GET /api/sample` - Get sample data
- `GET /` - Web UI interface

For detailed API documentation, see [API.md](API.md)

## Troubleshooting

### Model Files Not Found
**Error:** `FileNotFoundError: Model files not found`

**Solution:** 
1. Run your Jupyter notebook completely (all cells)
2. Ensure model files are saved in the same directory as `app.py`
3. Verify file names match exactly

### Port Already in Use
**Error:** `OSError: [Errno 48] Address already in use`

**Solution:**
```bash
# Change the port in app.py or use:
python app.py --port 8001
```

### Dependencies Installation Fails
**Solution:**
```bash
# Create a fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Example Usage

### Using the Web UI
1. Navigate to `http://localhost:8000`
2. Enter student metrics
3. Click "Predict Performance"
4. View the predicted score and grade

### Using cURL
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "quiz_avg": 85,
    "assignment_avg": 88,
    "exam_score": 92,
    "login_frequency": 8,
    "time_spent_hours": 25,
    "course_progress": 95,
    "historical_gpa": 3.8,
    "eti_score": 88
  }'
```

### Using Python
```python
import requests

response = requests.post('http://localhost:8000/api/predict', json={
    "quiz_avg": 85,
    "assignment_avg": 88,
    "exam_score": 92,
    "login_frequency": 8,
    "time_spent_hours": 25,
    "course_progress": 95,
    "historical_gpa": 3.8,
    "eti_score": 88
})

print(response.json())
```

## Performance Notes
- Predictions are generated in milliseconds
- The API can handle multiple concurrent requests
- Feature scaling is applied automatically
- All scores are clamped to valid ranges (0-100)

## Next Steps
1. Test the API with various student data
2. Integrate with your LMS system
3. Monitor prediction accuracy
4. Collect feedback and refine the model
