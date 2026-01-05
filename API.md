# Student Engagement API

FastAPI service to classify student personas and score engagement. Service lives in [main.py](main.py). A simple HTML tester is in [webui/index.html](webui/index.html).

## Quickstart
- Install deps (inside your venv if using one): `pip install fastapi uvicorn[standard] joblib pandas scikit-learn`
- Run the API: `uvicorn main:app --reload`
- Swagger UI: http://127.0.0.1:8000/docs
- Test page: open [webui/index.html](webui/index.html) in a browser while the server is running.

## Artifacts expected beside the API
- Classification: `student_engagement_model.pkl`, `scaler.pkl`, `label_encoder.pkl`
- Regression: `engagement_score_model.pkl`, `engagement_score_scaler.pkl`

## Request schema
All POST endpoints take the same JSON body:
```json
{
  "login_frequency": 6,
  "session_duration": 75,
  "forum_participation": 8,
  "assignment_access": 9,
  "time_gap_avg": 1.2,
  "inactivity_days": 1
}
```

## Endpoints
- **GET /health** — returns `{ ok, classifier_loaded, regressor_loaded }`.
- **POST /persona** — returns predicted `persona` and class `probabilities`.
- **POST /score** — returns `engagement_score` and suggested `action`.
- **POST /predict** — returns combined persona, probabilities, engagement score, and action (only the models that are loaded are used).

## Examples
Health check:
```sh
curl http://127.0.0.1:8000/health
```

Persona only:
```sh
curl -X POST http://127.0.0.1:8000/persona \
  -H "Content-Type: application/json" \
  -d '{"login_frequency":6,"session_duration":75,"forum_participation":8,"assignment_access":9,"time_gap_avg":1.2,"inactivity_days":1}'
```
Response:
```json
{
  "persona": "Highly Engaged",
  "probabilities": {
    "At-Risk": 0.02,
    "Highly Engaged": 0.95,
    "Moderately Engaged": 0.03
  }
}
```

Score only:
```sh
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"login_frequency":3,"session_duration":35,"forum_participation":2,"assignment_access":4,"time_gap_avg":3.5,"inactivity_days":6}'
```
Response:
```json
{
  "engagement_score": 58.41,
  "action": "Light nudges"
}
```

Combined prediction:
```sh
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"login_frequency":1,"session_duration":18,"forum_participation":0,"assignment_access":1,"time_gap_avg":7.5,"inactivity_days":20}'
```
Response:
```json
{
  "persona": "At-Risk",
  "probabilities": {
    "At-Risk": 0.83,
    "Highly Engaged": 0.04,
    "Moderately Engaged": 0.13
  },
  "engagement_score": 22.17,
  "action": "Immediate intervention"
}
```

## Web UI usage
1. Start the API (`uvicorn main:app --reload`).
2. Open [webui/index.html](webui/index.html) in a browser.
3. Enter feature values and click a button to call the desired endpoint. The JSON response appears under “Response”.
