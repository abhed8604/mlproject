# Student Exam Performance Predictor

An end-to-end machine learning project that predicts a student's math score based on demographics, preparation, and reading/writing scores.

Built with Python, scikit-learn, CatBoost/XGBoost, and Flask.

## How to Run

1. **Set up virtual environment & dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train the model**
   Runs the pipeline (ingestion, transformation, model evaluation) and saves the best model to the `artifacts/` folder:
   ```bash
   python src/components/data_ingestion.py
   ```

3. **Run the web app**
   ```bash
   python app.py
   ```
   Open `http://localhost:5000` in your browser to test predictions.