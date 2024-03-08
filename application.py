# Import necessary modules and functions
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

# Import preprocessing and prediction pipeline
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create Flask application instance
application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html') 

# Route for prediction data
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    # Check if the request method is POST
    if request.method == 'POST':
        # Collect form data
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        # Convert data to DataFrame
        pred_df = data.get_data_as_data_frame()
        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()
        # Make prediction
        results = predict_pipeline.predict(pred_df)
        # Render index.html with prediction results
        return render_template('index.html', results=results[0])
    else:
        # Render index.html without any data
        return render_template('index.html')

# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0")
