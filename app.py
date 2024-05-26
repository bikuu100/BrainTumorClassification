from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import sys
sys.path.append('src')
import joblib
from predict_pipeline import preprocess_image, predict

app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Define the path to the trained model
MODEL_PATH = "C:/projects/BrainTumorPrediction/trained_model.pkl"

def load_trained_model(model_path):
    """Load the trained model."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        flash(f"Error loading the trained model: {e}")
        return None

# Load the trained model
model = load_trained_model(MODEL_PATH)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("Received POST request")
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"Saved file: {filepath}")
            # Make prediction
            prediction = predict(filepath, model)
            if prediction is not None:
                flash('Prediction completed successfully', 'success')
            else:
                flash('Error occurred during prediction', 'error')
            return render_template('predict_result.html', filename=filename, prediction=prediction)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        print("Received POST request for prediction")
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"Saved file: {filepath}")
            # Make prediction
            prediction = predict(filepath, model)
            if prediction is not None:
                flash('Prediction completed successfully', 'success')
            else:
                flash('Error occurred during prediction', 'error')
            return render_template('predict_result.html', filename=filename, prediction=prediction)
    return redirect(url_for('upload_file'))

@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
