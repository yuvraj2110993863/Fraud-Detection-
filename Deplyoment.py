from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import pandas as pd

# Load the model
with open('fraud_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict')
def predict():
    
    # data = request.get_json(force=True)
    # df = pd.DataFrame([data])
    
    # scaler=StandardScaler()
    # le=LabelEncoder()
    # # Preprocess the input data (scale and encode)
    # df['amt'] = scaler.transform(df[['amt']])
    # df['category'] = le.transform(df['category'])
    
    # # Make prediction
    # prediction = model.predict(df)
    return jsonify({'is_fraud': 12})
    

if __name__ == '__main__':
    app.run(debug=True)