from flask import Flask, request, jsonify,render_template
import pickle

app = Flask(__name__)
model= pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    input_mail = request.form['message']
    input_data_features = vectorizer.transform([input_mail])
    prediction = model.predict(input_data_features)
    
    if prediction[0] == 1:
        result = "✅Ham mail"
    else:
        result = "⚠️Spam mail"
    
    return render_template('index.html', prediction_text=result)
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)