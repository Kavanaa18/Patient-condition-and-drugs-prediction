# from flask import Flask, request ,jsonify,render_template
# import pickle
# import joblib
# import pandas as pd
# from utils import top_drugs_extract

# model = joblib.load("model.pkl")
# vectorizer = joblib.load("tfidf.pkl")
# result = joblib.load("results.pkl")
# df_filtered = pd.read_csv('df_filtered.csv') 

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     text = request.form["text"]
#     x=vectorizer.transform([text])
#     prediction = model.predict(x)[0]
#     top_drugs = top_drugs_extract(prediction,df_filtered)
#     top_drugs_str = ",".join(top_drugs)
#     return render_template("index.html",prediction=prediction , text=text , top_drugs=top_drugs_str)


# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import pickle
import joblib
import pandas as pd
from utils import top_drugs_extract

app = Flask(__name__)

# Assume model loading and other necessary imports are done here
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf.pkl")
result = joblib.load("results.pkl")
df_filtered = pd.read_csv('df_filtered.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    x = vectorizer.transform([text])
    prediction = model.predict(x)[0]
    top_drugs = top_drugs_extract(prediction, df_filtered)
    top_drugs_str = ','.join(top_drugs)
    
    # Return JSON response
    return jsonify({
        'text': text,
        'prediction': prediction,
        'top_drugs': top_drugs_str
    })

if __name__ == '__main__':
    app.run(debug=True)

