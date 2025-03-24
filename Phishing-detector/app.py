import pickle
from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
from sklearn import metrics
import warnings
from sqlalchemy import desc
from datetime import datetime
from convert import convertion
from feature import FeatureExtraction
import csv
from flask_migrate import Migrate

warnings.filterwarnings('ignore')

file = open("newmodel.pkl", "rb")
gbc = pickle.load(file)
file.close()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

class URLData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(2083), nullable=False)
    prediction = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)



@app.route("/social")
def social():
    return render_template("social.html")

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/result', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        url = request.form["name"]
        
        # Check if the URL is in the phishurls.csv file with utf-8 encoding
        with open('DataFiles/phishurls.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            phishing_urls = [row[0] for row in reader]
        
        if url in phishing_urls:
            y_pred=-1  
        else:
            obj = FeatureExtraction(url)
            x = np.array(obj.getFeaturesList()).reshape(1, 30)
            y_pred = gbc.predict(x)[0]
        
        if y_pred == 1:
            name = "Safe"
        else:
            name = "Phishing"
        
        # Save the result to the database
        new_url_data = URLData(url=url, prediction=int(y_pred))
        db.session.add(new_url_data)
        db.session.commit()
    
        return render_template("index.html", name=name, url=url)

@app.route('/usecases', methods=['GET', 'POST'])
def usecases():
    return render_template('usecases.html')

@app.route('/urls')
def urls():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    # Fetch URLs sorted by most recent first
    paginated_urls = URLData.query.order_by(desc(URLData.created_at)).paginate(
        page=page, 
        per_page=per_page, 
        error_out=False
    )
    
    total_pages = paginated_urls.pages
    
    return render_template(
        'urls.html', 
        url_data=paginated_urls.items, 
        total_pages=total_pages,
        current_page=page
    )


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)