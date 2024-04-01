from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import TextAreaField
from nltk.sentiment import SentimentIntensityAnalyzer
import random
import string

app = Flask(__name__)

# ランダムな文字列を生成してSECRET_KEYに設定
chars = string.ascii_letters + string.digits
secret_key = ''.join(random.choice(chars) for i in range(32))
app.config['SECRET_KEY'] = secret_key

sia = SentimentIntensityAnalyzer()

class TextForm(FlaskForm):
    text = TextAreaField()

@app.route('/', methods=['GET', 'POST'])
def home():
    form = TextForm()
    if request.method == 'POST':
        text = request.form['text']
        scores = sia.polarity_scores(text)
        return render_template('result.html', text=text, scores=scores)
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
