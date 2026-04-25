from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
app = Flask(__name__)




# 📄 Extract text using PyPDF2
def extract_text(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    
    for page in reader.pages:
        try:
            text += page.extract_text() or ""
        except:
            pass
            
    return text

# 🧹 Preprocess
def preprocess(text):
    words = text.lower().split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# 🎯 ATS Score
def calculate_score(resume, jd):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume, jd])
    score = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(score[0][0]*100, 2)

# 🔍 Missing keywords
def missing_keywords(resume, jd):
    resume_words = set(resume.split())
    jd_words = set(jd.split())
    missing = jd_words - resume_words
    return list(missing)[:20]

@app.route("/", methods=["GET", "POST"])
def home():
    score = None
    missing = []

    if request.method == "POST":
        file = request.files["resume"]
        jd = request.form["jd"]

        resume_text = extract_text(file)
        resume_text = preprocess(resume_text)
        jd = preprocess(jd)

        score = calculate_score(resume_text, jd)
        missing = missing_keywords(resume_text, jd)

    return render_template("index.html", score=score, missing=missing)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
