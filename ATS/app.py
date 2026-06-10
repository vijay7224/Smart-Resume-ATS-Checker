
from flask import Flask, render_template, request, redirect, flash, session
from pymongo import MongoClient
import PyPDF2
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# Flask App Configuration
# =====================================

app = Flask(__name__)
app.secret_key = "secretkey123"

# =====================================
# MongoDB Connection
# =====================================

client = MongoClient("mongodb://localhost:27017/")

db = client["resume_analyzer"]

users_collection = db["users"]

# =====================================
# PDF Text Extraction
# =====================================

def extract_text(file):

    text = ""

    try:

        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

    except Exception as e:

        print("PDF Error:", e)

    return text


# =====================================
# Text Preprocessing
# =====================================

def preprocess(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        ' ',
        text
    )

    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but',
        'is', 'are', 'was', 'were', 'to',
        'of', 'in', 'on', 'for', 'with',
        'at', 'by', 'from', 'up', 'about',
        'into', 'over', 'after'
    }

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# =====================================
# ATS Score Calculation
# =====================================

def calculate_score(resume, jd):

    try:

        tfidf = TfidfVectorizer()

        vectors = tfidf.fit_transform(
            [resume, jd]
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )

        return round(
            similarity[0][0] * 100,
            2
        )

    except:

        return 0


# =====================================
# Missing Keywords
# =====================================

def missing_keywords(resume, jd):

    resume_words = set(
        re.findall(
            r'\w+',
            resume.lower()
        )
    )

    jd_words = set(
        re.findall(
            r'\w+',
            jd.lower()
        )
    )

    missing = jd_words - resume_words

    ignore_words = {
        "the", "and", "or", "is", "are",
        "a", "an", "to", "for", "of",
        "in", "on", "with"
    }

    filtered = [
        word for word in missing
        if word not in ignore_words
    ]

    return filtered[:20]


# =====================================
# Home Page
# =====================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# =====================================
# Registration
# =====================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form["name"]

        email = request.form["email"]

        password = request.form["password"]

        existing_user = users_collection.find_one(
            {"email": email}
        )

        if existing_user:

            flash(
                "Email already registered!"
            )

            return redirect(
                "/register"
            )

        users_collection.insert_one({

            "name": name,

            "email": email,

            "password": password

        })

        flash(
            "Registration Successful!"
        )

        return redirect("/login")

    return render_template(
        "register.html"
    )


# =====================================
# Login
# =====================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form["email"]

        password = request.form["password"]

        user = users_collection.find_one({

            "email": email

        })

        if user and user["password"] == password:

            session["user"] = user["name"]

            flash(
                "Login Successful!"
            )

            return redirect("/ats")

        flash(
            "Invalid Email or Password"
        )

    return render_template(
        "login.html"
    )


# =====================================
# ATS Resume Analyzer
# =====================================

@app.route(
    "/ats",
    methods=["GET", "POST"]
)
def ats():

    if "user" not in session:

        return redirect("/login")

    score = None

    missing = []

    if request.method == "POST":

        if "resume" not in request.files:

            flash("Please upload resume")

            return redirect("/ats")

        file = request.files["resume"]

        jd = request.form["jd"]

        resume_text = extract_text(file)

        processed_resume = preprocess(
            resume_text
        )

        processed_jd = preprocess(
            jd
        )

        score = calculate_score(
            processed_resume,
            processed_jd
        )

        missing = missing_keywords(
            resume_text,
            jd
        )

    return render_template(

        "index.html",

        score=score,

        missing=missing,

        user=session["user"]

    )


# =====================================
# Logout
# =====================================

@app.route("/logout")
def logout():

    session.clear()

    flash("Logged Out Successfully")

    return redirect("/login")


# =====================================
# Run Application
# =====================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )

