from flask import Flask, request, jsonify
from Recommendation import ArticleRecommendation
from flask import Response
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)


@app.route("/")
def index():
    return "Hello World"


@app.route("/recommend", methods=["POST"])
def recommend():
    input_user = request.get_json(force=True)
    if "text" not in input_user:
        return (
            jsonify(error="Bad Request", message="JSON must contain 'text' attribute"),
            400,
        )
    text = input_user["text"]
    model = ArticleRecommendation()
    result = model.recommendation(
        "C:\Users\Reyhan Dwi\Documents\Reyhan\Bangkit Academy\Final Capstone\englishArticle.csv",
        text,
    )
    return result.to_json(orient="records", lines=True)


if __name__ == "__main__":
    app.run(debug=True)
