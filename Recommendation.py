import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class ArticleRecommendation:
    def __init__(self):
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopwords = StopWordRemoverFactory().get_stop_words()
        self.vectorizer = TfidfVectorizer(
            stop_words=self.stopwords,
            analyzer="word",
            lowercase=True,
            ngram_range=(1, 3),
            min_df=0,
            use_idf=True,
        )

    def clean_text(self, text):
        text = text.strip()
        text = re.sub("[^a-z ]", "", text)
        text = self.stemmer.stem(text)
        text = " ".join(word for word in text.split() if word not in self.stopwords)
        text = text.strip()
        return text

    def recommendation(self, dataframe_path, user_input):
        df = pd.read_csv(dataframe_path)
        user_input = self.clean_text(user_input)
        data_vector = self.vectorizer.fit_transform((df.content_preprocessed)).toarray()
        user_vector = self.vectorizer.transform([user_input]).toarray()
        cos_sim = cosine_similarity(data_vector, user_vector)
        df["cos_sim"] = cos_sim
        result = df.sort_values("cos_sim", ascending=False).head(5)
        return result[["Kode", "title", "content", "date_created", "author", "link"]]
