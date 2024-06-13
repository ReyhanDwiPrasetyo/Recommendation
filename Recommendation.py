import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class ArticleRecommendation:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words=list(self.stopwords), analyzer='word', lowercase=True, ngram_range=(1, 3), min_df=1, use_idf=True)

    def stopwords_remove(self, text):
        words = nltk.word_tokenize(text)
        stopword_removed = " ".join(word for word in words if word.lower() not in self.stopwords)
        return stopword_removed

    def stem(self, text):
        words = nltk.word_tokenize(text)
        stemmed_text = " ".join([self.stemmer.stem(word) for word in words])
        return stemmed_text

    def preprocessed(self, text):
        text = text.strip()
        text = re.sub("[^a-zA-Z\s]", "", text)  # Allow spaces and letters
        text = self.stopwords_remove(text)
        text = self.stem(text)
        text = text.strip()
        return text

    def recommendation(self, dataframe_path, user_input):
        df = pd.read_csv(dataframe_path)
        user_input = self.preprocessed(user_input)
        data_vector = self.vectorizer.fit_transform(df['content_preprocessed']).toarray()
        user_vector = self.vectorizer.transform([user_input]).toarray()
        cos_sim = cosine_similarity(data_vector, user_vector)
        df['cos_sim'] = cos_sim[:, 0]  # Assigning the similarity score to a new column
        result = df.sort_values("cos_sim", ascending=False).head(5)
        return result[['Kode', 'title', 'raw_content', 'clean_content', 'date_created', 'author', 'articleLink','imageSrc']]

# Usage example:
# recommendation_system = ArticleRecommendation()
# result = recommendation_system.recommendation("path_to_your_dataframe.csv", "sample user input text")
# print(result)