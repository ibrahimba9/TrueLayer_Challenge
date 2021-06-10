FROM python:3.6.9

ADD MovieClassifier.py .
ADD xgb_movie_classifier.pkl .
ADD tfidf_vectorizer.pkl .
ADD label_encoder.pkl .

RUN pip install Click==7.0 nltk pandas==1.1.4 matplotlib==2.2.3 scikit-learn==0.24.1 xgboost==0.90

RUN python -m nltk.downloader stopwords

RUN python -m nltk.downloader wordnet

ENTRYPOINT [ "python", "MovieClassifier.py" ]