# import modules
import click
import re
from nltk.corpus import stopwords
import pickle
import nltk
import os
import sys

def remove_stopwords(text):
    """
    Description: 
        Remove stopwords from a given text
    Input:
        Text: str
    Output:
        str
    """
    stopwords_list = set(stopwords.words('english'))
    clean_text = [x for x in text.split() if not x in stopwords_list]
    return " ".join(clean_text)

def lemmatization(text):
    """
    Description: 
        Apply Lemmatization to a given text. This function allows to take each word from the given string to its root form called Lemma. It helps to bring words to their dictionary form.
    Input:
        Text: str
    Output:
        str
    """
    lemmatizer = nltk.WordNetLemmatizer()
    clean_text = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(clean_text)

def text_preprocessing(text):
    """
    Description: 
        this function cleans an input text and return a 'clean' version ready to be vectorized.
        This function removes extra spaces, remove apostrophies and all the special characters and lower all the remaining characters from a given string.
        this function also performs lemmatization and removes stopwords.
    Input:
        Text: str
    Output:
        A clean version of the input text: str
    """
    text = re.sub("\s+", " ", text)
    text = re.sub("\'", "", text) 
    text = re.sub("[^A-Za-z ]", "", text)
    text = text.lower()
    text = remove_stopwords(text)
    text = lemmatization(text)
    return text

def vectorize_text(text):
    """
    Description: 
        this function transform a text to a feature vector using the TF-IDF vectorizer algorithm
    Input:
        Text: str
    Output:
        Ascipy.sparse.csr.csr_matrix
    """
    print(os.path.dirname(__file__))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    features = tfidf_vectorizer.transform([text])
    return features

def classify_movie(input_features):
    """
    Description: 
        this function loads a classifier and predicts a genre given a feature vector. 
        Then transforms the model prediction to the original class using a label encoder.
    Input:
        Feature Vector: Ascipy.sparse.csr.csr_matrix
    Output:
        str
    """
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgb_movie_classifier.pkl"), "rb") as f:
        movie_classifier_model = pickle.load(f)
    predicted_genre = movie_classifier_model.predict(input_features)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_encoder.pkl"), "rb") as f2: 
        label_encoder = pickle.load(f2)
    predicted_genre = label_encoder.inverse_transform(predicted_genre)
    return predicted_genre.item()

@click.command() # configure input arguments
@click.option( # title argument
    '--title',
    help='The movie title.',
    required=True,
    type=str
    )
@click.option( # description argument
    '--description',
    help='The movie description.',
    required=True,
    type=str
    ) 
def main(title, description):
    """
        CLI to classify movies by genre.
    """
    try:
        assert title is not None, "Title does not exist!" # check if title argument exists
        assert description is not None, "Description does not exist!" # check if description argument exists
        assert len(title) > 0, "Title is empty!" # check if title argument is not empty
        assert len(description) > 0, "Description is empty!" # check if description argument is not empty

        clean_text = text_preprocessing(description) # clean input description

        features_vector = vectorize_text(clean_text) # create feature vector using TF-IDF vectorizer

        predicted_genre = classify_movie(features_vector) # apply prediciton using a pre-trained model

        # prepare output format
        output = {
            "title": title,
            "description": description,
            "genre": predicted_genre
        }
        click.echo(output) # print output
    except AssertionError as msg: # catch exceptions 
        print(msg)
        return msg 


if __name__ == '__main__': # run app
    main()