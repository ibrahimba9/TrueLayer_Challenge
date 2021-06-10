# True Layer Challenge - Movie Classifier
## Name: Ibrahim Ben Abdallah
## E-mail: ibrahim.b.abdallah@gmail.com

# Description:
- A command line application that allows to classifiy a movie by genre given the title and the description.
- Input format example: MovieClassifier --title "movie title" --description "A description of the movie."
- Output format example: 
{
    "title": "movie title",
    "description": "A description of the movie.",
    "genre": "drama"
}

# Attachments:
- TrueLayer_Challenge folder contains the files for this challenge's solution
- test.py: python Script. Contains the script for the automated test of the application.
- movie_classifier.ipynb: Jupyter Notebook. Contains the Workflow of the training process.
- movie_classifier: Folder: Contains the script for movie classifier application, dependecies and the Dockerfile.
- MovieClassifier.py: Python Script. Contains the script for the application.
- label_encoder.pkl: Pickle file. Label Encoder.
- tfidf_vectorizer.pkl: Pickle file. TF-IDF vectorizer.
- xgb_movie_classifier.pkl: Pickle file. XGBoost classifier. Trained model.
- Dockerfile


# How to run the application:
- install Docker Desktop
- run Docker Desktop
- Launch VS-Code
- Open a new Terminal
- Open TrueLayer_Challenge/movie_classifier folder
- run the command : "docker build -t movie-classifier ."
- run the command : "docker run --rm movie-classifier --title "example name" --description "example description"

# How to run the automated tests:
- Launch VS-Code
- Open Folder "TrueLayerChallenge"
- The file test.py contains the script for the automated test of the application
- We used the unittest module for running the automated tests
- Open new terminal
- run command: python test.py

# Programming Language:
- Python 3.6.9

# Training Workflow:
- The attached notebook movie_classifier.ipynb contains the pipeline followed for training the movie classifier model.
- It's well commented and contains a clear description for every step.

# Training Data Set:
- https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv

# Training Algorithms:
- Linear Model: Logistic Regression
- Tree Based Model: Decision Tree
- Ensemble Method: Random Forest
- Boosted Model: XGBoost

# Preprocessing Techniques:
- TF-IDF vectorizer
- Label Encoder

# Requirements:
- Click==7.0
- nltk==3.4.5
- pandas==1.1.4
- matplotlib==2.2.3
- scikit-learn==0.24.1
- xgboost==0.90