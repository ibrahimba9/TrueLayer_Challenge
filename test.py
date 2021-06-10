import unittest
import numpy as np

from movie_classifier import MovieClassifier

class TestMovieClassifier(unittest.TestCase):
    def test_vectorize_text_none(self):
        """
        Description: 
            Test vectorize_text output is not none
        Input:
            Text: str
        Output:
            str
        """
        result = MovieClassifier.vectorize_text("random text")
        self.assertIsNotNone(result)
    
    def test_vectorize_text_shape(self):
        """
        Description: 
            Test vectorize_text output shape
        Input:
            Text: str
        Output:
            str
        """
        result = MovieClassifier.vectorize_text("random text")
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 4000)
    
    def test_text_preprocessing_none(self):
        """
        Description: 
            Test text_preprocessing output is not none
        Input:
            Text: str
        Output:
            str
        """
        result = MovieClassifier.text_preprocessing("random text")
        self.assertIsNotNone(result)
    
    def test_text_preprocessing_type(self):
        """
        Description: 
            Test text_preprocessing output type is string
        Input:
            Text: str
        Output:
            str
        """
        result = MovieClassifier.text_preprocessing("random text")
        self.assertIsInstance(result, str)
    
    def test_text_preprocessing_none(self):
        """
        Description: 
            Test text_preprocessing output is correct
        Input:
            Text: str
        Output:
            str
        """
        result1 = MovieClassifier.text_preprocessing("Random text") # check if text is lowerd
        self.assertEqual(result1, "random text") 

        result2 = MovieClassifier.text_preprocessing("Random text!") # check if special characters are removed
        self.assertEqual(result2, "random text")

        result3 = MovieClassifier.text_preprocessing("A random     tExt") # check if extra spaces are removed
        self.assertEqual(result3, "random text")

        result4 = MovieClassifier.text_preprocessing("The movies aRe    great!!!") # check if stopwords are removed and words are lemmatized
        self.assertEqual(result4, "movie great")
    
    def test_classify_movie_none(self):
        """
        Description: 
            Test classify_movie output is not none
        Input:
            Text: str
        Output:
            str
        """
        movie_plot_example = "V for Vendetta takes place in post-nuclear-war fascist England, where a ruthless vigilante appears signing his deeds with the letter V. Obsessed with the memory of a culture now forbidden and gone, cruel and terribly intelligent, V attacks the strongest symbols of the dictatorship, driven by an immense desire for revenge and unspeakable hatred..  The Commander’s police are ordered to put an end to their actions as soon as possible..."
        features_example = MovieClassifier.vectorize_text(MovieClassifier.text_preprocessing(movie_plot_example))
        result = MovieClassifier.classify_movie(features_example)
        self.assertIsNotNone(result)

    def test_classify_movie_type(self):
        """
        Description: 
            Test classify_movie output type is string
        Input:
            Text: str
        Output:
            str
        """
        movie_plot_example = "V for Vendetta takes place in post-nuclear-war fascist England, where a ruthless vigilante appears signing his deeds with the letter V. Obsessed with the memory of a culture now forbidden and gone, cruel and terribly intelligent, V attacks the strongest symbols of the dictatorship, driven by an immense desire for revenge and unspeakable hatred..  The Commander’s police are ordered to put an end to their actions as soon as possible..."
        features_example = MovieClassifier.vectorize_text(MovieClassifier.text_preprocessing(movie_plot_example))
        result = MovieClassifier.classify_movie(features_example)
        self.assertIsInstance(result, str)
    
    def test_classify_movie_correct(self):
        """
        Description: 
            Test classify_movie output is correct and exists 
        Input:
            Text: str
        Output:
            str
        """
        movie_genres = ['Animation', 'Adventure', 'Romance', 'Comedy', 'Action', 'Family',
            'History', 'Drama', 'Crime', 'Fantasy', 'Science Fiction',
            'Thriller', 'Music', 'Horror', 'Documentary', 'Mystery', 'Western',
            'TV Movie', 'War', 'Foreign']
        movie_plot_example = "V for Vendetta takes place in post-nuclear-war fascist England, where a ruthless vigilante appears signing his deeds with the letter V. Obsessed with the memory of a culture now forbidden and gone, cruel and terribly intelligent, V attacks the strongest symbols of the dictatorship, driven by an immense desire for revenge and unspeakable hatred..  The Commander’s police are ordered to put an end to their actions as soon as possible..."
        features_example = MovieClassifier.vectorize_text(MovieClassifier.text_preprocessing(movie_plot_example))
        result = MovieClassifier.classify_movie(features_example)
        self.assertIn(result, movie_genres)


if __name__ == '__main__':
    unittest.main()