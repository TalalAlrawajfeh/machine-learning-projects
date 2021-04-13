# IMDB Reviews Classifier

## Overview

This classifier is trained to decide whether a given review is positive or negative.

Data Set: https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset/
Processed & Normalized Data Set: https://drive.google.com/open?id=1FzxYpKlSWH_Qw26f-j-sb2ZG5u359N2u

## Data Preprocessing

We start by pre-processing the reviews within the `imdb_master.csv` file. The output of the pre-processed reviews is a list of normalzied representations of the original reviews. The normalized representations are obtained by the following steps:

* Remove any empty review.
* Downcase all the letters in the reviews.
* Tokenize the reviews.
* Remove all stop words and non-alphabetic forms.
* Lemmatize all the words in each review. 

The output will be a list of lemmatized words representing the original review.

We faced a problem in the last step since lemmatization is relatively slow. To speed things up we did the following:

* Use LRU cache to maintain lemmas of most common words.
* Use multiple workers and divide the data to be processed by these workers.

This way we obtained a good speed multiplier (theoretically 12 times faster = 6 workers each working on an independent cpu core * 2 times speed-up from using the LRU-cache).

## Model Choice

Over many available choices for models, we preferred using an SVM for two reasons:

* It's known high accuracy for binary classification.
* Although being slightly slower than Naive Bayes, its much faster than many neural network models.

## Results

We obtained over `88%` accuracy score on the test data. The program needed less than 2 hours to pre-process the data and train the model on a machine with core i7 1.80 GHZ intel cpu and 8 GB RAM running fedora linux. Python version is `3.7.3`.

## Files

* `classifier.py` contains the source code.
* `normalized_data.py` contains the pre-processed data so we don't wait everytime we run the model.
* `model.pkl` contains the svm model.
* `report` contains the scikit-learn metrics classifications report.

## External Libraries

These are all the external libraries used in the source code:

* pandas
* scikit-learn
* nltk

Please make sure to download nltk packages using the following python code:
```
import nltk
nltk.download()
```
