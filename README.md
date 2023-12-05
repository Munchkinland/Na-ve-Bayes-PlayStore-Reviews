# Naive Bayes Classifier for Sentiment Analysis

This repository contains a Jupyter notebook (`NaiveBayes.ipynb`) that implements a Naive Bayes classifier for sentiment analysis. The classifier is trained on a dataset of Play Store reviews and predicts the sentiment (positive or negative) of new reviews.

## Notebook Overview

- **Dependencies:**
  - pandas
  - scikit-learn
  - nltk
    
- **Dataset:** The notebook uses the Play Store reviews dataset, which can be accessed [here](https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv).

## Dataset Details

The Play Store reviews dataset contains user reviews along with sentiment labels (positive or negative). The dataset is sourced from [4GeeksAcademy's tutorial](https://github.com/4GeeksAcademy/naive-bayes-project-tutorial). It serves as the basis for training and evaluating the sentiment analysis model.

## Preprocessing Steps

1. Tokenization and removal of stop words.
2. Lemmatization (optional).
3. Removal of special characters and punctuation.
4. Text normalization (convert to lowercase).
5. Text vectorization using TF-IDF.
6. Optional: Removal of rare or very frequent words.

## Data Splitting

The dataset is split into training and testing sets using scikit-learn's `train_test_split` function.

## Model Training

The notebook utilizes the Multinomial Naive Bayes algorithm for sentiment classification. The model is trained and evaluated using accuracy as the performance metric.

## Model Hyperparameter Tuning

A random search is performed to find optimal hyperparameters for the Multinomial Naive Bayes model, including the alpha parameter and whether to fit class prior probabilities.

## Model Evaluation

The model's accuracy is evaluated on the test set, and the best hyperparameters are printed.

## Saving the Model

The trained Multinomial Naive Bayes model is saved to a file (`multinomial_nb_model.pkl`). Additionally, a version of the model with performance information is saved to `multinomial_nb_model_with_info.pkl`.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/naive-bayes-sentiment-analysis.git
Install the required dependencies:

pip install -r requirements.txt
Open and run the Jupyter notebook (NaiveBayes.ipynb) in your local environment.

Feel free to explore, modify, and use this repository for your own sentiment analysis tasks!
