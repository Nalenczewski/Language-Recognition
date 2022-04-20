![](./README_files/flag.jpg)

# Language Recognition and First Sentence Prediction

## Contents
1. **Introduction**
2. **My Approach**
3. **Findings**
4. **Ideas for Further Research**
5. **Recommendations**

## 1. Introduction
Today, we take it for granted that Google can automatically detect a what language a given text is written in and then translate the text with a high degree of accuracy. The goal of my model is to recognize what language a given text is written in and output the language name.

I used a [language identification dataset from Kaggle](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst) that comprises 22,000 paragraphs of 22 languages and is taken from the original WiLI-2018 Wikipedia language identification benchmark dataset that contians 235,000 paragraphs of 235 langauges. The languages in the dataset I used are:
* English
* Arabic
* French
* Hindi
* Urdu
* Portuguese
* Persian
* Pushto
* Spanish
* Korean
* Tamil
* Turkish
* Estonian
* Russian
* Romanian
* Chinese
* Swedish
* Latin
* Indonesian
* Dutch
* Japanese
* Thai

The data consist of two columns: a natural language text and a categorial label. The data contains 1,000 examples of each language for a total of 22,000 examples.

A secondary goal of this project is building a model that can predict whether a sentence is the first in a paragraph or not.

## 2. My Approach

**Language Recognition**

I built four different models by first vectorizing the text data using Count Vectorizer and Tf-idf, and then training Logistic Regression and Naive Bayes models on the training data. I then evaluated the performance of each model using accuracy score, precision, and recall. I also did cross validation for each model to make sure the accuracies I was getting were reasonable.


After finding that the Tf-idf model with Naive Bayes had the highest accuracy on the test data, I reduced overfitting by instantiating models with different alpha values and found that a model with alpha equal to 0.1 was ideal.

**Predicting Language of Text Taken from the Internet**

I further tested each model's predictive ability by building a function that would take in a paragraph of text in one of the 22 langauges and output its label. The data was taken from various news and other sites on the internet.

Lastly, I built a model that tokenized Chinese and Japanese and merged the document term matrices for these langauges with that of the other languages. I then made predictions and evaluated the model.

**First Sentence Prediction**

For the second half of this project, I built a model that can identify whether a sentence is the first in its paragraph or not. To do this, I first chose one language to work with, in this case, Chinese. I also chose Chinese because it was the only data which still had punctuation in it. 

Next, I needed to create a new self-supervised dataset containing individual sentences and a label indicating if the sentence is the first of a paragraph or not. To this end, I constructed a function that would use spaCy to create spaCy document objects from the paragraphs, split the data up into sentences using the sents method on the document object, label each sentence as the first in its paragraph or not, and construct the new dataframe containing all of the individual sentences from each paragraph and their labels.

To keep the sentneces from being taken out of context during the train-test split, I first did train-test split on the individual paragraphs. Then, I applied the function to create new dataframes for training data and test data.

Because I was dealing with a highly imbalanced dataset, I did random oversampling on the minority class (the first sentneces).

At this point, I was ready to do a latent semantic analysis (LSA) on the data in order to create document vectors that could then be fed to a machine learning model for making predictions. To do this, I used CountVectorizer and Tf-idf to transform the sentences into a bag of words. I then used Truncated SVD with 75 components to turn the document term matrices into latent semantic analyses. 

Once I had the latent semantic analysis vectors, I fit a Logistic Regression model on the data, made predictions on the training and test sets, and evaluated the results using accuracy score, precision, and recall.

In order to prevent overfitting and improve accuracy, I tried creating models with different n_components values for the SVD, namely, 25, 50, and 100. I then fit models to these new datas, made predictions, and evaluated the results.

After getting the results, I found that Tf-idf with LSA with 50 SVD components performed the best. So far, the model has been optimizing for accuracy. But in a classification problem with a miority class, it is more important to optimize for F1 score. So, I created a grid search with the scoring parameter set to f1 in order to optimize C for f1 score instead of accuracy. 

## 3. Findings

**Language Recognition**

Below is a table showing the accuracy scores of the models. Count Vectorizer with Naive Bayes performed the best.


```python
%store -r accuracy_df
accuracy_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Training Data Accuracy</th>
      <th>Test Data Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Count Vectorizer</th>
      <th>Logistic Regression</th>
      <td>0.993295</td>
      <td>0.939773</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.991023</td>
      <td>0.959318</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Tf-idf</th>
      <th>Logistic Regression</th>
      <td>0.974034</td>
      <td>0.941364</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.983920</td>
      <td>0.939545</td>
    </tr>
  </tbody>
</table>
</div>



Below are the cross validation scores.


```python
%store -r cross_val_scores
cross_val_scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Cross Validation Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Count Vectorizer</th>
      <th>Logistic Regression</th>
      <td>0.947773</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.955409</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Tf-idf</th>
      <th>Logistic Regression</th>
      <td>0.955000</td>
    </tr>
    <tr>
      <th>Naive Bayes</th>
      <td>0.954545</td>
    </tr>
  </tbody>
</table>
</div>



**Predicting Language of Text Taken from the Internet**

Most languages were predicted correctly, however Japanese and Chinese were sometimes predicted as other languages, such as Japanese as Russian or Chinese as Japanese. Sometimes, when a prediction was made on the same language multiple times, the algorithm would predict different languages from one time to the next. I believe the reason for this is because Chinese and Japanese do not contain spaces and therefore cannot be tokenized.

Therefore, I built a model that tokenized Japanese and Chinese and merged that data with the rest of the language data before training the previous best model (Count Vectorizer with Naive Bayes, alpha=0.1) on the data and making predictions. I found that the accuracy was 98% on the test and training data, and the F1 scores for Chinese and Japanese were vastly better than before at around 99%.

**First Sentence Prediction**

The Tf-idf LSA model performed better than the CountVectorizer model. When I change the SVD components parameter, I found that 50 was the ideal value. Below are the results.


```python
%store -r accuracy_f1
accuracy_f1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Training Data</th>
      <th>Test Data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Count Vectorizer</th>
      <th>Accuracy</th>
      <td>0.645660</td>
      <td>0.608365</td>
    </tr>
    <tr>
      <th>F1 Score</th>
      <td>0.608329</td>
      <td>0.558414</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Tf-idf</th>
      <th>Accuracy</th>
      <td>0.689931</td>
      <td>0.621673</td>
    </tr>
    <tr>
      <th>F1 Score</th>
      <td>0.678198</td>
      <td>0.619139</td>
    </tr>
  </tbody>
</table>
</div>




```python
%store -r tf_svd_25_50_75_100
tf_svd_25_50_75_100
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Training Data</th>
      <th>Test Data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">25 components</th>
      <th>Accuracy</th>
      <td>0.665799</td>
      <td>0.618821</td>
    </tr>
    <tr>
      <th>F1 Score</th>
      <td>0.655820</td>
      <td>0.630415</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">50 components</th>
      <th>Accuracy</th>
      <td>0.662326</td>
      <td>0.629753</td>
    </tr>
    <tr>
      <th>F1 Score</th>
      <td>0.645138</td>
      <td>0.638850</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">75 components</th>
      <th>Accuracy</th>
      <td>0.689931</td>
      <td>0.621673</td>
    </tr>
    <tr>
      <th>F1 Score</th>
      <td>0.678198</td>
      <td>0.619139</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">100 components</th>
      <th>Accuracy</th>
      <td>0.659549</td>
      <td>0.614068</td>
    </tr>
    <tr>
      <th>F1 Score</th>
      <td>0.628669</td>
      <td>0.565310</td>
    </tr>
  </tbody>
</table>
</div>



When I optimized C for F1 score, I found that the ideal value of C was still 1, so the accuracies didn't change when compared with the previous best model (Tf-idf with 50 components SVD).


```python
%store -r tf_optimized
tf_optimized
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Training Data</th>
      <th>Test Data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Un-optimized</th>
      <th>Accuracy</th>
      <td>0.662326</td>
      <td>0.629753</td>
    </tr>
    <tr>
      <th>F1 Score</th>
      <td>0.645138</td>
      <td>0.638850</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Optimized for F1</th>
      <th>Accuracy</th>
      <td>0.662326</td>
      <td>0.629753</td>
    </tr>
    <tr>
      <th>F1 Score</th>
      <td>0.645138</td>
      <td>0.638850</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Ideas for Further Research

**Language Recognition and Predicting Langauge of Text Taken from the Internet**

* Try to solve the same problem but instead of using machine learning, use language dictionaries.
* I could expand the langauges corpus to include more languages.

**First Sentence Prediction**

* I could try inputting some sentences from paragraphs gathered on the Chinese web and see how well the model predicts the sentence labels.
* Follow the same process but for Wikipedia data in English. To do this, I would need to scrape Wikipedia.


## 5. Recommendations

**Language Recognition and Predicting Langauge of Text Taken from the Internet**

The language recognition model could be incorporated into an app that recognizes a language in order to translate it correctly.

**First Sentence Prediction**

It could be used to build a list of most common first sentences, which could then be characterized. 
