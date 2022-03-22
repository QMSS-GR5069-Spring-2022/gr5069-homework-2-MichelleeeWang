# Sentiment Analysis: Final Project Report

### Authors:
(1) Zujun Peng (zp2224)
(2) Jingyuan Wang(jw4000)
(3) Wen Chen(wc2787)
(4) Rahul Subramaniam (rs4128)
(5) Sung Jun Won (sw3049)
___

## Problem Statement: 
In this project, we work on sentiment140 dataset on tweets contents and our goal is to make sentiment analysis to predict whether it is a positive or negative word. We take the advantages of natural language processing techniques to preprocess the text data and construct machine learning and deep learning models such as logistic regression, decision tree, lstm, and so on to conduct sentiment classification.

___ 

## Data Cleaning and Data Exploration:
We collect the sentiment 140 dataset from kaggle. Since the dataset has equal numbers of positive and negative labels, we obtain a balanced dataset. Moreover, the dataset has no missing values. Except for text data, the data cleaning and preprocessing we do on the whole dataset includes reorganizing the posting time of tweets, changing the positive/negative labels to 1 and 0. We will explain how we process text data in the next section. 
We also conduct data exploration analysis on the distribution of the labels, sentiment change over time and different users. We also draw two word cloud images for positive tweets (left) and negative tweets (right). The left includes most frequently used words in positive tweets including thank, good, day, love; the right includes most frequently used words in negative tweets including `work, now, go, day`. Overall, the frequent words shown by the word cloud are good representations of the emotion. For example, people might think working is tedious and therefore make negative tweets on that.

___ 

## Text-Preprocessing Techniques Employed:
### TFIDF Vectorizer: 
Tf-idf vectorizer counts the token frequency considering the document frequency of each token. It scales down the impact of tokens that occur very frequently in the text dataset. This can help we learn more features on the tokens that occur in a small fraction of the training text dataset. The formula of Tf-idf can be expressed as:

    idf(t) = log [ (1 +n ) / (1 + df (t))] + 1
    
Where n is the total number of documents in the document set and df(t) is the document frequency of t. In our project, we use a n-gram range from 1 to 3, we also set the max_df = 0.8 to further filter out the terms that have a document frequency strictly higher than the given value of max_df.

### Word2Vec:
The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text by taking information in the context window into account. After training the model, we get a dense vector representation for each word in the text data by extracting the weights matrix updated after the training process. The following graph shows the basic structure of how a word2vec model works. In our word embedding training model, We utilize the gensim library to train our Word2Vec model. Regarding the parameter setting, we set the context word window-size to be 3, dimension of the word embedding to be 100 to obtain an informative word embedding matrix of our input text data. 

Since sentences in different samples have different lengths, we take both the mean value and the maximum value of word embedding over the whole sentence. In this way, we can obtain a sentence representation of dimension 100 (same dimension as word embedding) for each sentence in the dataset, and these sentence representations can then be used in the classification model for sentiment analysis.

### Countvectorizer using Keras Tokenizer
Countvectorizer with Keras Tokenizer allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count. Keras tokenizer also gives an option to use the tfidf method of vectorization by specifying text_to_matrix(mod=’tfidf’) instead of text_to_sequence(). The tokenizer allows limiting the number of embedding entries for each sentence. The project uses the Keras Tokenizer for the LSTM modelling. Above one (diagram to the right) explains how the Keras Tokenizer uses the Count Vectorizer.

___ 

## Machine Learning Models Employed:
### Logistic Regression
Logistic regression is a classification algorithm and it is widely used to predict a binary outcome given a set of independent variables. It is easier to implement, interpret, and very efficient to train so we firstly try to fit the logistic regression model on datasets.

### Support Vector Machines (SVM)
Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both
classification or regression challenges. It is relatively memory efficient and is more effective in high dimensional space. In this algorithm, we plot each data item as a point in n-dimensional space (where n is the number of features in raw data) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiate the two classes.

### KNN (k-nearest neighbors)
K-nearest neighbors (KNN) is a type of supervised learning algorithm used for both regression and classification. It does not have any parameters to learn and is also easy to implement. It tries to predict the correct class for the test data by calculating the distance between the test data and all the training points. Then select the K number of points which is closest to the test data.

### Decision Trees
Decision Tree is a supervised machine learning algorithm that splits the data according to some parameters (max_depth, min_samples_split, etc) and makes decisions on the leaf nodes. Decision Tree is used for both regression and classification. The training takes a huge amount of time if the data is big, and it is very prone to overfit if the tree branches out too far.

### XGBoost
XGBoost is gradient boosted decision trees, mainly designed to outspeed and outperform the sklearn’s GradientBoostingClassifier. It has a more regularized model formalization, which means that it is able to deal with overfitting better than GradientBoostingClassifier. XGBoost is used for both classification and regression. Similar to Decision Tree(as it is a group of boosted decision trees), training takes a huge amount of time if the data is big.

### LTSM
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. It is an improvement over the traditional RNN Model as it helps solve the popular vanishing gradient problem. A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell. It has great applications in the field of Time Series Modelling, Image Captioning, and several others.

___ 

## Conclusion
Compared with different models, logistic regression has the best performance than the other models followed by LSTM. Decision tree has the lowest performance. But considering the efficiency, we use only part of the dataset when training SVM, KNN and decision tree and full dataset for the other models. We also utilized three vectorization techniques: one is TF-IDF another one is Word2vec and third one is COuntVectorizer (Keras Tokenizer). Model tends to perform better after we use the TF-IDF vectorization technique than the other one (Word2vec). Exception is observed incase of LSTM models.

## Reflections
- The major limitation of Logistic Regression is the assumption of linearity between the dependent variable and the independent variables.
- Both SVM and KNN are not suitable for large and high dimensional datasets.
- WRT LSTM, we would prefer using Bidirectional LSTM over a simple LSTM. However, given the training contains 960,000 rows, coupled with the lack of GPU resources, we have to settle with a single LSTM layer.
- The entire set of sparse representations is not considered in case of LSTM and is limited to 500. Ideally we would like a system that has the capability to run as many representations.