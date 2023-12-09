**News Category Detection**

Social media platforms have come under unwanted scrutiny in recent years for their role in perpetuating online misinformation, so-called “fake news.” As of this writing in May 2020, a flood of conspiracy theories and falsehoods about the Covid-19 pandemic have become a problem for Facebook and Twitter (Cellan-Jones 2020). In this project, we discuss a variety of means that have been proposed for social media companies to attempt to automate fake news detection. We look at the great difficulty involved in obtaining labeled training data for supervised learning to train deep neural networks in this task, and experience this difficulty first-hand as we unsuccessfully try to train a model to detect misinformation using a public dataset. 

As a related surrogate task, we train a deep recurrent neural network to identify the topic of news articles across a wide range of subjects, achieving 75% test accuracy across 40 categories of news stories, and test how well it generalizes to new news articles that we scraped from the internet. We find that the model generalizes well even to stories that come from a different publication and time period. Finally, we discuss how such a model can serve as one facet of a technological arsenal in the fight against fake news.


## Table of Contents
* [Initial Project Plan](https://github.com/jwoh1323/News-Category-Detection-Project/blob/cb3c17d004c19a0d1ac17f172a4525200754fd6b/Final-Project-Plan.ipynb)
* [Building a model with Fake News Dataset](https://github.com/jwoh1323/News-Category-Detection-Project/blob/cb3c17d004c19a0d1ac17f172a4525200754fd6b/load-liar-data.ipynb)
* [Building a model with Liar Liar Dataset](https://github.com/jwoh1323/News-Category-Detection-Project/blob/cb3c17d004c19a0d1ac17f172a4525200754fd6b/load-liar-data.ipynb)
* [Exploratory Data Analysis for News Category Dataset](https://github.com/jwoh1323/News-Category-Detection-Project/blob/cb3c17d004c19a0d1ac17f172a4525200754fd6b/For_ML_Project_news_exploratory_analysis.ipynb)
* [Text Preprocessing and Apply to a Single Model](https://github.com/jwoh1323/News-Category-Detection-Project/blob/cb3c17d004c19a0d1ac17f172a4525200754fd6b/news-cat-data-lstm-multi.ipynb)
* [Exploration of Various Model Structures](https://github.com/jwoh1323/News-Category-Detection-Project/blob/cb3c17d004c19a0d1ac17f172a4525200754fd6b/news-cat-data-lstm-multi.ipynb)
* [Model Comparison](https://github.com/jwoh1323/News-Category-Detection-Project/blob/cb3c17d004c19a0d1ac17f172a4525200754fd6b/news-cat-data-alt-models.ipynb)
* [News Articles Scraping Process](https://github.com/jwoh1323/News-Category-Detection-Project/blob/cb3c17d004c19a0d1ac17f172a4525200754fd6b/Scraping.ipynb)
* [K Means Clustering with Scraped Data](https://github.com/jwoh1323/News-Category-Detection-Project/blob/cb3c17d004c19a0d1ac17f172a4525200754fd6b/K-means.ipynb)
* [Final Model and Prediction on Scraped Data](https://github.com/jwoh1323/News-Category-Detection-Project/blob/cb3c17d004c19a0d1ac17f172a4525200754fd6b/news-cat-final-model.ipynb)


## Dataset Info
The sources of our dataset are:

* [Fake News Dataset from Kaggle](https://www.kaggle.com/mrisdal/fake-news)
* [Liar Lair Dataset from William Wang](https://sites.cs.ucsb.edu/~william/software.html)
* [News Category Dataset from Kaggle](https://www.kaggle.com/rmisra/news-category-dataset)
* [NPR News Archive (Scraping)](https://www.npr.org/sections/news/archive)
	
## Technologies Used
Project is created with:

* [Google Colab](https://colab.research.google.com/notebooks/)
* [D2L](http://d2l.ai/index.html)
* [mxnet](https://mxnet.apache.org/)
* [Pandas](https://pandas.pydata.org/docs/index.html)
* [Numpy](https://numpy.org/)
* [nltk](https://www.nltk.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [BeatifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [Browser](https://docs.python.org/3/library/webbrowser.html)
* [MongoDB](https://www.mongodb.com/)
* [scikit-learn](https://scikit-learn.org/stable/)

	
## Reference

* [Tech Tent: Social media fights a fresh flood of fake news (BBC News)](https://www.bbc.com/news/technology-52245992)
* [The Fake Americans Russia Created to Influence the Election (NY Times)](https://www.nytimes.com/2017/09/07/us/politics/russia-facebook-twitter-election.html)
* [This Site Uses AI to Generate Fake News Articles](https://futurism.com/site-ai-generate-fake-news-articles)
* [Catching a Unicorn with GLTR: A tool to detect automatically generated text](http://glhttp://gltr.io/tr.io)
* [The Limitations of Stylometry for Detecting Machine-Generated Fake News](https://arxiv.org/abs/1908.09805)
* [Fake news detection on social media: A data mining perspective](https://dl.acm.org/doi/abs/10.1145/3137597.3137600)
* [Global vectors for word representation](https://nlp.stanford.edu/projects/glove/)
* [Web Scraping news articles in Python](https://towardsdatascience.com/web-scraping-news-articles-in-python-9dd605799558)
* [Easily Web Scrape and Summarize News Articles Using Python](https://towardsdatascience.com/easily-scrape-and-summarize-news-articles-using-python-dfc7667d9e74)
* [How to Determine the Optimal K for K-Means?](https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb)
