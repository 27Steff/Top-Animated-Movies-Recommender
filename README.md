## Description

This is an exploratory analysis of an animated movies database, where I delve into the fascinating world of animated cinema to uncover valuable insights and provide personalized movie recommendations using machine learning.

As a fan of animated movies, I've embarked on a journey to analyze a rich dataset containing information on animated films, including details such as runtime, directors, studios, and more. Through data analysis, I aim to uncover trends, correlations, and intriguing patterns that shed light on the dynamic landscape of animated filmmaking.

From examining the impact of runtime on audience reception to identifying the most prolific and influential directors and studios in the animation industry, my analysis explores various facets of animated movie production and reception. By visualizing data trends and relationships using powerful tools like matplotlib, seaborn, and plotly, I paint a comprehensive picture of the animated movie landscape.

Outisde of this, using **machine learning, I've developed a recommendation system. Using TF-IDF vectorization and cosine similarity**, my recommendation system analyzes the content of animated movies and identifies similarities to provide tailored recommendations based on user preferences.

Moreover, **I utilized neural networks with TensorFlow to classify movies by their certificate** (e.g., G, PG, PG-13) while employing methods to avoid overfitting. Additionally, I utilized one-hot coding for the certification, ensuring the model's effectiveness in predicting movie certificates accurately.


Some of the tasks completed on this project were:

### Data Loading and Inspection:

Loads a dataset named 'TopAnimatedImDb.csv' using pandas.
Checks the information about the dataset.
Checks for missing values (NaN) in the dataset.

### Data Preprocessing:
Cleans and converts columns like 'Runtime', 'Votes', and 'Gross' to appropriate data types for analysis.
Calculates and prints the average duration of the top animated movies.

### Exploratory Data Analysis:
Explores categorical variables like 'Certificate' and 'Genre' using seaborn pairplots and countplots.
Creates visualizations (bar plots and scatter plots) to analyze relationships between variables such as rating vs. votes, rating vs. gross earnings, director vs. rating, director vs. genre, and certificate vs. count.

### Time Series Analysis:
Calculates the number of movies released each year and plots it over time.

### Director Analysis:
Analyzes the top 10 directors with the highest count of movies.

### Recommendation System:
Implements a content-based recommendation system using TF-IDF vectorization and cosine similarity to recommend movies similar to a given movie ('Klaus' in this case).

### Neural Network Implementation 
TensorFlow was used to classify movies by their certificate while employing methods to avoid overfitting. We used tokenization methods and one hot coding to avoid unwanted relationships in the data.
