import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv('data/TopAnimatedImDb.csv')

df.info()#Reviewing the type of each column

df.isna().values.any()#Review Nan

print(df.isnull().any())#Finding Nan

#Preprocessing Data

df.Runtime
#Cleaning Runtime to compare
df.Runtime = df.Runtime.astype(str).str.replace(' min', '')
df.Runtime = pd.to_numeric(df.Runtime)
df.Runtime

df.Votes = df.Votes.astype(str).str.replace(',', '')
df.Gross = df.Gross.astype(str).str.replace('$', '')
df.Gross = df.Gross.astype(str).str.replace('M', '')

df.Votes = pd.to_numeric(df.Votes)
df.Gross =df.Gross.astype(float)

print(f'The average duration of the Top animated movies is: {df.Runtime.mean()} min')

df.Certificate.unique()#Finding categories

categories = df.Certificate.value_counts()
categories

sns.pairplot(data=df, hue = "Certificate")

rvotes = px.bar(df, x ='Rating', y ='Votes', color= 'Rating', hover_data='Title')
rvotes.show()

rgross = px.bar(df, x = 'Rating', y = 'Gross', hover_data=["Title"], color = 'Rating')
rgross.show()

genresn = df.Genre.value_counts()
genresn

sns.pairplot(data=df, hue = "Genre")



fig = px.scatter(df, x="Director", y="Rating",
                 color="Director",
                 hover_data=['Title'], size ='Year', width=1000, height=600)

fig.update(layout_showlegend=False)

fig.show()


fig = px.scatter(df, x="Director", y="Genre",
                 color="Director",
                 hover_data=['Title'], size ='Year', width=1000, height=600)

fig.update(layout_showlegend=False)

fig.show()



chart = sns.countplot(x=df["Certificate"])
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)


chart = sns.countplot(x=df["Genre"])
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)


moviesyear = df.groupby('Year')['Title'].count()

# Create the line plot
sns.set(style="darkgrid")
plt.plot(moviesyear.index[:-1], moviesyear.values[:-1])
plt.xlabel('Year')
plt.ylabel('Number of movies released')
plt.title('Number of movies released each year')
plt.show()


directorCount = df['Director'].value_counts()
display(directorCount.head(10))


fig = px.scatter(df, x="Director", y="Votes",
                 color="Director",
                 hover_data=['Title'], size ='Year', width=1000, height=600)

fig.update(layout_showlegend=False)

fig.show()


#Recommendation System

#Remove stop words
tfidf = TfidfVectorizer(stop_words='english')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['Description'])

# Calculate the cosine similarity matrix, cosineSim is how similar and the tfidf is to know how relevant they are
cosineSim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['Title'])


def getRecommendations(title, cosine_sim=cosineSim):
    indice = indices[title]

    # Get the similarity scores of all movies with that movie
    simScores = list(enumerate(cosine_sim[indice]))

    # Sort the movies based on the similarity scores
    simScores = sorted(simScores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top 10 similar movies
    simScores = simScores[1:11]

    # Get the indices
    movieIndices = [i[0] for i in simScores]

    # Return the top 10 most similar movies
    return df['Title'].iloc[movieIndices]

getRecommendations('Klaus')

#df.budget = pd.to_numeric(df.budget, errors='coerce')
#df.box_office = pd.to_numeric(df.box_office, errors='coerce')#Replacing the Not Available with Nan

#genres_list = df.genre.str.split(",", expand = True)#Split the genres and creating a column for each
#genres_list = df['genre'].value_counts()#Cuantas peliculas hay por genero
#genres_list.unique()