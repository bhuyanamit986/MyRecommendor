from flask import Flask
from flask import render_template
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# define a function that creates similarity matrix
# if it doesn't exist
def create_sim():
    data = pd.read_csv('data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    return data,sim

# defining a function that recommends 10 most similar movies
def rcmd(m):
    m = m.lower()
    # check if data and sim are already assigned
    try:
        data.head()
        sim.shape
    except:
        data, sim = create_sim()
    # check if the movie is in our database or not
    if m not in data['movie_title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:
        # getting the index of the movie in the dataframe
        i = data.loc[data['movie_title']==m].index[0]

        # fetching the row containing similarity scores of the movie
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[i]))

        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)

        # taking top 1- movie scores
        # not taking the first index since it is the same movie
        lst = lst[1:11]

        # making an empty list that will containg all 10 movie recommendations
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

PEOPLE_FOLDER=os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/')
def home():
    text = { 'content': 'Welcome to your flask application !' } 
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'yin-yang-symbol.jpg')
    return render_template("home.html",
        title = 'Home',
        text = text,
        user_image = full_filename
        )

app.config['UPLOAD_FOLDER1'] = PEOPLE_FOLDER
app.config['UPLOAD_FOLDER2'] = PEOPLE_FOLDER


@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = rcmd(movie)
    movie = movie.upper()
    full_filename1 = os.path.join(app.config['UPLOAD_FOLDER1'], 'linkedin.png')
    full_filename2 = os.path.join(app.config['UPLOAD_FOLDER2'], 'github.jpeg')
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s', user_image= [full_filename1, full_filename2])
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l', user_image= [full_filename1, full_filename2])


app.debug = False
if __name__ == "__main__":
    app.run()
