import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
pd.set_option("display.max.columns", None)
pd.set_option("display.precision", 2)
ratings = pd.read_csv('ratings_small.csv')
credit = pd.read_csv('credits.csv')
movies = pd.read_csv('movies_metadata.csv')

# print(ratings.head())
# print(credit.head())
# print(movies.head())

credit.rename(columns={'id': 'Id'}, inplace=True)

# print(ratings.head())
#
# print(movies.shape)
# print(credit.shape)
# print(ratings.shape)

MOVIE = pd.concat([movies, credit], axis=1)

# print(MOVIE.head())


MOVIE.drop(columns=['homepage', 'imdb_id', 'original_language', 'belongs_to_collection',
                    'budget', 'poster_path', 'production_companies',
                    'production_countries', 'release_date', 'revenue',
                    'runtime', 'spoken_languages','status', 'tagline', 'video', 'crew','Id','title','overview','adult','cast'], inplace=True)



# print(MOVIE.head())


# print(MOVIE.isnull().sum())
# print(MOVIE.shape)
MOVIE.dropna(inplace=True)
# print(MOVIE.isnull().sum())


v_c=MOVIE['vote_count']
v_a=MOVIE['vote_average']
C=v_a.mean()
m=v_c.quantile(0.7)

MOVIE['weighted_average']=((v_c*v_a) + (C*m))/(v_c+m)
W_A = MOVIE['weighted_average']
# print(MOVIE.head())

scaler=MinMaxScaler()

movie_scaled=scaler.fit_transform(MOVIE[['weighted_average', 'popularity']])

movie_norm=pd.DataFrame(movie_scaled,columns=['weighted_average', 'popularity'])

MOVIE[['norm_weight_score','norm_popularity']]=movie_norm



MOVIE['score'] = MOVIE['norm_weight_score'] * 0.5 + MOVIE['norm_popularity'] * 0.5

MOVIE_rank=MOVIE.sort_values('weighted_average',ascending=False)

top=MOVIE_rank.iloc[:,2].values

print(MOVIE_rank.head(15))
# i=10
#
# print(top[0:i])

MOVIE_rank.reset_index(inplace=True)
MOVIE_rank.drop(columns='index',inplace=True)
print(MOVIE_rank.head(15))
Genere = MOVIE_rank['genres'].values
#
print(Genere[4])
corpus=[]
for k in range(0,len(Genere)):
    review=re.sub('[^a-zA-Z ]','',Genere[k])
    corpus.append(review)

corpus2=[]
for kc in range(0,len(corpus)):
    word = 'id'
    word_list1 = corpus[kc].split()
    review1 = ' '.join([w for w in word_list1 if w not in word])
    word = 'name'
    word_list2 = review1.split()
    review2 = ' '.join([w for w in word_list2 if w not in word])
    corpus2.append(review2)

print(corpus2[4])


Genere1=pd.DataFrame(corpus2)

print(Genere1.head(15))

MOVIE_rank['genres'] = Genere1

print(MOVIE_rank.head(15))


Genre3=MOVIE_rank['genres'].values


Length=[]
for ck in range(0, len(Genre3)):
    str1 = Genre3[0]
    str2 = Genre3[ck]
    str1_words = set(str1.split())
    str2_words = set(str2.split())
    common = str1_words & str2_words
    ln=len(common)
    Length.append(int(ln))

# print(Length)

numberofgeners_matched=pd.DataFrame(Length)

MOVIE_rank['number_of_same_genres']=numberofgeners_matched

print(MOVIE_rank.head(15))

MOVIE_rank_1=MOVIE_rank.sort_values('number_of_same_genres',ascending=False)

MOVIE_rank_1.reset_index(inplace=True)

print(MOVIE_rank_1.head(15))





