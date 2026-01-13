import pandas as pd 
import numpy as np 
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

# Load the rating data into a DataFrame:
column_names = ['User_ID', 'User_Names','Movie_ID','Rating','Timestamp']
movies_df = pd.read_csv('/usercode/Movie_data.csv', sep = ',', names = column_names)

#Load the move information in a DataFrame:
movies_title_df = pd.read_csv("/usercode/Movie_Id_Titles.csv")
movies_title_df.rename(columns = {'item_id':'Movie_ID', 'title':'Movie_Title'}, inplace = True)

#Merge the DataFrames:
movies_df = pd.merge(movies_df,movies_title_df, on='Movie_ID')

#View the DataFrame:
print(movies_df.head())

print(f"\n Size of the movie_df dataset is {movies_df.shape}")

movies_df.describe()

movies_df.groupby('User_ID')['Rating'].count().sort_values(ascending = True).head()

n_users = movies_df.User_ID.unique().shape[0]
n_movies = movies_df.Movie_ID.unique().shape[0]
print( str(n_users) + ' users')
print( str(n_movies) + ' movies')

#This would be a 2D array matrix to display user-movie_rating relationship
#Rows represent users by IDs, columns represent movies by IDs
ratings = np.zeros((n_users, n_movies))
for row in movies_df.itertuples():
    ratings[row[1], row[3]-1] = row[4]

# View the matrix
print(ratings)

sparsity = 1.0 - (float(len(ratings.nonzero()[0])) / (ratings.shape[0] * ratings.shape[1]))
sparsity *= 100
print(sparsity)

rating_cosine_similarity = cosine_similarity(ratings)

def movie_recommender(user_item_m, X_user, user, k=10, top_n=10):
    # Get the location of the actual user in the User-Items matrix
    # Use it to index the User similarity matrix
    user_similarities = X_user[user]
    # obtain the indices of the top k most similar users
    most_similar_users = user_item_m.index[user_similarities.argpartition(-k)[-k:]]
    # Obtain the mean ratings of those users for all movies
    rec_movies = user_item_m.loc[most_similar_users].mean(0).sort_values(ascending=False)
    # Discard already seen movies
    m_seen_movies = user_item_m.loc[user].gt(0)
    seen_movies = m_seen_movies.index[m_seen_movies].tolist()
    rec_movies = rec_movies.drop(seen_movies).head(top_n)
    # return recommendations - top similar users rated movies
    rec_movies_a=rec_movies.index.to_frame().reset_index(drop=True)
    rec_movies_a.rename(columns={rec_movies_a.columns[0]: 'Movie_ID'}, inplace=True)
    return rec_movies_a

#Converting the 2D array into a DataFrame as expected by the movie_recommender function
ratings_df=pd.DataFrame(ratings)

user_ID=12
movie_recommender(ratings_df, rating_cosine_similarity,user_ID)

def movie_recommender_run(user_Name):
    #Get ID from Name
    user_ID=movies_df.loc[movies_df['User_Names'] == user_Name].User_ID.values[0]
    #Call the function
    temp=movie_recommender(ratings_df, rating_cosine_similarity, user_ID)
    # Join with the movie_title_df to get the movie titles
    top_k_rec=temp.merge(movies_title_df, how='inner')
    return top_k_rec

