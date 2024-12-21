#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np 
import pandas as pd


# In[40]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[41]:


movies.head(1)


# In[42]:


credits.head(1)


# In[43]:


movies = movies.merge(credits,on='title')


# In[44]:


movies.head(1)


# In[45]:


#genres
#id
#keywords
#title
#overview
#cast
#crew

movies = movies[['movie_id','genres','keywords','title','overview','cast','crew']]


# In[46]:


movies.info()


# In[47]:


movies.head()


# In[48]:


movies.isnull().sum()


# In[49]:


movies.dropna(inplace=True)


# In[50]:


movies.duplicated().sum()


# In[51]:


movies.iloc[0].genres


# In[57]:


#('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'


# In[59]:


def convert3(obj):
    L = []
    for i in ast.literal_eval(obj):
            L.append(i['name'])
    return L


# In[60]:


movies['genres'] = movies['genres'].apply(convert)


# In[61]:


movies.head()


# In[62]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[104]:


movies['cast'][0]


# In[63]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
                break
    return L


# In[64]:


movies['cast'] = movies['cast'].apply(convert3)


# In[65]:


movies.head()


# In[66]:


movies['crew'][0]


# In[67]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i ['job'] == 'Director':
            L.append(i['name'])
    return L


# In[69]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[70]:


movies.head()


# In[71]:


movies['overview'][0]


# In[74]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[75]:


movies.head()


# In[78]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[79]:


movies.head()


# In[80]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[81]:


movies.head()


# In[82]:


new_df = movies[['movie_id','title','tags']]


# In[85]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[86]:


new_df.head()


# In[96]:


import nltk


# In[97]:


get_ipython().system('pip install nltk')


# In[98]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[100]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)
    


# In[103]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[104]:


new_df['tags'][0]


# In[105]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[106]:


new_df.head()


# In[107]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[108]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[109]:


vectors


# In[110]:


vectors[0]


# In[111]:


cv.get_feature_names()


# In[ ]:


['loved','loving','love']
['love','love','love']


# In[99]:


ps.stem('loved')


# In[101]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[113]:


from sklearn.metrics.pairwise import cosine_similarity


# In[116]:


similarity = cosine_similarity(vectors)


# In[127]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[137]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
      
    


# In[1]:


recommend('Avatar')


# In[136]:


new_df.iloc[1216].title


# In[140]:


import pickle


# In[143]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))


# In[144]:


new_df.to_dict()


# In[145]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




