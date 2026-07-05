# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 15:11:10 2026

@author: LaptopIran
"""

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import joblib

rating = pd.read_csv("D:/ml-latest-small/ratings.csv")
movies = pd.read_csv("D:/ml-latest-small/movies.csv")
tags = pd.read_csv("D:/ml-latest-small/tags.csv")
links = pd.read_csv("D:/ml-latest-small/links.csv")

movies.head()
tags.head()
links.head()

print(rating.shape)
print(movies.shape)
print(tags.shape)

print("تعداد کاربران:", rating["userId"].nunique())
print("تعداد فیلم‌های امتیازدهی‌شده:", rating["movieId"].nunique())
print("تعداد کل امتیازها:", len(rating))

rating.describe()

#EDA

ratings = rating.sort_values("timestamp")

train_ratings = []
test_ratings = []

for user_id, group in ratings.groupby("userId"):
    
    if len(group) < 5:
        train_ratings.append(group)
        continue
    
    test_part = group.tail(1)
    train_part = group.iloc[:-1]
    
    train_ratings.append(train_part)
    test_ratings.append(test_part)

train_ratings = pd.concat(train_ratings).reset_index(drop=True)
test_ratings = pd.concat(test_ratings).reset_index(drop=True)

print(train_ratings.shape)
print(test_ratings.shape)

RELEVANT_THRESHOLD = 4.0

test_relevant = test_ratings[
    test_ratings["rating"] >= RELEVANT_THRESHOLD
].copy()

test_relevant.head()

def ndcg_at_k(recommended_items, relevant_items, k=10):
    recommended_items = recommended_items[:k]
    
    dcg = 0.0
    
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            dcg += 1 / np.log2(i + 2)
    
    ideal_hits = min(len(relevant_items), k)
    
    idcg = sum(
        1 / np.log2(i + 2)
        for i in range(ideal_hits)
    )
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def precision_at_k(recommended_items, relevant_items, k=10):
    recommended_items = recommended_items[:k]
    
    if len(recommended_items) == 0:
        return 0.0
    
    hits = len(set(recommended_items) & set(relevant_items))
    
    return hits / k

def recall_at_k(recommended_items, relevant_items, k=10):
    recommended_items = recommended_items[:k]
    
    if len(relevant_items) == 0:
        return 0.0
    
    hits = len(set(recommended_items) & set(relevant_items))
    
    return hits / len(relevant_items)

movie_stats = (
    train_ratings
    .groupby("movieId")
    .agg(
        mean_rating=("rating", "mean"),
        rating_count=("rating", "count")
    )
    .reset_index()
)
"""  agg مخفف Aggregate است. یعنی:
برای هر گروه، یک یا چند آمار محاسبه کن.
فرض کن گروه فیلم ۱۰ این باشد:
    
rating
5
4
3
حالا می‌توانیم بگوییم:
میانگین را حساب کن.
تعداد را حساب کن.
بیشینه را حساب کن.
کمینه را حساب کن.
و ...
همه‌ی این‌ها عملیات تجمیعی (Aggregation) هستند.
"""

movie_stats.head()



C = movie_stats["mean_rating"].mean()
m = movie_stats["rating_count"].quantile(0.80)

movie_stats["weighted_score"] = (
    (movie_stats["rating_count"] / (movie_stats["rating_count"] + m))
    * movie_stats["mean_rating"]
    +
    (m / (movie_stats["rating_count"] + m))
    * C
)

popular_movies = (
    movie_stats
    .sort_values("weighted_score", ascending=False)
    .merge(movies, on="movieId")
)

popular_movies[
    ["movieId", "title", "genres", "mean_rating", "rating_count", "weighted_score"]
].head(10)


def recommend_popular(user_id, k=10):
    
    watched_movies = set(
        train_ratings[
            train_ratings["userId"] == user_id
        ]["movieId"]
    )
    
    recommendations = popular_movies[
        ~popular_movies["movieId"].isin(watched_movies)
    ].head(k)
    
    return recommendations[
        ["movieId", "title", "genres", "weighted_score"]
    ]



# as example
recommend_popular(user_id=1, k=10)
#or
recommend_popular(user_id=5)













