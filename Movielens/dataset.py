import gc
import pickle
import random
import os
import sys
import time
import numpy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from tqdm import tqdm


# Return dict， userId: [positive_sample(list), negative_sample(list(list))]
def read_user_positive_negative_movies(user_positive_movie_csv, refresh=False):
    pkl_name = 'pkl/user_pn_dict.pkl'
    if os.path.exists("pkl") is False:
        os.makedirs("pkl")
    if (os.path.exists(pkl_name) is True) and (refresh is False):
        pkl_file = open(pkl_name, 'rb')
        data = pickle.load(pkl_file)
        return data['user_pn_dict']
    user_position_dict = {}
    last_user = -1
    user_position_dict[last_user] = [-1, -1]
    for index, row in tqdm(user_positive_movie_csv.iterrows()):
        u = row['userId']
        if u != last_user:
            user_position_dict[u] = [index, index]
            user_position_dict[last_user] = [user_position_dict.get(last_user)[0], index-1]
            last_user = u
    # update last item
    user_position_dict[last_user] = [user_position_dict.get(last_user)[0], user_positive_movie_csv.__len__()-1]
    with open(pkl_name, 'wb') as file:
        pickle.dump({'user_pn_dict': user_position_dict}, file)
    return user_position_dict


def read_img_feature(img_feature_csv):
    df = pd.read_csv(img_feature_csv, dtype={'feature': object, 'movie_id': int})
    img_feature_dict = {}
    for index, row in df.iterrows():
        item = row['movie_id']
        feature = list(map(float, row['feature'][1:-1].split(",")))
        img_feature_dict[item] = feature
    return img_feature_dict


def read_genres(genres_csv):
    df = pd.read_csv(genres_csv, dtype={'movieId': int})
    genres_dict = {}
    for index, row in df.iterrows():
        item = row['movieId']
        genres = list(map(int, row['genres_onehot'][1:-1].split(',')))
        genres_dict[item] = genres
    return genres_dict


def serialize_user(user_set):
    user_set = set(user_set)
    user_idx = 0
    # key: user original id，value: user ordered id
    user_serialize_dict = {}
    for user in user_set:
        user_serialize_dict[user] = user_idx
        user_idx += 1
    return user_serialize_dict


# 输入user和item的set，输出user和item从1到n有序的字典
def serialize_item(item_set):
    item_set = set(item_set)
    item_idx = 0
    item_serialize_dict = {}
    for item in item_set:
        item_serialize_dict[item] = item_idx
        item_idx += 1
    return item_serialize_dict


# def


class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, train_data, img_features, movies_onehot, positive_items_for_user, negative_items_for_user,
                 n_users, n_items, n_positive, n_negative):
        self.train_data = train_data

        # Read other content
        self.img_features = img_features
        self.movies_onehot = movies_onehot
        self.negative_items_for_user = negative_items_for_user
        self.positive_items_for_user = positive_items_for_user

        self.user = self.train_data[:, 0]
        self.item = self.train_data[:, 1]
        self.neg_user = self.train_data[:, 2]

        # set the number of users in the entire set and the number of items in the training set.
        self.n_users = n_users
        self.n_items = n_items
        self.n_positive = n_positive
        self.n_negative = n_negative
        print("The number of users is:", self.n_users)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        user = self.user[index]
        item = self.item[index]
        neg_user = self.neg_user[index]

        # Deal with item genres
        genres = self.movies_onehot[item]

        # Deal with item feature
        img_feature = self.img_features[item]

        # Deal with positive items
        positive_movies = np.random.choice(self.positive_items_for_user[user], self.n_positive, replace=True)

        # Get the negative items needed for the user
        total_negative_movies = np.random.choice(self.negative_items_for_user[user], (self.n_positive + 1) * self.n_negative, replace=True)

        # Deal with negative items
        negative_movies = np.zeros((self.n_positive, self.n_negative), dtype=int)
        for i in range(self.n_positive):
            negative_movies[i] = total_negative_movies[i*self.n_negative:(i+1)*self.n_negative]

        # Deal with self negative items
        self_negative_movies = total_negative_movies[self.n_positive*self.n_negative:]

        return torch.tensor(user), torch.tensor(item), torch.tensor(genres), torch.tensor(img_feature), \
               torch.tensor(neg_user), torch.tensor(positive_movies), torch.tensor(negative_movies), \
               torch.tensor(self_negative_movies)
