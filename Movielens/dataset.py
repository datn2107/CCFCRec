import numpy as np
import torch
from torch.utils.data import Dataset


def load_postive_negative_items_each_user(train_data, warm_items):
    positive_items_for_user = {}
    for data in train_data:
        if data[0] not in positive_items_for_user:
            positive_items_for_user[data[0]] = []
        positive_items_for_user[data[0]].append(data[1])

    negative_items_for_user = {}
    for user in positive_items_for_user:
        negative_items_for_user[user] = list(warm_items - set(positive_items_for_user[user]))

    return positive_items_for_user, negative_items_for_user


class RatingDataset(Dataset):
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
