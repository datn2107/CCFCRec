import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from myargs import get_args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model, genres, img, k):
    user_embedding = model.user_embedding
    user_idx = torch.tensor(list(range(user_embedding.shape[0])))
    user_idx = user_idx.to(device)

    # [138493*64]
    user_emb = user_embedding[user_idx]
    genres = genres.unsqueeze(dim=0)
    img = img.unsqueeze(dim=0)

    # input into the model
    attr_present = model.attr_matrix(genres)
    attr_tmp1 = model.h(torch.matmul(attr_present, model.attr_W1.T) + model.attr_b1)
    attr_attention_b = model.softmax(torch.matmul(attr_tmp1, model.attr_W2))

    # z_v is the vector of attributes after attention-weighted fusion.
    z_v = torch.matmul(attr_attention_b.transpose(1, 2), attr_present).squeeze()
    # Image embedding vector of item
    p_v = torch.matmul(img, model.image_projection)
    q_v_a = torch.cat((z_v.unsqueeze(dim=0), p_v), dim=1)
    q_v_c = model.gen_layer2(model.h(model.gen_layer1(q_v_a)))
    ratings = torch.mul(user_emb, q_v_c).sum(dim=1)
    index = torch.argsort(-ratings)

    return index[0:k].cpu().detach().numpy().tolist(), ratings.cpu().detach().numpy().tolist()


def hr_at_k(idx, recommend, grouthtruths, k):
    grouthtruth = set(grouthtruths.get(idx))
    recommend = set(recommend[0:k])
    inter = grouthtruth.intersection(recommend)
    return len(inter)


def dcg_k(r):
    r = np.asarray(r)
    val = np.sum((np.power(2, r) - 1) / (np.log2(np.arange(1 + 1, r.size + 2))))
    return val


def ndcg_k(idx, recommend, grouthtruths, k):
    grouthtruth = set(grouthtruths.get(idx))
    recommend = recommend[0:k]
    ratings = []
    ndcg = 0.0
    for u in recommend:
        if u in grouthtruth:
            ratings.append(1.0)
        else:
            ratings.append(0.0)
    ratings_ideal = sorted(ratings, reverse=True)
    ideal_dcg = dcg_k(ratings_ideal)
    if ideal_dcg != 0:
        ndcg = (dcg_k(ratings) / ideal_dcg)
    return ndcg


class ValidateItems:
    def __init__(self, val_data, img_features, onehot_features):
        print("validateItem class init")
        self.item = set(val_data[:, 1])
        self.item_user_dict = {}
        for data in val_data:
            if data[1] not in self.item_user_dict:
                self.item_user_dict[data[1]] = []
            self.item_user_dict[data[1]].append(data[0])

        self.img_features = img_features
        self.onehot_features = onehot_features

    def start_validate(self, model):
        # Start assessment
        hr_hit_cnt_5, hr_hit_cnt_10, hr_hit_cnt_20 = 0, 0, 0
        ndcg_sum_5, ndcg_sum_10, ndcg_sum_20 = 0.0, 0.0, 0.0
        max_k = 20

        for it in self.item:
            # output
            model = model.to(device)  # move to cpu
            genre = torch.tensor(self.onehot_features[it])
            img_feature = torch.tensor(self.img_features[it])
            genre = genre.to(device)
            img_feature = img_feature.to(device)
            with torch.no_grad():
                recommend_users, _ = predict(model, genre, img_feature, max_k)

            # Calculate hr indicator
            # Calculate p@k indicator
            hr_hit_cnt_5 += hr_at_k(it, recommend_users, self.item_user_dict, 5)
            hr_hit_cnt_10 += hr_at_k(it, recommend_users, self.item_user_dict, 10)
            hr_hit_cnt_20 += hr_at_k(it, recommend_users, self.item_user_dict, 20)

            # Calculate NDCG indicator
            ndcg_sum_5 += ndcg_k(it, recommend_users, self.item_user_dict, 5)
            ndcg_sum_10 += ndcg_k(it, recommend_users, self.item_user_dict, 10)
            ndcg_sum_20 += ndcg_k(it, recommend_users, self.item_user_dict, 20)

        item_len = len(self.item)
        hr_5 = hr_hit_cnt_5 / (item_len * 5)
        hr_10 = hr_hit_cnt_10 / (item_len * 10)
        hr_20 = hr_hit_cnt_20 / (item_len * 20)
        ndcg_5 = ndcg_sum_5 / item_len
        ndcg_10 = ndcg_sum_10 / item_len
        ndcg_20 = ndcg_sum_20 / item_len
        print("hr@5:", "hr_10:", "hr_20:", 'ndcg@5', 'ndcg@10', 'ndcg@20')
        print(hr_5, ',', hr_10, ',', hr_20, ',', ndcg_5, ',', ndcg_10, ',', ndcg_20)
        return hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20


class ValidateUsers:
    def __init__(self, val_data, img_features, onehot_features, n_users, n_items):
        print("validate class init")
        self.n_users, self.n_items = n_users, n_items
        self.user = set(val_data[:, 0])
        self.user_item_dict = {}
        self.items_id = {}
        for data in val_data:
            if data[0] not in self.user_item_dict:
                self.user_item_dict[data[0]] = []
            self.user_item_dict[data[0]].append(data[1])

            if data[1] not in self.items_id:
                self.items_id[data[1]] = len(self.items_id)

        self.img_features = img_features
        self.onehot_features = onehot_features

    def start_validate(self, model):
        # Start assessment
        hr_hit_cnt_5, hr_hit_cnt_10, hr_hit_cnt_20 = 0, 0, 0
        ndcg_sum_5, ndcg_sum_10, ndcg_sum_20 = 0.0, 0.0, 0.0
        max_k = 20
        all_ratings = np.zeros((self.n_users, self.n_items))

        for it in self.items_id.keys():
            # output
            model = model.to(device)  # move to cpu
            genre = torch.tensor(self.onehot_features[it])
            img_feature = torch.tensor(self.img_features[it])
            genre = genre.to(device)
            img_feature = img_feature.to(device)
            with torch.no_grad():
                _, ratings = predict(model, genre, img_feature, max_k)

            rankings = np.unique(ratings, return_inverse=True)[1]
            all_ratings[:, self.items_id[it]] = rankings

        for u in self.user:
            recommend_items = np.argsort(-all_ratings[u])

            # Calculate hr indicator
            # Calculate p@k indicator
            hr_hit_cnt_5 += hr_at_k(u, recommend_items, self.user_item_dict, 5)
            hr_hit_cnt_10 += hr_at_k(u, recommend_items, self.user_item_dict, 10)
            hr_hit_cnt_20 += hr_at_k(u, recommend_items, self.user_item_dict, 20)

            # Calculate NDCG indicator
            ndcg_sum_5 += ndcg_k(u, recommend_items, self.user_item_dict, 5)
            ndcg_sum_10 += ndcg_k(u, recommend_items, self.user_item_dict, 10)
            ndcg_sum_20 += ndcg_k(u, recommend_items, self.user_item_dict, 20)

        item_len = len(self.user)
        hr_5 = hr_hit_cnt_5 / (item_len * 5)
        hr_10 = hr_hit_cnt_10 / (item_len * 10)
        hr_20 = hr_hit_cnt_20 / (item_len * 20)
        ndcg_5 = ndcg_sum_5 / item_len
        ndcg_10 = ndcg_sum_10 / item_len
        ndcg_20 = ndcg_sum_20 / item_len
        print("hr@5:", "hr_10:", "hr_20:", 'ndcg@5', 'ndcg@10', 'ndcg@20')
        print(hr_5, ',', hr_10, ',', hr_20, ',', ndcg_5, ',', ndcg_10, ',', ndcg_20)
        return hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20
