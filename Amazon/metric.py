import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, genres, image_feature, k):
    genres = genres.unsqueeze(dim=0)
    image_feature = image_feature.unsqueeze(dim=0)

    q_v_c = model(genres, image_feature, 1)
    user_emb = model.user_embedding

    ratings = torch.mul(user_emb, q_v_c).sum(dim=1)
    index = torch.argsort(-ratings)

    return (
        index[0:k].cpu().detach().numpy().tolist(),
        ratings.cpu().detach().numpy().tolist(),
    )


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
        ndcg = dcg_k(ratings) / ideal_dcg
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

            n_groundtruth = len(set(self.item_user_dict[it]))

            # Calculate hr indicator
            # Calculate p@k indicator
            hr_hit_cnt_5 += hr_at_k(it, recommend_users, self.item_user_dict, 5) / min(5, n_groundtruth) if n_groundtruth > 0 else 1
            hr_hit_cnt_10 += hr_at_k(it, recommend_users, self.item_user_dict, 10) / min(10, n_groundtruth) if n_groundtruth > 0 else 1
            hr_hit_cnt_20 += hr_at_k(it, recommend_users, self.item_user_dict, 20) / min(20, n_groundtruth) if n_groundtruth > 0 else 1

            # Calculate NDCG indicator
            ndcg_sum_5 += ndcg_k(it, recommend_users, self.item_user_dict, 5)
            ndcg_sum_10 += ndcg_k(it, recommend_users, self.item_user_dict, 10)
            ndcg_sum_20 += ndcg_k(it, recommend_users, self.item_user_dict, 20)

        item_len = len(self.item)
        hr_5 = hr_hit_cnt_5 / item_len
        hr_10 = hr_hit_cnt_10 / item_len
        hr_20 = hr_hit_cnt_20 / item_len
        ndcg_5 = ndcg_sum_5 / item_len
        ndcg_10 = ndcg_sum_10 / item_len
        ndcg_20 = ndcg_sum_20 / item_len
        print("hr@5:", "hr_10:", "hr_20:", "ndcg@5", "ndcg@10", "ndcg@20")
        print(hr_5, ",", hr_10, ",", hr_20, ",", ndcg_5, ",", ndcg_10, ",", ndcg_20)
        return hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20, None


class ValidateUsers:
    def __init__(self, val_data, img_features, onehot_features, n_users, n_items):
        print("validate class init")
        self.n_users, self.n_items = n_users, n_items
        self.user = set(val_data[:, 0])
        self.user_item_dict = {}
        self.items_id = {}
        for data in val_data:
            if data[1] not in self.items_id:
                self.items_id[data[1]] = len(self.items_id)

            if data[0] not in self.user_item_dict:
                self.user_item_dict[data[0]] = []
            self.user_item_dict[data[0]].append(self.items_id[data[1]])

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

            # rankings = np.zeros_like(ratings)
            # rankings[np.argsort(ratings)] = np.arange(len(ratings))
            all_ratings[:, self.items_id[it]] = np.zeros_like(ratings)
            all_ratings[np.argsort(ratings), self.items_id[it]] = np.arange(len(ratings))

        for u in self.user:
            recommend_items = np.argsort(-all_ratings[u])
            n_groundtruth = len(set(self.user_item_dict[u]))

            # Calculate hr indicator
            # Calculate p@k indicator
            hr_hit_cnt_5 += hr_at_k(u, recommend_items, self.user_item_dict, 5) / min(5, n_groundtruth) if n_groundtruth > 0 else 1
            hr_hit_cnt_10 += hr_at_k(u, recommend_items, self.user_item_dict, 10) / min(10, n_groundtruth) if n_groundtruth > 0 else 1
            hr_hit_cnt_20 += hr_at_k(u, recommend_items, self.user_item_dict, 20) / min(20, n_groundtruth) if n_groundtruth > 0 else 1

            # Calculate NDCG indicator
            ndcg_sum_5 += ndcg_k(u, recommend_items, self.user_item_dict, 5)
            ndcg_sum_10 += ndcg_k(u, recommend_items, self.user_item_dict, 10)
            ndcg_sum_20 += ndcg_k(u, recommend_items, self.user_item_dict, 20)

        item_len = len(self.user)
        hr_5 = hr_hit_cnt_5 / item_len
        hr_10 = hr_hit_cnt_10 / item_len
        hr_20 = hr_hit_cnt_20 / item_len
        ndcg_5 = ndcg_sum_5 / item_len
        ndcg_10 = ndcg_sum_10 / item_len
        ndcg_20 = ndcg_sum_20 / item_len
        print("hr@5:", "hr_10:", "hr_20:", "ndcg@5", "ndcg@10", "ndcg@20")
        print(hr_5, ",", hr_10, ",", hr_20, ",", ndcg_5, ",", ndcg_10, ",", ndcg_20)
        return hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20, all_ratings
