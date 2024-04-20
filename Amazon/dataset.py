import numpy as np
import torch
from torch.utils.data import Dataset


# 新建一个user-item的交互字典
def build_interaction_dict(train_data):
    users_items_interaction = {}
    items_users_interaction = {}

    for interaction in train_data:
        user = interaction[0]
        item = interaction[1]
        if user not in users_items_interaction:
            users_items_interaction[user] = set()
        if item not in items_users_interaction:
            items_users_interaction[item] = set()
        users_items_interaction[user].add(item)
        items_users_interaction[item].add(user)

    return users_items_interaction, items_users_interaction


class RatingDataset(torch.utils.data.Dataset):
    def __init__(self, train_data, img_features, genres, category_num, n_users, n_items, n_positive, n_negative):
        self.train_data = train_data

        # 读其他内容
        self.img_feature_dict = img_features
        self.genres_dict = genres

        # print(self.item_pn_df)
        self.user = self.train_data[:, 0]
        self.item = self.train_data[:, 1]
        self.neg_user = self.train_data[:, 2]

        self.item_set = set(self.item)

        # 返回个数时，返回全集的user数和训练集的item数
        self.n_users = n_users
        self.n_items = n_items
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.category_num = category_num
        self.user_item_interaction_dict, self.item_user_interaction_dict = build_interaction_dict()
        print("The number of users in the entire data set is:", self.n_users, "The number of users in train_set is:", len(set(self.user)))

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, index):
        user = self.user[index]
        item = self.item[index]

        # process item genres
        genres = self.genres_dict.get(item)
        genres[genres == 0] = -1
        genres = genres.squeeze(dim=1)

        # process item feature
        img_feature = self.img_feature_dict.get(item)

        # neg_user = sample_negative_user(self.user, interaction_user_set)
        neg_user = self.neg_user[index]

        # print('sample neg user:', time.time()-get_item_start)
        # ------------------------ #
        #  process positive items  #
        #  runtime sampling        #
        # ------------------------ #
        positive_items_ = self.user_item_interaction_dict.get(user)
        positive_items = list(np.random.choice(list(positive_items_), self.n_positive, replace=True))

        # runtime sampling negative
        negative_item_list = []
        neg_item_set = list(self.item_set - set(positive_items_))

        # merge multi negative sample result
        negative_items_ = list(np.random.choice(neg_item_set, self.n_negative*(self.n_positive+1), replace=True))
        for i in range(self.n_positive):
            start_idx = self.n_negative*i
            end_idx = self.n_negative*(i+1)
            negative_item_list.append(negative_items_[start_idx:end_idx])

        # self neg list 完成 序列化, self的抽样放在和collaborative items中一起抽样负例子，最后分割出来就行了
        self_neg_list = negative_items_[self.n_positive*self.n_negative:]

        return torch.tensor(user), torch.tensor(item), torch.tensor(genres), torch.tensor(img_feature), \
               torch.tensor(neg_user), torch.tensor(positive_items), torch.tensor(negative_item_list), \
               torch.tensor(self_neg_list)
