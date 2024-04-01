import os
import time
import torch
import pandas as pd
import numpy as np

from .model import CCFCRec
from .dataset import RatingDataset
from .dataset import read_img_feature, read_genres, serialize_user
from .myargs import get_args, args_tostring
from .metric import Validate


if __name__ == "__main__":
    args = get_args()

    print(
        "progress start at:",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
    )

    metadata = np.load(os.path.join(args.data_path, "metadata.npy"), allow_pickle=True).item()

    train_path = os.path.join(args.data_path, "train_all_warm_interactions_negative_users.npy")
    val_path = os.path.join(args.data_path, "val_cold_interactions.npy")

    # negative user id
    train_data = np.load(train_path, allow_pickle=True).item()

    cold_items = np.load(os.path.join(args.data_path, "cold_items.npy"), allow_pickle=True).items()
    warm_items = np.load(os.path.join(args.data_path, "warm_items.npy"), allow_pickle=True).items()
    img_features = np.load(os.path.join(args.data_path, "v_feature.npy"))
    movies_onehot = np.load(os.path.join(args.data_path, "onehot_feature.npy"))

    positive_items_for_user = {}
    for data in train_data:
        if data[0] not in positive_items_for_user:
            positive_items_for_user[data[0]] = []
        positive_items_for_user[data[0]].append(data[1])

    negative_items_for_user = {}
    for user in positive_items_for_user:
        negative_items_for_user[user] = list(warm_items - set(positive_items_for_user[user]))


    # load dataset
    dataSet = RatingDataset(
        train_data,
        img_features,
        movies_onehot,
        positive_items_for_user,
        negative_items_for_user,
        metadata['n_users'],
        metadata['n_items'],
        args.n_positive,
        args.n_negative,
    )

    train_loader = torch.utils.data.DataLoader(
        dataSet, batch_size=args.batch_size, shuffle=True, num_workers=0
    )gi
    print("Model hyperparameters:", args_tostring(args))

    myModel = CCFCRec(metadata['n_users'], metadata['n_items'], args)

    optimizer = torch.optim.Adam(
        myModel.parameters(), lr=args.learning_rate, weight_decay=0.1
    )
    validator = Validate(
        validate_csv=val_path,
        user_serialize_dict=user_serialize_dict,
        img=img_feature,
        genres=movies_onehot,
    )

    train(myModel, train_loader, optimizer, validator, args)
