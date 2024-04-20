import os
import time
import torch
import pandas as pd
import numpy as np

from model import CCFCRec, train
from dataset import RatingDataset
from myargs import get_args, args_tostring
from metric import ValidateItems, ValidateUsers


if __name__ == "__main__":
    # args
    args = get_args()
    print(
        "progress start at:",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
    )

    # load data
    metadata = np.load(os.path.join(args.data_path, "metadata.npy"), allow_pickle=True).item()

    train_path = os.path.join(args.data_path, "train_all_warm_interactions_negative_users.npy")
    val_path = os.path.join(args.data_path, "val_cold_interactions.npy")
    test_path = os.path.join(args.data_path, "test_cold_interactions.npy")

    # negative user id
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    test_data = np.load(test_path)

    img_features = np.load(os.path.join(args.data_path, "v_features.npy")).astype(np.float32)
    movies_onehot = np.load(os.path.join(args.data_path, "onehot_features.npy")).astype(np.int32)
    category_map = np.load(os.path.join(args.data_path, "categories_id.npy"), allow_pickle=True).item()

    # load dataset
    dataSet = RatingDataset(
        train_data,
        img_features,
        movies_onehot,
        len(category_map),
        metadata["n_users"],
        metadata["n_items"],
        args.n_positive,
        args.n_negative,
    )

    train_loader = torch.utils.data.DataLoader(
        dataSet, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    print("Model hyperparameters:", args_tostring(args))

    args.n_users, args.n_items = metadata['n_users'], metadata['n_items']
    myModel = CCFCRec(args)

    optimizer = torch.optim.Adam(
        myModel.parameters(), lr=args.learning_rate, weight_decay=0.1
    )

    validators = {
        'items_validator_validation_set': ValidateItems(
            val_data,
            img_features,
            movies_onehot
        ),
        'users_validator_validation_set': ValidateUsers(
            val_data,
            img_features,
            movies_onehot,
            metadata['n_users'],
            metadata['n_val_cold_items']
        ),
        'items_validator_test_set': ValidateItems(
            test_data,
            img_features,
            movies_onehot
        ),
        'users_validator_test_set': ValidateUsers(
            test_data,
            img_features,
            movies_onehot,
            metadata['n_users'],
            metadata['n_test_cold_items']
        )
    }

    train(myModel, train_loader, optimizer, validators, args)
