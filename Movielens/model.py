import os
import math
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


from myargs import args_tostring


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CCFCRec
class CCFCRec(nn.Module):
    def __init__(self, args):
        super(CCFCRec, self).__init__()

        # The last dimension in the attribute embedding table uses 0 as padding. If an attribute of an item is missing, fill it with 0 at the corresponding position. The corresponding index is 18
        self.args = args
        self.attr_matrix = nn.Embedding(
            args.attr_num + 1, args.attr_present_dim, padding_idx=args.attr_num
        )

        # Define attribute attention layer
        self.attr_W1 = torch.nn.Parameter(
            torch.FloatTensor(args.attr_present_dim, args.attr_present_dim)
        )
        self.attr_b1 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim))

        # Control the activation function of the entire model
        self.h = nn.LeakyReLU()

        # self.h = nn.ReLU()
        self.attr_W2 = torch.nn.Parameter(torch.FloatTensor(args.attr_present_dim, 1))
        self.softmax = torch.nn.Softmax(dim=0)

        # image mapping matrix
        self.image_projection = torch.nn.Parameter(
            torch.FloatTensor(4096, args.implicit_dim)
        )
        self.sigmoid = torch.nn.Sigmoid()

        # The embedding layer of user and item can be initialized with pre-trained ones.
        self.user_embedding = nn.Parameter(
            torch.FloatTensor(args.n_users, args.implicit_dim)
        )
        self.item_embedding = nn.Parameter(
            torch.FloatTensor(args.n_items, args.implicit_dim)
        )

        # Define the generation layer to jointly generate q_v_c from the information of (q_v_a, u),
        # and generate item embeddings containing collaborative information.
        self.gen_layer1 = nn.Linear(args.attr_present_dim * 2, args.cat_implicit_dim)
        self.gen_layer2 = nn.Linear(args.attr_present_dim, args.attr_present_dim)

        # Parameter initialization
        self.__init_param__()

    def __init_param__(self):
        nn.init.xavier_normal_(self.attr_W1)
        nn.init.xavier_normal_(self.attr_W2)
        nn.init.xavier_normal_(self.image_projection)

        # Generate layer initialization
        # Initialization of user, item embedding layer, initialization without pre-training
        if self.args.pretrain is None:
            nn.init.xavier_normal_(self.user_embedding)
            nn.init.xavier_normal_(self.item_embedding)
        nn.init.xavier_normal_(self.gen_layer1.weight)
        nn.init.xavier_normal_(self.gen_layer2.weight)

    def forward(self, attribute, image_feature, user_id):
        attr_present = self.attr_matrix(attribute)
        attr_tmp1 = self.h(torch.matmul(attr_present, self.attr_W1.T) + self.attr_b1)
        attr_attention_b = self.softmax(torch.matmul(attr_tmp1, self.attr_W2))
        z_v = torch.matmul(
            attr_attention_b.transpose(1, 2), attr_present
        ).squeeze()  # z_v is the vector of attributes after attention-weighted fusion.

        p_v = torch.matmul(
            image_feature, self.image_projection
        )  # image embedding vector of item
        q_v_a = torch.cat((z_v, p_v), dim=1)
        q_v_c = self.gen_layer2(self.h(self.gen_layer1(q_v_a)))
        return q_v_c


def train(model, train_loader, optimizer, validators, args):
    print("Model start train!")
    model_save_dir = os.path.join(args.save_path)
    os.makedirs(model_save_dir, exist_ok=True)

    print(
        "Model train at:",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
    )
    # Write hyperparameters
    with open(model_save_dir + "/readme.txt", "a+") as f:
        str_ = args_tostring(args)
        f.write(str_)
        f.write("\nSave dir:" + model_save_dir)
        f.write(
            "\nModel train time:"
            + (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        )

    for name, validator in validators.items():
        test_save_path = os.path.join(model_save_dir, name + "_result.csv")
        if not os.path.exists(test_save_path):
            with open(test_save_path, "a+") as f:
                f.write(
                    "loss,contrast_loss,self_contrast_loss,p@5,p@10,p@20,ndcg@5,ndcg@10,ndcg@20\n"
                )

    best_recall = 0
    if args.pretrain is not None:
        last_result_csv = pd.read_csv(os.path.join(model_save_dir, args.key_validators_name + "_result.csv"), header=0)
        best_recall = last_result_csv["p@10"].max()
    print("Best Recall:", best_recall)

    for i_epoch in range(args.epoch):
        for (
            user,
            item,
            item_genres,
            item_img_feature,
            neg_user,
            positive_item_list,
            negative_item_list,
            self_neg_list,
        ) in tqdm(train_loader):
            optimizer.zero_grad()
            model.train()

            # allocate memory cpu to gpu
            model = model.to(device)
            user = user.to(device)
            item = item.to(device)
            item_genres = item_genres.to(device)
            item_img_feature = item_img_feature.to(device)
            neg_user = neg_user.to(device)
            positive_item_list = positive_item_list.to(device)
            negative_item_list = negative_item_list.to(device)

            # run model
            q_v_c = model(item_genres, item_img_feature, user)
            q_v_c_unsqueeze = q_v_c.unsqueeze(dim=1)

            # Calculate contrast loss
            positive_item_emb = model.item_embedding[positive_item_list]
            pos_contrast_mul = torch.sum(
                torch.mul(q_v_c_unsqueeze, positive_item_emb), dim=2
            ) / (
                args.tau
                * torch.norm(q_v_c_unsqueeze, dim=2)
                * torch.norm(positive_item_emb, dim=2)
            )
            pos_contrast_exp = torch.exp(pos_contrast_mul)  # shape = 1024*10

            # Calculate negative examples
            neg_item_emb = model.item_embedding[negative_item_list]
            q_v_c_un2squeeze = q_v_c_unsqueeze.unsqueeze(dim=1)
            neg_contrast_mul = torch.sum(
                torch.mul(q_v_c_un2squeeze, neg_item_emb), dim=3
            ) / (
                args.tau
                * torch.norm(q_v_c_un2squeeze, dim=3)
                * torch.norm(neg_item_emb, dim=3)
            )
            neg_contrast_exp = torch.exp(neg_contrast_mul)
            neg_contrast_sum = torch.sum(neg_contrast_exp, dim=2)  # shape = [1024, 10]
            contrast_val = -torch.log(
                pos_contrast_exp / (pos_contrast_exp + neg_contrast_sum)
            )  # shape = [1024*10]
            contrast_examples_num = contrast_val.shape[0] * contrast_val.shape[1]
            contrast_sum = (
                torch.sum(torch.sum(contrast_val, dim=1), dim=0) / contrast_val.shape[1]
            )  # 同一个batch求mean

            # self contrast
            self_neg_item_emb = model.item_embedding[self_neg_list]
            self_neg_contrast_mul = torch.sum(
                torch.mul(q_v_c_unsqueeze, self_neg_item_emb), dim=2
            ) / (
                args.tau
                * torch.norm(q_v_c_unsqueeze, dim=2)
                * torch.norm(self_neg_item_emb, dim=2)
            )
            self_neg_contrast_sum = torch.sum(torch.exp(self_neg_contrast_mul), dim=1)
            item_emb = model.item_embedding[item]
            self_pos_contrast_mul = torch.sum(torch.mul(q_v_c, item_emb), dim=1) / (
                args.tau * torch.norm(q_v_c, dim=1) * torch.norm(item_emb, dim=1)
            )
            self_pos_contrast_exp = torch.exp(self_pos_contrast_mul)  # shape = 1024*1
            self_contrast_val = -torch.log(
                self_pos_contrast_exp / (self_pos_contrast_exp + self_neg_contrast_sum)
            )
            self_contrast_sum = torch.sum(self_contrast_val)

            # L_rank z_v & u
            user_emb = model.user_embedding[user]
            item_emb = model.item_embedding[item]
            neg_user_emb = model.user_embedding[neg_user]
            logsigmoid = torch.nn.LogSigmoid()
            y_uv = torch.mul(item_emb, user_emb).sum(dim=1)
            y_kv = torch.mul(item_emb, neg_user_emb).sum(dim=1)
            y_ukv = -logsigmoid(y_uv - y_kv).sum()

            # L_rank q_v & u
            y_uv2 = torch.mul(q_v_c, user_emb).sum(dim=1)
            y_kv2 = torch.mul(q_v_c, neg_user_emb).sum(dim=1)
            y_ukv2 = -logsigmoid(y_uv2 - y_kv2).sum()
            total_loss = args.lambda1 * (contrast_sum + self_contrast_sum) + (
                1 - args.lambda1
            ) * (y_ukv + y_ukv2)

            if math.isnan(total_loss):
                print("loss is nan!, exit.", total_loss)
                exit(255)

            total_loss.backward()
            optimizer.step()

        model.eval()

        print(
            "Epoch:",
            i_epoch,
            "loss:",
            total_loss.item(),
            "contrast_loss:",
            contrast_sum.item(),
        )
        for name, validator in validators.items():
            print("Start validate:", name)
            with torch.no_grad():
                hr_5, hr_10, hr_20, ndcg_5, ndcg_10, ndcg_20, ratings = (
                    validator.start_validate(model)
                )

            if name == args.key_validators_name:
                if hr_10 >= best_recall:
                    best_recall = hr_10
                    torch.save(model.state_dict(), model_save_dir + "/best_model.pt")
                    np.save(model_save_dir + "/best_model_ratings.npy", ratings)

            test_save_path = os.path.join(model_save_dir, name + "_result.csv")
            with open(test_save_path, "a+") as f:
                f.write(
                    "{},{},{},{},{},{},{},{},{}\n".format(
                        y_ukv + y_ukv2,
                        contrast_sum,
                        self_contrast_sum,
                        hr_5,
                        hr_10,
                        hr_20,
                        ndcg_5,
                        ndcg_10,
                        ndcg_20,
                    )
                )

        # save model
        if not args.save_best_only:
            torch.save(
                model.state_dict(), model_save_dir + "/epoch_" + str(i_epoch) + ".pt"
            )
        print("")
