import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/", help="data path")
    parser.add_argument(
        "--save-path", type=str, default="data/result", help="save path"
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="batch-size")
    parser.add_argument(
        "--learning-rate", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=0.1, help="weight decay")
    parser.add_argument(
        "--n-positive", type=int, default=5, help="contrast positive number"
    )
    parser.add_argument(
        "--n-negative", type=int, default=40, help="contrast negative number"
    )
    parser.add_argument(
        "--n-self-negative", type=int, default=40, help="contrast negative number"
    )
    parser.add_argument(
        "--attr-present-dim", type=int, default=256, help="the dimension of present"
    )
    parser.add_argument(
        "--implicit-dim", type=int, default=256, help="the dimension of u/i present"
    )
    parser.add_argument(
        "--cat-implicit-dim", type=int, default=256, help="the q-v-c dimension"
    )
    parser.add_argument(
        "--user-number", type=int, default=138493, help="user number in training set"
    )
    parser.add_argument(
        "--item-number", type=int, default=16803, help="item number in training set"
    )
    parser.add_argument(
        "--tau", type=float, default=0.1, help="contrast loss temperature"
    )
    parser.add_argument(
        "--lambda1", type=float, default=0.6, help="collaborative contrast loss weight"
    )
    parser.add_argument("--epoch", type=int, default=100, help="training epoch")
    parser.add_argument(
        "--pretrain", type=bool, default=False, help="user/item embedding pre-training"
    )
    parser.add_argument(
        "--pretrain-update",
        type=bool,
        default=False,
        help="u/i pretrain embedding update",
    )
    parser.add_argument(
        "--contrast-flag", type=bool, default=True, help="contrast job flag"
    )
    parser.add_argument(
        "--user-flag", type=bool, default=False, help="use user to q-v-c flag"
    )
    parser.add_argument(
        "--key-validators-name",
        type=str,
        default="users_validator_test_set",
        help="validators name to indicate the best model",
    )
    parser.add_argument(
        '--save_best_only',
        default=False,
        action='store_true',
    )
    args = parser.parse_args()
    return args


def args_tostring(args):
    str_ = ""
    for arg in vars(args):
        str_ += str(arg) + ":" + str(getattr(args, arg)) + "\n"
    return str_
