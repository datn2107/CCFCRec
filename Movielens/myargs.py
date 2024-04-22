import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/", help="data path")
    parser.add_argument(
        "--save-path", type=str, default="data/result", help="save path"
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="batch_size")
    parser.add_argument(
        "--learning-rate", type=float, default=0.000005, help="learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=0.1, help="weight decay")
    parser.add_argument(
        "--n-positive", type=int, default=10, help="contrast positive number"
    )
    parser.add_argument(
        "--n-negative", type=int, default=40, help="contrast negative number"
    )
    parser.add_argument(
        "--n-self-negative", type=int, default=40, help="contrast negative number"
    )
    parser.add_argument(
        "--attr-num", type=int, default=18, help="item attribute number"
    )
    parser.add_argument(
        "--attr-present-dim", type=int, default=128, help="the dimension of present"
    )
    parser.add_argument(
        "--implicit-dim", type=int, default=128, help="the dimension of u/i present"
    )
    parser.add_argument(
        "--cat-implicit-dim", type=int, default=128, help="the q_v_c dimension"
    )
    parser.add_argument(
        "--tau", type=float, default=0.1, help="contrast loss temperature"
    )
    parser.add_argument(
        "--lambda1", type=float, default=0.5, help="collaborative contrast loss weight"
    )
    parser.add_argument("--epoch", type=int, default=10, help="training epoch")
    parser.add_argument(
        "--pretrain", type=str, default=None, help="model pre-training weight"
    )
    parser.add_argument(
        "--contrast-flag", type=bool, default=True, help="contrast job flag"
    )
    parser.add_argument(
        "--user-flag", type=bool, default=False, help="use user to q_v_c flag"
    )
    parser.add_argument(
        "--key-validators-name",
        type=str,
        default="users_validator_test_set",
        help="validators name to indicate the best model",
    )
    parser.add_argument(
        '--save-best-only',
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
