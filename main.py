import yaml, os
from models.model import model
from utils.data_utils import data_preparation
from utils.model_utils import train
from utils.visualization import return_plot
from argparse import ArgumentParser, BooleanOptionalAction

import torch
from torchsummary import summary

parser = ArgumentParser()
parser.add_argument("--train", dest="is_train", help="To turn on training mode", action=BooleanOptionalAction)
parser.add_argument("--config-file", dest="config_path", help="Path to configuration file", metavar="str",
                    default=os.path.join(os.getcwd(), "configs", "config.yaml"))

args = parser.parse_args()
is_train = args.is_train
config_path = args.config_path

with open(config_path) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

data = data_preparation(configs)
data_train_iter, data_val_iter, data_test_iter = data[0]
time_stamps_train, time_stamps_val, time_stamps_test = data[1]

forecaster = model(configs)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

if __name__ == "__main__":

    if is_train:
        train(forecaster, data_train_iter, data_val_iter, device, configs)
        torch.save(forecaster.state_dict(), configs["pretrain"])
    else:
        forecaster.load_state_dict(torch.load(configs["pretrain"]))

    return_plot(forecaster, data_test_iter, time_stamps_test, device, configs)

    x = next(iter(data_train_iter))
    print(summary(forecaster, x[0]))
