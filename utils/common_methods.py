import torch
import numpy as np


def change_to_tensor(data_np, dtype=torch.float32):
    """
    change numpy array to torch.tensor
    :param dtype:
    :param data_np:
    :return:
    """
    data_tensor = torch.from_numpy(data_np).type(dtype)
    if torch.cuda.is_available():
        data_tensor = data_tensor.cuda()
    return data_tensor
