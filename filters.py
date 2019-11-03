import torch

MOVEMENT_FILTERS = torch.tensor([[[0, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 0]],
                                 [[0, 1, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 0, 0]],
                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 1, 0]]], dtype=torch.float32).unsqueeze(1)

SNAKE_FILTERS = torch.tensor([[[0, 0, 0],
                               [3, 2, 1],
                               [0, 0, 0]],
                              [[0, 3, 0],
                               [0, 2, 0],
                               [0, 1, 0]],
                              [[0, 0, 0],
                               [1, 2, 3],
                               [0, 0, 0]],
                              [[0, 1, 0],
                               [0, 2, 0],
                               [0, 3, 0]]], dtype=torch.float32).unsqueeze(1)
