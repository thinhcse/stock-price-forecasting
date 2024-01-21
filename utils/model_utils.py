import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

def test(model, data_test_iter, device):
  model.to(device)
  test_loss = torch.tensor(0.0).to(device)

  for input, groundtruth in data_test_iter:
      input = input.to(device)
      groundtruth = groundtruth.to(device)
      preds = model(input).ravel().to(device)
      test_loss += F.mse_loss(preds, groundtruth).to(device)

  return test_loss

def train(model, data_train_iter, data_val_iter, device, configs):

  model.apply(init_weights)
  optimizer = optim.Adam(model.parameters(), lr=configs["train"]["learning_rate"])
  loss = nn.MSELoss()
  model.to(device)

  print(f"___Start training on {device}___")
  tic = time.time()

  for epoch in range(configs["train"]["num_epochs"]):

    model.train()
    train_loss = torch.tensor(0.0).to(device)
    for input, groundtruth in data_train_iter:
      optimizer.zero_grad()
      input = input.to(device)
      groundtruth = groundtruth.to(device)
      preds = model(input).ravel().to(device)
      current_loss = loss(preds, groundtruth)
      current_loss.backward()
      optimizer.step()

    with torch.no_grad():
      train_loss += current_loss
      model.eval()
      eval_loss = test(model, data_val_iter, device).to(device)
      if (epoch % 20 == 0):
        print("Epoch %d Training loss %.3f Validation loss %.3f" %(epoch + 1, train_loss, eval_loss))
  
  tac = time.time()
  print("___Finish training in %.3f sec___" %(tac - tic))

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
