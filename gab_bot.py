import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import time
import pickle

from util import PLAYERS_PER_TEAM, MAP_WIDTH, MAP_HEIGHT
from util import Board, generate_teams, save_all, load_all, pad

torch.set_printoptions(threshold=10_000)

def main():
    with open('./data/ss', 'rb') as ss_file:
        all_states = pickle.load(ss_file)
    with open('./data/res', 'rb') as res_file:
        results = pickle.load(res_file)
    with open('./data/lens', 'rb') as lens_file:
        time_points = pickle.load(lens_file)

    bot = gab_bot()

    MODEL_FILE = "./model_saves/in_game"
    bot.load(MODEL_FILE)

    bot.enter_data(all_states, results, time_points)
    val_loss, accuracy, y_pred, y_true, time_true = bot.validate()

    print(f"Overall Accuracy: {accuracy:.2f}\n")

    for i in range(30):
        correct = len([1 for j, x in enumerate(time_true) if x == i and y_pred[j] == y_true[j]])
        total = len([1 for j, x in enumerate(time_true) if x == i])
        if(total > 0): print(f"Accuracy at time step {i}: {correct/total:.2f}")
    #bot.train_model()
    #bot.save(MODEL_FILE)

class gab_bot(nn.Module):
    def __init__(self):
        super(gab_bot, self).__init__()

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(9, 16, 4) #(16,16,6)
        self.conv2 = nn.Conv2d(16, 32, 2) # (14,14,16)
        # an affine operation: y = Wx + b
        #self.fc1 = nn.Linear(64 * 6 * 6, 240)  # 5*5 from image dimension
        self.fc1 = nn.Linear(32 * 6 * 6, 240)  # 5*5 from image dimension
        self.fc2 = nn.Linear(240, 64)
        self.fc3 = nn.Linear(64, 3)

        self.crit = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        #self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

        self.batch_size = 16

    def enter_data(self, starting_states, results, time_points):
        results = [2 if x_ == -1 else x_ for x_ in results]

        cut = int(0.9 * len(results))
        self.X_train = torch.Tensor(np.stack(starting_states[:cut]))
        self.y_train = torch.Tensor(results[:cut]).type(torch.LongTensor)
        self.time_train = torch.Tensor(time_points[:cut]).type(torch.LongTensor)

        self.X_val = torch.Tensor(np.stack(starting_states[cut:]))
        self.y_val = torch.Tensor(results[cut:]).type(torch.LongTensor)
        self.time_val = torch.Tensor(time_points[cut:]).type(torch.LongTensor)

        self.train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train, self.time_train)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.val_dataset = torch.utils.data.TensorDataset(self.X_val, self.y_val, self.time_val)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def train_model(self):
        stop_window = 1

        val_losses, accs = [], []
        print("Starting Model Training...")
        t0 = time.perf_counter()
        for e in range(100):
            losses = []
            self.train()
            for i, (x, y, t) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                out = self(x)
                loss = self.crit(out, y) / y.shape[0]
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
            train_loss = sum(losses) / len(losses)

            val_loss, accuracy, y_pred, y_true, time_true = self.validate()

            t1 = time.perf_counter()
            print(f"train_loss:{train_loss:.2f}\tval_loss:{val_loss:.2f}\tacc:{accuracy:.2f}\ttime:{t1-t0:.2f} sec")
            accs.append(accuracy)
            val_losses.append(val_loss)
            if(len(val_losses) >= stop_window * 2 and sum(val_losses[-stop_window*2:-stop_window]) < sum(val_losses[-stop_window:])):
                print("Stopping at epoch:", e)
                break

    def validate(self):
        val_loss = 0
        y_pred, y_true, time_true = [], [], []

        self.eval()
        with torch.no_grad():
            for i, (x, y, t) in enumerate(self.val_loader):
                out = self(x)
                pred = torch.argmax(out, dim=1)
                y_pred.extend(pred)
                y_true.extend(y)
                time_true.extend(t)
                loss = self.crit(out, y) / y.shape[0]
                val_loss += loss.item()
            val_loss = val_loss / len(self.val_loader)
        accuracy = len([1 for i, y in enumerate(y_pred) if y == y_true[i]]) / len(y_true)
        return val_loss, accuracy, y_pred, y_true, time_true

    def forward(self, x):
        if(len(x.shape) == 3):
            x = torch.Tensor(x).unsqueeze(dim=0)
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))

        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, path):
        print("Saving Model at:", path)
        torch.save({'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_val': self.X_val,
            'y_val': self.y_val}, path)
        print("Save Complete!")

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.X_train = checkpoint['X_train']
        self.y_train = checkpoint['y_train']
        self.X_val = checkpoint['X_val']
        self.y_val = checkpoint['y_val']

if(__name__ == "__main__"):
    main()
