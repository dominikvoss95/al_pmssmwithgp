import numpy as np
import copy
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, x_train, y_train, x_valid, y_valid, input_dim=12, output_dim=1, seed=42):
        super().__init__()

        # Set seed for reproducibility
        self.set_seed(seed)

        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = input_dim
        self.batch_size = 64
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
            )
        
        self.to(self.device)  

    def set_seed(self, seed):
        '''Set the seed for reproducibility'''
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, x):
        return self.model(x)

    def do_train_loop(self, lr=0.005, iters=1000, optimizer_type="Adam"):
        '''Train the MLP model'''
        print(f"[DEBUG] x_train shape: {self.x_train.shape}")

        train_dataset = TensorDataset(self.x_train, self.y_train)
        valid_dataset = TensorDataset(self.x_valid, self.y_valid)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        if optimizer_type == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0)
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr)

        criterion = nn.MSELoss()

        best_loss = float('inf')
        best_model = None
        losses_train = []
        losses_valid = []
        patience, patience_counter = 20, 0

        for i in range(iters):
            # Training
            self.train()
            epoch_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                output = self(x_batch)
                y_batch = y_batch.unsqueeze(-1)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_avg = epoch_loss / len(train_loader)
            losses_train.append(epoch_avg)

            # Validation
            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for x_valid, y_valid in valid_loader:
                    x_valid, y_valid = x_valid.to(self.device), y_valid.to(self.device)
                    y_valid = y_valid.unsqueeze(-1)
                    output_valid = self(x_valid)
                    loss_valid = criterion(output_valid, y_valid)
                    valid_loss += loss_valid.item()

            losses_valid.append(valid_loss / len(valid_loader))

            # Save best model
            if loss_valid.item() < best_loss:
                best_loss = loss_valid.item()
                best_model = copy.deepcopy(self.state_dict())
                patience_counter = 0
            # else:
            #     patience_counter += 1
            #     if patience_counter >= patience:
            #         print(f"Early stopping at epoch {i}")
            #         break
            
            if i % 100 == 0:
                print(f"Iter {i}/{iters} - Loss (Train): {epoch_avg:.3f} - Loss (Val): {loss_valid.item():.3f}")
        
        if best_model is not None:
            self.load_state_dict(best_model)

        return self, losses_train, losses_valid
