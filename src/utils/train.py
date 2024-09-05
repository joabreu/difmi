from . import misc as misc
from tqdm import tqdm
import torch

def train_helper(model, train_loader, loss_fn, test_loader=None, lr=1e-3, epochs=50, device="cuda"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pbar = tqdm(range(epochs), "train")
    for i in pbar:
        model.train()
        for _, (x, y) in enumerate(train_loader):
            X, Y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = model.train_helper(X, Y, loss_fn)
            optimizer.step()

        pbar.set_postfix(loss=loss.item() / len(train_loader))
        if test_loader is not None:
            acc = 0
            model.eval()
            for _, (x, y) in enumerate(tqdm(test_loader, "test")):
                y_pred = model(x.to(device))
                acc += (y.to(device) == torch.argmax(y_pred, dim=-1))
            print(f"epoch: {i}, accuracy: {torch.mean(acc.float()).item() / len(test_loader)}")