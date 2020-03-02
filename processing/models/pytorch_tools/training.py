import torch
import numpy as np
from torch.utils.data import DataLoader
from .early_stopping import EarlyStopping
from misc.utils import printd

def loss_batch(model, loss_func, xb, yb, opt=None):
    """ compute the loss for a mini batch"""
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, batch_size, model, loss_func, opt, train_ds, valid_ds, patience, checkpoint_file):
    """ fit the model on the training data given the loss, optimizer, batch size, epochs, and earlystopping patience """
    train_dl, valid_dl = create_dataloaders_from_datasets(train_ds, valid_ds, batch_size)

    early_stopping = EarlyStopping(patience=patience,
                                   path=checkpoint_file)

    model.eval()
    early_stopping, res = evaluate(0, early_stopping, model, loss_func, [train_dl, valid_dl])

    for epoch in range(epochs):
        model.train()
        zip(*[loss_batch(model, loss_func, xb, yb, opt) for xb, yb in train_dl])

        model.eval()
        early_stopping, res = evaluate(epoch, early_stopping, model, loss_func, [train_dl, valid_dl])

        if early_stopping.early_stop:
            printd("Early Stopped.")
            break

    early_stopping.save()


def evaluate(epoch, early_stopping, model, loss_func, dls):
    """ evaluate the dataloaders after 1 epoch of training """
    dls_names = ["[train]", "[valid]"]
    with torch.no_grad():
        loss = []
        for dl, name in zip(dls, dls_names):
            if loss_func.__class__.__name__ == "DALoss":
                losses, mse, nll, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in dl])
                sum = np.sum(nums)
                loss_dl = [np.sum(np.multiply(losses, nums)) / sum, np.sum(np.multiply(mse, nums)) / sum,
                           np.sum(np.multiply(nll, nums)) / sum]
                es_loss = loss_dl[1]
            else:
                losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in dl])
                loss_dl = np.sum(np.multiply(losses, nums)) / np.sum(nums)
                es_loss = loss_dl

            loss.append(loss_dl)

            if name == "[valid]":
                early_stopping(es_loss, model, epoch)

    res = np.r_[epoch, np.c_[dls_names, loss].ravel()]
    printd(*res)

    return early_stopping, res


def predict(model, ds):
    """ make the prediction """
    model.eval()
    dl = DataLoader(ds, batch_size=len(ds))

    trues = np.reshape(dl.dataset.tensors[1].cpu().detach().numpy(),(-1,1))
    preds = np.reshape(model(dl.dataset.tensors[0]).cpu().detach().numpy(),(-1,1))

    return trues, preds


def cuda2np(tensors):
    """ convert an array of cuda tensors to numpy """
    np_tensors = []
    for tensor in tensors:
        np_tensors.append(tensor.cpu().detach().numpy())
    return np_tensors


def create_dataloaders_from_datasets(train_ds, valid_ds, bs):
    """ create dataloader from datasets """
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
