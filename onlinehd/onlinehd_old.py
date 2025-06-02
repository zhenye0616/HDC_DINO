import math
from typing import Union

import torch
import torch.nn as nn

from . import spatial
from .encoder import Encoder

from onlinehd import _fasthd


class OnlineHD(nn.Module):
    def __init__(self, classes : int, features : int, dim : int = 4000):
        super(OnlineHD, self).__init__()
        self.classes = classes
        self.dim = dim
        #self.encoder = Encoder(features, dim)
        self.encoder = nn.Linear(features, dim)
        self.model = nn.Linear(dim, classes)

    def __call__(self, x : torch.Tensor, encoded : bool = False):
        return self.scores(x, encoded=encoded) #.argmax(1)

    def predict(self, x : torch.Tensor, encoded : bool = False):
        return self(x, encoded=encoded)

    def probabilities(self, x : torch.Tensor, encoded : bool = False):
        return self.scores(x, encoded=encoded).softmax(1)

    def scores(self, x : torch.Tensor, encoded : bool = False):
        h = x if encoded else self.encode(x)
        logits = self.model(h)
        return logits

    def encode(self, x : torch.Tensor):
        return self.encoder(x)

    def fit(self,
            x : torch.Tensor,
            y : torch.Tensor,
            encoded : bool = False,
            lr : float = 0.035,
            epochs : int = 120,
            batch_size : Union[int, None, float] = 1024,
            one_pass_fit : bool = True,
            bootstrap : Union[float, str] = 0.01):
        h = x if encoded else self.encode(x)
        if one_pass_fit:
            self._one_pass_fit(h, y, lr, bootstrap)
        self._iterative_fit(h, y, lr, epochs, batch_size)
        return self

    def to(self, *args):
        '''
        Moves data to the device specified, e.g. cuda, cpu or changes
        dtype of the data representation, e.g. half or double.
        Because the internal data is saved as torch.tensor, the parameter
        can be anything that torch accepts. The change is done in-place.

        Args:
            device (str or :class:`torch.torch.device`) Device to move data.

        Returns:
            :class:`OnlineHD`: self
        '''


        self.model = self.model.to(*args)
        self.encoder = self.encoder.to(*args)
        return self

    def _one_pass_fit(self, h, y, lr, bootstrap):
        # initialize class hypervectors from a single datapoint
        if bootstrap == 'single-per-class':
            # get binary mask containing whether one datapoint belongs to each
            # class
            idxs = y == torch.arange(self.classes, device=h.device).unsqueeze_(1)
            # choses first datapoint for every class
            # banned will store already seen data to avoid using it later
            banned = idxs.byte().argmax(1)
            self.model.add_(h[banned].sum(0), alpha=lr)
        else:
            # will accumulate data from 0 to cut
            cut = math.ceil(bootstrap*h.size(0))
            h_ = h[:cut]
            y_ = y[:cut]
            # updates each class hypervector (accumulating h_)
            for lbl in range(self.classes):
                self.model[lbl].add_(h_[y_ == lbl].sum(0), alpha=lr)
            # banned will store already seen data to avoid using it later
            banned = torch.arange(cut, device=h.device)

        # todo will store not used before data
        n = h.size(0)
        todo = torch.ones(n, dtype=torch.bool, device=h.device)
        todo[banned] = False

        # will execute one pass learning with data not used during model
        # bootstrap
        h_ = h[todo]
        y_ = y[todo]
        _fasthd.onepass(h_, y_, self.model, lr)

    def _iterative_fit(self, h, y, lr, epochs, batch_size):
        n = h.size(0)
        for epoch in range(epochs):
            for i in range(0, n, batch_size):
                h_ = h[i:i+batch_size]
                y_ = y[i:i+batch_size]
                scores = self.scores(h_, encoded=True)
                y_pred = scores.argmax(1)
                wrong = y_ != y_pred

                # computes alphas to update model
                # alpha1 = 1 - delta[lbl] -- the true label coefs
                # alpha2 = delta[max] - 1 -- the prediction coefs
                aranged = torch.arange(h_.size(0), device=h_.device)
                alpha1 = (1.0 - scores[aranged,y_]).unsqueeze_(1)
                alpha2 = (scores[aranged,y_pred] - 1.0).unsqueeze_(1)

                for lbl in y_.unique():
                    m1 = wrong & (y_ == lbl) # mask of missed true lbl
                    m2 = wrong & (y_pred == lbl) # mask of wrong preds
                    self.model[lbl] += lr*(alpha1[m1]*h_[m1]).sum(0)
                    self.model[lbl] += lr*(alpha2[m2]*h_[m2]).sum(0)
