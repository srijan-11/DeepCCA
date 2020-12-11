import torch
import torch.nn as nn
import numpy as np
from linear_cca import linear_cca
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from DeepCCAModels import Model
from utils import load_data, svm_classify
import time
import logging
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import torch.nn as nn
torch.set_default_tensor_type(torch.DoubleTensor)


class DeepCCA():
    def __init__(self, model, linear_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=torch.device('cpu')):
        self.model = nn.DataParallel(model)  # parallizing the data
        self.model.to(device) # select GPU or CPU
        self.epoch_num = epoch_num # Total no. of iteration
        self.batch_size = batch_size # size of batch whose loss is calculated at once
        self.loss = model.loss # correlation loss
        self.optimizer = torch.optim.RMSprop(  # optimizer 
            self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.device = device # GPU / CPU

        self.linear_cca = linear_cca # output of model is inputed on linear cca model.

        self.outdim_size = outdim_size # output dimension

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("DCCA.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)

    def fit(self, x1, x2, vx1=None, vx2=None, tx1=None, tx2=None, checkpoint='checkpoint.model'):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        vx1 & vx2 are validation data

        tx1 & tx2 are test data

        """
        x1.to(self.device) # input view 1
        x2.to(self.device) # input view 2

        data_size = x1.size(0) # rows of data

        if vx1 is not None and vx2 is not None:  
            best_val_loss = 0
            vx1.to(self.device)
            vx2.to(self.device)
        if tx1 is not None and tx2 is not None:
            tx1.to(self.device)
            tx2.to(self.device)

        train_losses = []
        for epoch in range(self.epoch_num): # running iterations on model
            epoch_start_time = time.time()
            self.model.train()  # setting model for train (recored grads and use dropouts)
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))  # making batches of index
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad() # removing older grads
                batch_x1 = x1[batch_idx, :] # selecting batch of data
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2) # output from model
                loss = self.loss(o1, o2) # loss calculation (-corr(o1,o2))
                train_losses.append(loss.item()) 
                loss.backward() # packpropogation of loss
                self.optimizer.step() # updating weights
            train_loss = np.mean(train_losses)

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx1 is not None and vx2 is not None: # validation data
                with torch.no_grad():
                    self.model.eval() # model in evalution mode (no dropouts and grads)
                    val_loss = self.transform(vx1, vx2)
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), checkpoint)
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_val_loss))
            else:
                torch.save(self.model.state_dict(), checkpoint) # save model
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss))
        # train_linear_cca
        if self.linear_cca is not None: # linear cca on output of model
            _, outputs = self._get_outputs(x1, x2)
            self.train_linear_cca(outputs[0], outputs[1])

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)
        if vx1 is not None and vx2 is not None:
            loss = self.transform(vx1, vx2)
            self.logger.info("loss on validation data: {:.4f}".format(loss))

        if tx1 is not None and tx2 is not None:
            loss = self.transform(tx1, tx2)
            self.logger.info('loss on test data: {:.4f}'.format(loss))

    def transform(self, x1, x2, use_linear_cca=False): # output for new data
        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)

            if use_linear_cca:
                print("Linear CCA started!")
                outputs = self.linear_cca.test(outputs[0], outputs[1])
                return np.mean(losses), outputs
            
            return np.mean(losses), outputs

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.model.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False)) # making batch of index
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2) #output of model
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()] # converting output to numpy form tensor
        return losses, outputs
