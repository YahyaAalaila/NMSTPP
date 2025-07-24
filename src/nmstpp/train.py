import itertools
from NMSTPP.src.nmstpp.model.stpp import DeepBasisPointProcess
import torch
import torch.optim as optim
import arrow
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader




def train(model,
          train_loader,
          trg_model,
          test_data,
          ts, T, S, ngrid, 
          rootpath,
          modelname="pp", 
          num_epochs=10, 
          lr=1e-4, 
          print_iter=10):
    """training procedure"""
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    plot_fitted_2D_model(model, test_data, ts=ts, T=T, S=S, ngrid=ngrid, filename="initialization")

    for i in range(num_epochs):
        epoch_loss = 0
        for j, data in enumerate(train_loader):

            optimizer.zero_grad()
            X_batch   = data[0]
            _, loglik = model(X_batch)
            loss      = - loglik.mean()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss
            if j % print_iter == 0:
                print("[%s] Epoch : %d,\titer : %d,\tloss : %.5e" % (arrow.now(), i, j, loss / print_iter))
                torch.save(model.state_dict(), "%s/saved_models/%s.pth" % (rootpath, modelname))
        
        plot_fitted_2D_model(model, test_data, ts=ts, T=T, S=S, ngrid=ngrid, filename="epoch %d iter %d" % (i, j))
        print("[%s] Epoch : %d,\tTotal loss : %.5e" % (arrow.now(), i, epoch_loss))

def plot_fitted_2D_model(model, points, 
                         ts=[10, 50, 80], 
                         T=(0., 100.),
                         S=(0., 100.), 
                         ngrid=1000, filename="epoch 0"):
    """
    visualize the fitted model
    """
    # calculate 2d kernel (time only) given the fitted kernel module and one of its input
    def calc_2d_kernel(kernel, t, S, ngrid):
        tt = torch.FloatTensor([t]).unsqueeze(0).repeat(ngrid**2, 1)    # [ ngrid^2, 1 ]
        s  = np.linspace(S[0], S[1], ngrid)                             # [ ngrid ] 
        ss = torch.FloatTensor(np.array(list(itertools.product(s, s)))) # [ ngrid^2, 2 ] 
        xx = torch.cat([tt, ss[:, [0]]], 1)                             # [ ngrid^2, 2 ]
        yy = torch.cat([tt, ss[:, [1]]], 1)                             # [ ngrid^2, 2 ]
        vals = kernel(xx, yy)                                           # [ ngrid * ngrid ]
        vals = vals.reshape(ngrid, ngrid).detach().numpy()              # [ ngrid, ngrid ]
        return vals
        
    # calculate 2d point process (time only) given the fitted point process model
    def calc_2d_pointprocess(model, points, T, ngrid):
        ts      = torch.linspace(T[0], T[1], ngrid)
        lamvals = []
        for t in ts:
            _t     = t.unsqueeze(0).repeat(ngrid, 1)                    # [ ngrid, 1 ]
            s      = torch.linspace(S[0], S[1], ngrid).unsqueeze(1)     # [ ngrid, 1 ]
            x      = torch.cat([_t, s], 1)                              # [ ngrid, 2 ]
            ind    = np.where((points[:, 0] <= t) & (points[:, 0] > 0))[0]
            his_x  = points[ind, :]                                     # [ seq_len, 2 ]
            his_x  = his_x.unsqueeze(0).repeat(ngrid, 1, 1)             # [ ngrid, seq_len, 2 ]
            lamval = model.cond_lambda(x, his_x)                        # [ ngrid ]
            lamval = lamval                                             # [ ngrid ]
            lamval = (torch.nn.functional.softplus(lamval) + 1e-5).detach().numpy()
            lamvals.append(lamval)
        lamvals = np.stack(lamvals, 0)                                  # [ ngrid, ngrid ]
        return ts, lamvals  

    def calc_pairwise_kernel(kernel, points):
        xx, yy = [], []
        for i in range(len(points)):
            for j in range(len(points)):
                xx.append(points[i, :])
                yy.append(points[j, :])
        xx   = torch.stack(xx, 0)                                       # [ seq_len^2, 2 ]
        yy   = torch.stack(yy, 0)                                       # [ seq_len^2, 2 ]
        vals = kernel(xx, yy)                                           # [ ngrid * ngrid ] 
        vals = vals.reshape(len(points), len(points)).detach().numpy()  # [ seq_len, seq_len ]
        return vals                   

    # plot
    kernelvals     = []
    for t in ts:
        kernelval  = calc_2d_kernel(model.kernel, t, S, ngrid)
        kernelvals.append(kernelval)
    pairkernelvals = calc_pairwise_kernel(model.kernel, points)
    tt, lamvals    = calc_2d_pointprocess(model, points, T, ngrid)
    avglamvals     = lamvals.mean(1)

    fig, handles = plt.subplots(1, len(ts)+3, figsize=(5 * (len(ts)+3), 4))
    fig.suptitle(filename, fontsize=15)

    for i, ax in enumerate(handles[:len(ts)]):
        imi = ax.imshow(kernelvals[i])
        ax.set_xlabel("mark 1")
        ax.set_ylabel("mark 2")
        fig.colorbar(imi, ax=ax, shrink=.9)
        ax.title.set_text('kernel at time %d' % ts[i])

    ax = handles[-3]
    imi = ax.imshow(pairkernelvals)
    ax.set_xlabel("point y")
    ax.set_ylabel("point x")
    fig.colorbar(imi, ax=ax, shrink=.9)
    ax.title.set_text('pairwise kernel eval for a seq')

    ax = handles[-2]
    imi = ax.imshow(lamvals.transpose())
    ax.set_xlabel("time")
    ax.set_ylabel("mark")
    fig.colorbar(imi, ax=ax, shrink=.9)
    ax.title.set_text('lambda for a sequence')

    ax = handles[-1]
    ax.plot(tt, avglamvals, linestyle="-", color="black")
    ax.set_xlabel("time")
    ax.set_ylabel("lambda value")
    ax.title.set_text('avg lambda for a sequence')
    plt.show()