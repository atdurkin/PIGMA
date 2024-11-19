import matplotlib.pyplot as plt
import numpy as np

from botorch.models.gpytorch import GPyTorchModel
import gpytorch
from sklearn.preprocessing import StandardScaler
import torch


class ExactGPModel(gpytorch.models.ExactGP, GPyTorchModel):
    '''
    An exact GP regression model using GPyTorch.
    '''
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

def scale_data(x):
    '''
    A function to standardise data and save the scaler for later use.
    '''
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled, scaler


def fit_exact_gp(X, y, state_dict=None, iprint=True, raw_noise=1e-6):
    '''
    A function to fit the GP models.
    '''
    X = torch.tensor(X, dtype=dtype)
    y = torch.tensor(y, dtype=dtype)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X, y.ravel(), likelihood)

    # constrain the noise parameter to be greater than some value
    # TODO tune this value
    model.likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(raw_noise))

    if state_dict is not None:
        model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()

    training_iter = 500
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, y.ravel())
        loss.backward()
        if iprint:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()

    return model


def p1(x):
    '''The function p1 is assumed to be unknown'''
    noise = np.random.normal(0, 2, x.shape)
    return 120 * np.sin(0.6 * x) + 10 * np.cos(5 * x) - 10# + noise

def p2(x):
    '''The function p2 is assumed to be known and serves as the PI'''
    noise = np.random.normal(0, 2, x.shape)
    return 140 * x - p1(x)# + noise

def obj(x):
    '''The optimisation objective to be minimised'''
    return -p1(x)

def con(x):
    '''The optimisation constraint to be satisfied'''
    return p2(x) - 140


dtype = torch.double

# data
x = np.array([0.1, 0.2, 0.5]).reshape(-1, 1)
y1 = p1(x)
y2 = p2(x)

# scaling
x_scaled, x_scaler = scale_data(x)
y1_scaled, y1_scaler = scale_data(y1)
y2_scaled, y2_scaler = scale_data(y2)

# initial GPs
gp1 = fit_exact_gp(x_scaled, y1_scaled, iprint=False)
gp2 = fit_exact_gp(x_scaled, y2_scaled, iprint=False)

# predictions
x_plot = np.linspace(0, 2, 500)
x_test = x_scaler.transform(x_plot.reshape(-1, 1))

# GP1 predictions
gp1.eval()
gp1_mean = gp1(torch.tensor(x_test, dtype=dtype)).mean.detach().numpy().ravel()
gp1_std = gp1(torch.tensor(x_test, dtype=dtype)).variance.sqrt().detach().numpy().ravel()
gp1_mean = y1_scaler.inverse_transform(gp1_mean.reshape(-1, 1)).ravel()
gp1_std = gp1_std * y1_scaler.scale_

# GP2 predictions
gp2.eval()
gp2_mean = gp2(torch.tensor(x_test, dtype=dtype)).mean.detach().numpy().ravel()
gp2_std = gp2(torch.tensor(x_test, dtype=dtype)).variance.sqrt().detach().numpy().ravel()
gp2_mean = y2_scaler.inverse_transform(gp2_mean.reshape(-1, 1)).ravel()
gp2_std = gp2_std * y2_scaler.scale_

# determine initial correction factors
c_opt = []
for i in range(len(x_plot)):
    if (i + 1) % 10 == 0:
        print('%d/%d' % (i + 1, len(x_plot)))
    incumbent = x_plot[i]
    incumbent_scaled = x_scaler.transform(np.array([incumbent]).reshape(1, 1))
    incumbent_mu1_scaled = gp1(torch.tensor(incumbent_scaled, dtype=dtype)).mean.detach().numpy().ravel()
    incumbent_mu1 = y1_scaler.inverse_transform(incumbent_mu1_scaled.reshape(-1, 1)).ravel()
    incumbent_sigma1_scaled = gp1(torch.tensor(incumbent_scaled, dtype=dtype)).variance.sqrt().detach().numpy().ravel()
    incumbent_sigma1 = incumbent_sigma1_scaled * y1_scaler.scale_
    incumbent_mu2_scaled = gp2(torch.tensor(incumbent_scaled, dtype=dtype)).mean.detach().numpy().ravel()
    incumbent_mu2 = y2_scaler.inverse_transform(incumbent_mu2_scaled.reshape(-1, 1)).ravel()
    incumbent_sigma2_scaled = gp2(torch.tensor(incumbent_scaled, dtype=dtype)).variance.sqrt().detach().numpy().ravel()
    incumbent_sigma2 = incumbent_sigma2_scaled * y2_scaler.scale_

    c1 = np.linspace(-100, 100, 1000)
    c2 = np.linspace(-200, 200, 1000)
    w1 = 1 / incumbent_sigma1
    w2 = 1 / incumbent_sigma2

    # predict a grid of c1 and c2 values
    c_grid = np.meshgrid(c1, c2)
    c1_grid = c_grid[0].ravel()
    c2_grid = c_grid[1].ravel()

    picof_obj = (incumbent_mu1 + incumbent_mu2 - 140 * incumbent + c1_grid + c2_grid) ** 2 + w1 * c1_grid ** 2 + w2 * c2_grid ** 2
    picof_ind = np.argmin(picof_obj)
    c1_opt = c1_grid[picof_ind]
    c2_opt = c2_grid[picof_ind]
    c_opt.append([c1_opt, c2_opt])
c_opt = np.array(c_opt)

plt.figure()
plt.contourf(c1, c2, picof_obj.reshape(len(c1), len(c2)), levels=20)
plt.plot(c1_opt, c2_opt, 'r*', label='optimal')
plt.text(c1_opt, c2_opt, '  (%.2f, %.2f)' % (c1_opt, c2_opt), fontsize=10, color='r', va='center')
plt.xlabel('c1')
plt.ylabel('c2')
plt.colorbar()
plt.legend()
plt.title('PI-CoF objective')
plt.tight_layout()

# maximise UCB of p1 subject to UCB of p2 <= 140
feas_ind = np.where(gp2_mean + 1.96 * gp2_std <= 140)[0]
gp1_mean_feas = gp1_mean[feas_ind]
gp1_std_feas = gp1_std[feas_ind]
x_feas = x_plot[feas_ind]
opt_ind = np.argmax(gp1_mean_feas + 1.96 * gp1_std_feas)
x_opt = x_feas[opt_ind]

# maximise PI-COF UCB of p1 subject to PICoF UCB of p2 <= 140
feas_ind = np.where(gp2_mean + c_opt[:, 1] + 1.96 * gp2_std <= 140)[0]
picof1_mean_feas = gp1_mean[feas_ind] + c_opt[feas_ind, 0]
gp1_std_feas = gp1_std[feas_ind]
x_feas = x_plot[feas_ind]
opt_ind = np.argmax(picof1_mean_feas + 1.96 * gp1_std_feas)
x_opt_picof = x_feas[opt_ind]

# plotting p1
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(x_plot, p1(x_plot), 'k--', alpha=0.2, label='true')
axs[0].plot(x, y1, 'bx', label='initial samples')
axs[0].plot(x_plot, gp1_mean, 'b', label='GP mean')
axs[0].fill_between(x_plot, gp1_mean - 1.96 * gp1_std, gp1_mean + 1.96 * gp1_std, alpha=0.2, label='GP 95% CI')
axs[0].plot(x_opt, p1(x_opt), 'b*', label='optimal (no PI)')
axs[0].plot(x_opt_picof, p1(x_opt_picof), 'g*', label='optimal (PI-CoF)')
axs[0].plot(x_plot, gp1_mean + c_opt[:, 0], 'g--', label='PI-CoF')
axs[0].fill_between(x_plot, 
                    gp1_mean + c_opt[:, 0] - 1.96 * gp1_std, 
                    gp1_mean + c_opt[:, 0] + 1.96 * gp1_std, color='g', alpha=0.2, label='PI-CoF 95% CI')
axs[0].set_xlabel('x')
axs[0].set_ylabel('p1')
axs[0].legend()
# axs[0].set_title('p1 (fully unknown)')
plt.tight_layout()

# plotting p2
axs[1].plot(x_plot, p2(x_plot), 'k--', alpha=0.2, label='true')
axs[1].plot(x, y2, 'rx', label='initial samples')
axs[1].plot(x_plot, gp2_mean, 'r', label='GP mean')
axs[1].fill_between(x_plot, gp2_mean - 1.96 * gp2_std, gp2_mean + 1.96 * gp2_std, color='r', alpha=0.2, label='GP 95% CI')
axs[1].axhline(140, color='k', linestyle='--', label='constraint')
axs[1].plot(x_opt, p2(x_opt), 'r*', label='optimal (no PI)')
axs[1].plot(x_opt_picof, p2(x_opt_picof), 'g*', label='optimal (PI-CoF)')
axs[1].plot(x_plot, 140 * x_plot - gp1_mean, 'y--', label='PI')
axs[1].fill_between(x_plot, 
                    140 * x_plot - gp1_mean - 1.96 * gp1_std, 
                    140 * x_plot - gp1_mean + 1.96 * gp1_std, color='y', alpha=0.2, label='PI 95% CI')
axs[1].plot(x_plot, gp2_mean + c_opt[:, 1], 'g--', label='PI-CoF')
axs[1].fill_between(x_plot, 
                    gp2_mean + c_opt[:, 1] - 1.96 * gp2_std, 
                    gp2_mean + c_opt[:, 1] + 1.96 * gp2_std, color='g', alpha=0.2, label='PI-CoF 95% CI')
axs[1].set_xlabel('x')
axs[1].set_ylabel('p2')
axs[1].legend()
# axs[1].set_title('p2 (PI dependent on p1)')
plt.tight_layout()


# continue for more iterations and plot the final results
plot_obj = [x_opt]
for _ in range(10):
    x = np.r_[x, x_opt.reshape(-1, 1)]
    y1 = p1(x)
    y2 = p2(x)

    # scaling
    x_scaled, x_scaler = scale_data(x)
    y1_scaled, y1_scaler = scale_data(y1)
    y2_scaled, y2_scaler = scale_data(y2)

    # initial GPs
    gp1 = fit_exact_gp(x_scaled, y1_scaled, iprint=False)
    gp2 = fit_exact_gp(x_scaled, y2_scaled, iprint=False)

    # GP1 predictions
    x_test = x_scaler.transform(x_plot.reshape(-1, 1))
    gp1.eval()
    gp1_mean = gp1(torch.tensor(x_test, dtype=dtype)).mean.detach().numpy().ravel()
    gp1_std = gp1(torch.tensor(x_test, dtype=dtype)).variance.sqrt().detach().numpy().ravel()
    gp1_mean = y1_scaler.inverse_transform(gp1_mean.reshape(-1, 1)).ravel()
    gp1_std = gp1_std * y1_scaler.scale_

    # GP2 predictions
    gp2.eval()
    gp2_mean = gp2(torch.tensor(x_test, dtype=dtype)).mean.detach().numpy().ravel()
    gp2_std = gp2(torch.tensor(x_test, dtype=dtype)).variance.sqrt().detach().numpy().ravel()
    gp2_mean = y2_scaler.inverse_transform(gp2_mean.reshape(-1, 1)).ravel()
    gp2_std = gp2_std * y2_scaler.scale_

    # maximise UCB of p1 subject to UCB of p2 <= 140
    feas_ind = np.where(gp2_mean + 1.96 * gp2_std <= 140)[0]
    gp1_mean_feas = gp1_mean[feas_ind]
    gp1_std_feas = gp1_std[feas_ind]
    x_feas = x_plot[feas_ind]
    opt_ind = np.argmax(gp1_mean_feas + 1.96 * gp1_std_feas)
    x_opt = x_feas[opt_ind]
    plot_obj.append(x_opt)

# plotting p1
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(x_plot, p1(x_plot), 'k--', alpha=0.2, label='true')
axs[0].plot(x, y1, 'bx', label='samples')
axs[0].plot(x_plot, gp1_mean, 'b', label='GP mean')
axs[0].fill_between(x_plot, gp1_mean - 1.96 * gp1_std, gp1_mean + 1.96 * gp1_std, alpha=0.2, label='GP 95% CI')
axs[0].plot(x_opt, p1(x_opt), 'k*', label='optimal (no PI)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('p1')
axs[0].legend()
axs[0].set_title('p1 (fully unknown)')
plt.tight_layout()

# plotting p2
axs[1].plot(x_plot, p2(x_plot), 'k--', alpha=0.2, label='true')
axs[1].plot(x, y2, 'rx', label='samples')
axs[1].plot(x_plot, gp2_mean, 'r', label='GP mean')
axs[1].fill_between(x_plot, gp2_mean - 1.96 * gp2_std, gp2_mean + 1.96 * gp2_std, color='r', alpha=0.2, label='GP 95% CI')
axs[1].axhline(140, color='k', linestyle='--', label='constraint')
axs[1].plot(x_opt, p2(x_opt), 'k*', label='optimal (no PI)')
axs[1].set_xlabel('x')
axs[1].set_ylabel('p2')
axs[1].legend()
axs[1].set_title('p2 (PI dependent on p1)')
plt.tight_layout()

# PI-CoF iterations
x = x[:3]  # reset x
plot_picof_obj = [x_opt_picof]
for _ in range(10):
    x = np.r_[x, x_opt_picof.reshape(-1, 1)]
    y1 = p1(x)
    y2 = p2(x)

    # scaling
    x_scaled, x_scaler = scale_data(x)
    y1_scaled, y1_scaler = scale_data(y1)
    y2_scaled, y2_scaler = scale_data(y2)

    # initial GPs
    gp1 = fit_exact_gp(x_scaled, y1_scaled, iprint=False)
    gp2 = fit_exact_gp(x_scaled, y2_scaled, iprint=False)

    # GP1 predictions
    x_test = x_scaler.transform(x_plot.reshape(-1, 1))
    gp1.eval()
    gp1_mean = gp1(torch.tensor(x_test, dtype=dtype)).mean.detach().numpy().ravel()
    gp1_std = gp1(torch.tensor(x_test, dtype=dtype)).variance.sqrt().detach().numpy().ravel()
    gp1_mean = y1_scaler.inverse_transform(gp1_mean.reshape(-1, 1)).ravel()
    gp1_std = gp1_std * y1_scaler.scale_

    # GP2 predictions
    gp2.eval()
    gp2_mean = gp2(torch.tensor(x_test, dtype=dtype)).mean.detach().numpy().ravel()
    gp2_std = gp2(torch.tensor(x_test, dtype=dtype)).variance.sqrt().detach().numpy().ravel()
    gp2_mean = y2_scaler.inverse_transform(gp2_mean.reshape(-1, 1)).ravel()
    gp2_std = gp2_std * y2_scaler.scale_

    # determine correction factors
    c_opt = []
    for i in range(len(x_plot)):
        if (i + 1) % 10 == 0:
            print('%d/%d' % (i + 1, len(x_plot)))
        incumbent = x_plot[i]
        incumbent_scaled = x_scaler.transform(np.array([incumbent]).reshape(1, 1))
        incumbent_mu1_scaled = gp1(torch.tensor(incumbent_scaled, dtype=dtype)).mean.detach().numpy().ravel()
        incumbent_mu1 = y1_scaler.inverse_transform(incumbent_mu1_scaled.reshape(-1, 1)).ravel()
        incumbent_sigma1_scaled = gp1(torch.tensor(incumbent_scaled, dtype=dtype)).variance.sqrt().detach().numpy().ravel()
        incumbent_sigma1 = incumbent_sigma1_scaled * y1_scaler.scale_
        incumbent_mu2_scaled = gp2(torch.tensor(incumbent_scaled, dtype=dtype)).mean.detach().numpy().ravel()
        incumbent_mu2 = y2_scaler.inverse_transform(incumbent_mu2_scaled.reshape(-1, 1)).ravel()
        incumbent_sigma2_scaled = gp2(torch.tensor(incumbent_scaled, dtype=dtype)).variance.sqrt().detach().numpy().ravel()
        incumbent_sigma2 = incumbent_sigma2_scaled * y2_scaler.scale_

        c1 = np.linspace(-100, 100, 1000)
        c2 = np.linspace(-200, 200, 1000)
        w1 = 1 / incumbent_sigma1
        w2 = 1 / incumbent_sigma2

        # predict a grid of c1 and c2 values
        c_grid = np.meshgrid(c1, c2)
        c1_grid = c_grid[0].ravel()
        c2_grid = c_grid[1].ravel()

        picof_obj = (incumbent_mu1 + incumbent_mu2 - 140 * incumbent + c1_grid + c2_grid) ** 2 + w1 * c1_grid ** 2 + w2 * c2_grid ** 2
        picof_ind = np.argmin(picof_obj)
        c1_opt = c1_grid[picof_ind]
        c2_opt = c2_grid[picof_ind]
        c_opt.append([c1_opt, c2_opt])
    c_opt = np.array(c_opt)

    # maximise PI-COF UCB of p1 subject to PICoF UCB of p2 <= 140
    alpha = 2
    beta = 2
    feas_ind = np.where(gp2_mean + c_opt[:, 1] + beta * gp2_std <= 140)[0]
    picof1_mean_feas = gp1_mean[feas_ind] + c_opt[feas_ind, 0]
    gp1_std_feas = gp1_std[feas_ind]
    x_feas = x_plot[feas_ind]
    opt_ind = np.argmax(picof1_mean_feas + alpha * gp1_std_feas)
    x_opt_picof = x_feas[opt_ind]
    plot_picof_obj.append(x_opt_picof)


    # plotting p1
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(x_plot, p1(x_plot), 'k--', alpha=0.2, label='true')
    axs[0].plot(x, y1, 'bx', label='samples')
    axs[0].plot(x_plot, gp1_mean, 'b', label='GP mean')
    axs[0].fill_between(x_plot, gp1_mean - 1.96 * gp1_std, gp1_mean + 1.96 * gp1_std, alpha=0.2, label='GP 95% CI')
    axs[0].plot(x_opt_picof, p1(x_opt_picof), 'g*', label='optimal (PI-CoF)')
    axs[0].plot(x_plot, gp1_mean + c_opt[:, 0], 'g--', label='PI-CoF')
    axs[0].fill_between(x_plot, 
                        gp1_mean + c_opt[:, 0] - 1.96 * gp1_std, 
                        gp1_mean + c_opt[:, 0] + 1.96 * gp1_std, color='g', alpha=0.2, label='PI-CoF 95% CI')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('p1')
    axs[0].legend()
    axs[0].set_title('p1 (fully unknown)')
    plt.tight_layout()

    # plotting p2
    axs[1].plot(x_plot, p2(x_plot), 'k--', alpha=0.2, label='true')
    axs[1].plot(x, y2, 'rx', label='samples')
    # axs[1].plot(x_plot, gp2_mean, 'r', label='GP mean')
    # axs[1].fill_between(x_plot, gp2_mean - 1.96 * gp2_std, gp2_mean + 1.96 * gp2_std, color='r', alpha=0.2, label='GP 95% CI')
    axs[1].axhline(140, color='k', linestyle='--', label='constraint')
    axs[1].plot(x_opt_picof, p2(x_opt_picof), 'g*', label='optimal (PI-CoF)')
    # axs[1].plot(x_plot, 140 * x_plot - gp1_mean, 'y--', label='PI')
    # axs[1].fill_between(x_plot, 
    #                     140 * x_plot - gp1_mean - 1.96 * gp1_std, 
    #                     140 * x_plot - gp1_mean + 1.96 * gp1_std, color='y', alpha=0.2, label='PI 95% CI')
    axs[1].plot(x_plot, gp2_mean + c_opt[:, 1], 'g--', label='PI-CoF')
    axs[1].fill_between(x_plot, 
                        gp2_mean + c_opt[:, 1] - 1.96 * gp2_std, 
                        gp2_mean + c_opt[:, 1] + 1.96 * gp2_std, color='g', alpha=0.2, label='PI-CoF 95% CI')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('p2')
    axs[1].legend()
    axs[1].set_title('p2 (PI dependent on p1)')
    plt.tight_layout()

feas_ind = np.where(p2(x_plot) <= 140)[0]
p1_feas = p1(x_plot)[feas_ind]
x_feas = x_plot[feas_ind]
opt_ind = np.argmax(p1_feas)
x_opt = x_feas[opt_ind]

plt.figure()
plt.plot(p1(np.array(plot_obj)), label='no PI')
plt.plot(p1(np.array(plot_picof_obj)), label='PI-CoF')
plt.axhline(p1(x_opt), color='k', linestyle='--', label='true optimal')
plt.plot
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(p2(np.array(plot_obj)), label='no PI')
plt.plot(p2(np.array(plot_picof_obj)), label='PI-CoF')
plt.axhline(140, color='k', linestyle='--', label='constraint')
plt.legend()
plt.tight_layout()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(x_plot, p1(x_plot), 'k--', alpha=0.2, label='true')
axs[0].plot(x_opt, p1(x_opt), 'k*', label='optimal')
axs[1].plot(x_plot, p2(x_plot), 'k--', alpha=0.2, label='true')
axs[1].axhline(140, color='k', linestyle='--', label='constraint')
axs[1].plot(x_opt, p2(x_opt), 'k*', label='optimal')
axs[0].set_xlabel('x')
axs[1].set_xlabel('x')
axs[0].set_ylabel('p1')
axs[1].set_ylabel('p2')
axs[0].legend()
axs[1].legend()
plt.tight_layout()

# fin
print('Done')
plt.show()
