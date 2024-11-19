from IPython import display
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from botorch.models.gpytorch import GPyTorchModel
import gpytorch
from sklearn.preprocessing import StandardScaler
import torch


dtype = torch.double


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


def fit_exact_gp(X, y, state_dict=None, iprint=True, raw_noise=0.0):
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


# animate plots?
animate=False # True / False

# define CSTR model
def cstr(x,t,u,Tf,Caf):
    # Inputs (3):
    # Temperature of cooling jacket (K)
    Tc = u
    # Tf = Feed Temperature (K)
    # Caf = Feed Concentration (mol/m^3)

    # States (2):
    # Concentration of A in CSTR (mol/m^3)
    Ca = x[0]
    # Temperature in CSTR (K)
    T = x[1]

    # Parameters:
    # Volumetric Flowrate (m^3/sec)
    q = 100
    # Volume of CSTR (m^3)
    V = 100
    # Density of A-B Mixture (kg/m^3)
    rho = 1000
    # Heat capacity of A-B Mixture (J/kg-K)
    Cp = 0.239
    # Heat of reaction for A->B (J/mol)
    mdelH = 5e4
    # E - Activation energy in the Arrhenius Equation (J/mol)
    # R - Universal Gas Constant = 8.31451 J/mol-K
    EoverR = 8750
    # Pre-exponential factor (1/sec)
    k0 = 7.2e10
    # U - Overall Heat Transfer Coefficient (W/m^2-K)
    # A - Area - this value is specific for the U calculation (m^2)
    UA = 5e4
    # reaction rate
    rA = k0*np.exp(-EoverR/T)*Ca

    # Calculate concentration derivative
    dCadt = q/V*(Caf - Ca) - rA
    # Calculate temperature derivative
    dTdt = q/V*(Tf - T) \
            + mdelH/(rho*Cp)*rA \
            + UA/V/rho/Cp*(Tc-T)

    # Return xdot:
    xdot = np.zeros(2)
    xdot[0] = dCadt
    xdot[1] = dTdt
    return xdot

# Steady State Initial Conditions for the States
Ca_ss = 0.9519404136561265
T_ss = 312.6562010267356
x0 = np.empty(2)
x0[0] = Ca_ss
x0[1] = T_ss

# Steady State Initial Condition
u_ss = 290.0
# Feed Temperature (K)
Tf = 350
# Feed Concentration (mol/m^3)
Caf = 1

# Time Interval (min)
t = np.linspace(0,100,1000)

# Store results for plotting
Ca = np.ones(len(t)) * Ca_ss
T = np.ones(len(t)) * T_ss
u = np.ones(len(t)) * u_ss
Tf = np.ones(len(t)) * Tf + np.random.normal(0,1,len(t))
Caf = np.ones(len(t)) * Caf + np.random.normal(0,0.005,len(t))

# Step cooling temperature
for i in range(4):
    u[250*i:250*(i+1)] = 290.0 + i*3.0
u[-1] = u[-2]

# Create plot
plt.figure(figsize=(10,6))

if animate:
    plt.ion()
    plt.show()

# TODO write a steady state detector
u_steady_state = []
Ca_steady_state = []
T_steady_state = []
Caf_steady_state = []

# Simulate CSTR
for i in range(len(t)-1):
    ts = [t[i],t[i+1]]

    y = odeint(cstr,x0,ts,args=(u[i+1],Tf[i+1],Caf[i+1]))
    Ca[i+1] = y[-1][0]
    T[i+1] = y[-1][1]
    x0[0] = Ca[i+1]
    x0[1] = T[i+1]

    # check for steady state
    if abs(u[i+1] - u[i]) < 1e-6 and abs(Ca[i+1] - Ca[i]) < 0.01 and abs(T[i+1] - T[i]) < 0.5:
        # check if unique
        if u[i+1] not in u_steady_state:
            u_steady_state.append(u[i+1])
            Ca_steady_state.append(Ca[i+1])
            T_steady_state.append(T[i+1])
            Caf_steady_state.append(Caf[i+1])

    # plot results
    if animate:
        display.clear_output(wait=True)
        plt.figure(figsize=(10,6))
        # Plot the results
        plt.subplot(3,1,1)
        plt.plot(t[0:i+1],u[0:i+1],'b--')
        plt.ylabel('Cooling T (K)')
        plt.legend(['Jacket Temperature'],loc='best')

        plt.subplot(3,1,2)
        plt.plot(t[0:i+1],Ca[0:i+1],'r-')
        plt.ylabel('Ca (mol/L)')
        plt.legend(['Reactor Concentration'],loc='best')

        plt.subplot(3,1,3)
        plt.plot(t[0:i+1],T[0:i+1],'k-')
        plt.ylabel('T (K)')
        plt.xlabel('Time (min)')
        plt.legend(['Reactor Temperature'],loc='best')
        plt.pause(0.01)

# Construct results and save data file
# Column 1 = time
# Column 2 = cooling temperature
# Column 3 = reactor temperature
data = np.vstack((t,u,T)) # vertical stack
data = data.T             # transpose data
np.savetxt('cstr_data.txt',data,delimiter=',',comments="",header='time,u,y')

print(u_steady_state)
print(Ca_steady_state)
print(T_steady_state)

# TODO train a GP model
# scaling
x_scaled, x_scaler = scale_data(np.array(u_steady_state).reshape(-1,1))
y1_scaled, y1_scaler = scale_data(np.array(Ca_steady_state).reshape(-1,1))
y2_scaled, y2_scaler = scale_data(np.array(T_steady_state).reshape(-1,1))

# initial GPs
gp1 = fit_exact_gp(x_scaled, y1_scaled, iprint=False)
gp2 = fit_exact_gp(x_scaled, y2_scaled, iprint=False)

# predictions
x_plot = np.linspace(290, 305, 500)
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


# Plot the results
if not animate:
    plt.subplot(3,1,1)
    plt.plot(t,u,'b--')
    plt.ylabel('Cooling T (K)')
    plt.legend(['Jacket Temperature'],loc='best')

    # plt.subplot(4,1,2)
    # plt.plot(t,Caf,'g--')
    # plt.ylabel('Feed Ca (mol/L)')
    # plt.legend(['Feed Concentration'],loc='best')

    plt.subplot(3,1,2)
    plt.plot(t,Ca,'r-')
    plt.ylabel('Ca (mol/L)')
    plt.legend(['Reactor Concentration'],loc='best')

    plt.subplot(3,1,3)
    plt.plot(t,T,'k-')
    plt.ylabel('T (K)')
    plt.xlabel('Time (min)')
    plt.legend(['Reactor Temperature'],loc='best')

    plt.figure()
    plt.scatter(u_steady_state,Ca_steady_state)
    plt.plot(x_plot, gp1_mean, 'r-', label='GP1')
    plt.fill_between(x_plot, gp1_mean - 2 * gp1_std, gp1_mean + 2 * gp1_std, alpha=0.2, color='r')

    plt.figure()
    plt.scatter(u_steady_state,T_steady_state)
    plt.plot(x_plot, gp2_mean, 'r-', label='GP2')
    plt.fill_between(x_plot, gp2_mean - 2 * gp2_std, gp2_mean + 2 * gp2_std, alpha=0.2, color='r')
    # plt.axhline(330, color='k', linestyle='--', label='limit')

    plt.show()