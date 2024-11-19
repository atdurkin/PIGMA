import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# from IMC tuning
Kc = 4.61730615181 * 2.0
tauI = 0.913444964569 / 4.0
tauD = 0.0

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
Ca_ss = 0.87725294608097
T_ss = 324.475443431599
x0 = np.empty(2)
x0[0] = Ca_ss
x0[1] = T_ss

# Steady State Initial Condition
u_ss = 300.0
# Feed Temperature (K)
Tf = 350
# Feed Concentration (mol/m^3)
Caf = 1

# Time Interval (min)
t = np.linspace(0,30,601)

# Store results for plotting
Ca = np.ones(len(t)) * Ca_ss
T = np.ones(len(t)) * T_ss
u = np.ones(len(t)) * u_ss


# storage for recording values
op = np.ones(len(t))*u_ss  # controller output
pv = np.zeros(len(t))  # process variable
e = np.zeros(len(t))   # error
ie = np.zeros(len(t))  # integral of the error
dpv = np.zeros(len(t)) # derivative of the pv
P = np.zeros(len(t))   # proportional
I = np.zeros(len(t))   # integral
D = np.zeros(len(t))   # derivative
sp = np.ones(len(t))*T_ss  # set point

# define a setpoint ramp or steps
for i in range(6):
    sp[i*100:(i+1)*100] = 300 + i*15.0
sp[600]=sp[599]

# Upper and Lower limits on OP
op_hi = 350.0
op_lo = 250.0

u_steady_state = []
Ca_steady_state = []
T_steady_state = []
Caf_steady_state = []

pv[0] = T_ss
# loop through time steps    
for i in range(len(t)-1):
    delta_t = t[i+1]-t[i]
    e[i] = sp[i] - pv[i]
    if i >= 1:  # calculate starting on second cycle
        dpv[i] = (pv[i]-pv[i-1])/delta_t
        ie[i] = ie[i-1] + e[i] * delta_t
    P[i] = Kc * e[i]
    I[i] = Kc/tauI * ie[i]
    D[i] = - Kc * tauD * dpv[i]
    op[i] = op[0] + P[i] + I[i] + D[i]
    if op[i] > op_hi:  # check upper limit
        op[i] = op_hi
        ie[i] = ie[i] - e[i] * delta_t # anti-reset windup
    if op[i] < op_lo:  # check lower limit
        op[i] = op_lo
        ie[i] = ie[i] - e[i] * delta_t # anti-reset windup
    ts = [t[i],t[i+1]]
    u[i+1] = op[i]
    
    if i % 100 == 0:      
        if i < len(t) / 2:
            Caf += 0.1
        else:
            Caf -= 0.2
        
    
    T_noise = 0
    y = odeint(cstr,x0,ts,args=(u[i+1],Tf + T_noise,Caf))
    Ca[i+1] = y[-1][0]
    T[i+1] = y[-1][1]
    x0[0] = Ca[i+1]
    x0[1] = T[i+1]
    pv[i+1] = T[i+1]

    # check for steady state
    if abs(u[i+1] - u[i]) < 0.01 and abs(Ca[i+1] - Ca[i]) < 0.01 and abs(T[i+1] - T[i]) < 0.001:
        # check if unique
        T_rounded = round(T[i+1], 1)
        if T_rounded not in T_steady_state:
            u_steady_state.append(u[i+1])
            Ca_steady_state.append(Ca[i+1])
            T_steady_state.append(T_rounded)
            Caf_steady_state.append(Caf)



op[len(t)-1] = op[len(t)-2]
ie[len(t)-1] = ie[len(t)-2]
P[len(t)-1] = P[len(t)-2]
I[len(t)-1] = I[len(t)-2]
D[len(t)-1] = D[len(t)-2]

# Construct results and save data file
# Column 1 = time
# Column 2 = cooling temperature
# Column 3 = reactor temperature
data = np.vstack((t,u,T)) # vertical stack
data = data.T             # transpose data
np.savetxt('data_doublet.txt',data,delimiter=',')
    
# Plot the results
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(t,u,'b--')
plt.ylabel('Cooling T (K)')
plt.legend(['Jacket Temperature'],loc='best')

plt.subplot(3,1,2)
plt.plot(t,Ca,'g-')
plt.ylabel('Ca (mol/L)')
plt.legend(['Reactor Concentration'],loc='best')

plt.subplot(3,1,3)
plt.plot(t,T,'k:',label='Reactor Temperature')
plt.plot(t,sp,'r--',label='Set Point')
plt.axhline(400, color='k', linestyle='--', label='Upper Limit')
plt.ylabel('T (K)')
plt.xlabel('Time (min)')
plt.legend(loc='best')

# plt.subplot(4,1,4)
# plt.plot(t,op,'r--',label='Controller Output (OP)')
# plt.plot(t,P,'g:',label='Proportional (Kc e(t))')
# plt.plot(t,I,'b-',label='Integral (Kc/tauI * Int(e(t))')
# plt.plot(t,D,'k-',label='Derivative (-Kc tauD d(PV)/dt)')
# plt.legend(loc='best')
# plt.ylabel('Output')
plt.tight_layout()

print(u_steady_state)
print(Ca_steady_state)
print(T_steady_state)

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Caf_steady_state, u_steady_state, Ca_steady_state, c='r', marker='o')

fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Caf_steady_state, u_steady_state, T_steady_state, c='r', marker='o')

plt.show()
