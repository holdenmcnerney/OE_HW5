# Optimal Estimation - HW5 - Ballistic Vehicle Altimetry System Design

import numpy as np
import numpy.linalg as la
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

def f_km1_build(h_km1, s_km1, Cb_km1):
    f_km1 = np.array([[h_km1 + dt * s_km1], 
                     [s_km1 + dt * (rho0 * s_km1**2 / (2 * Cb_km1) * np.exp(-h_km1 / hp)
                                    - g0 * (RE / (RE + h_km1))**2)], 
                     [Cb_km1]])
    return f_km1

def F_km1_build(h_km1, s_km1, Cb_km1):
    '''
    Temp
    '''
    F_km1 = np.array([[1, dt, 0], 
                      [dt * (- rho0 * s_km1**2 / (2 * Cb_km1 * hp) * np.exp(-h_km1 / hp)
                             + 2 * g0 * (RE**2 / (RE + h_km1)**3)), 
                       1 + dt * rho0 * s_km1 / Cb_km1 * np.exp(-h_km1 / hp), 
                       - dt * rho0 * s_km1**2 / (2 * Cb_km1**2) * np.exp(-h_km1 / hp)], 
                      [0, 0, 1]])
    return F_km1

def H_km1_build(h_km1):
    '''
    Temp
    '''
    H_km1 = np.array([h_km1 * (d**2 + h_km1**2)**(-1/2), 0, 0])
    return H_km1

def EKF(y, x0, P0, Q, R):
    '''
    Temp
    '''
    x_hist = np.zeros((len(y), 3))
    x_km1_km1 = x0
    P_km1_km1 = P0

    for i, y_k in enumerate(y):
        if i == 0:
            x_hist[0, :] = np.atleast_2d(x0).T
        else:
            # Prediction Step
            h_km1 = x_km1_km1[0][0]
            s_km1 = x_km1_km1[1][0]
            Cb_km1 = x_km1_km1[2][0]
            x_k_km1 = f_km1_build(h_km1, s_km1, Cb_km1)
            F_km1 = F_km1_build(h_km1, s_km1, Cb_km1)
            P_k_km1 = F_km1 @ P_km1_km1 @ F_km1.T + Q
            h_k_km1 = x_k_km1[0][0]
            y_k_est = np.sqrt(d**2 + h_k_km1**2) 

            # Correction Step
            y_diff = y_k - y_k_est
            H_km1 = np.atleast_2d(H_km1_build(h_km1))
            S_k = H_km1 @ P_k_km1 @ H_km1.T + R
            K_k = P_k_km1 @ H_km1.T * S_k**-1
            x_k_k = x_k_km1 + K_k * y_diff
            # A = np.eye(3) - K_k @ H_km1
            # P_k_k = A @ P_k_km1 @ A.T + K_k * R @ K_k.T
            P_k_k = P_k_km1 - K_k @ S_k * K_k.T

            # Saving and reseting values
            x_hist[i, :] = np.atleast_2d(x_k_k).T
            x_km1_km1 = x_k_k
            P_km1_km1 = P_k_k

    return x_hist

def SP_UKF(y, x0, P0, Q, R):
    '''
    Temp
    '''
    x_hist = np.zeros((len(y), 3))
    x_km1_km1 = x0
    P_km1_km1 = P0
    alpha = 0.5
    beta = 2
    kappa = 0.5
    nx = 3
    lamb = alpha**2 * (nx + kappa) - nx

    for i, y_k in enumerate(y):
        if i == 0:
            x_hist[0, :] = np.atleast_2d(x0).T
        else:
            # Prediction Step
            x_aug = np.block([[x_km1_km1], [np.zeros((3, 1))], [0]])
            P_aug = np.block([[P_km1_km1, np.zeros((3, 4))], 
                              [np.zeros((3, 3)), Q, np.zeros((3, 1))], 
                              [np.zeros((1, 6)), R]])
            na = P_aug.shape[0]
            # Prior state sigma points
            chi_plus_k_km1 = []
            chi_minus_k_km1 = []
            w_vec = []
            w0m = lamb / (na + lamb)
            w0c = lamb / (na + lamb) + (1 - alpha**2 + beta)
            for i in np.arange(0, na):
                P_aug_temp = np.atleast_2d(P_aug[:, i]).T
                chi_plus = x_aug + np.sqrt((na + lamb) * P_aug_temp)
                chi_minus = x_aug - np.sqrt((na + lamb) * P_aug_temp)
                chi_plus_k_km1.append(f_km1_build(chi_plus[0][0], chi_plus[1][0], chi_plus[2][0]))
                chi_minus_k_km1.append(f_km1_build(chi_minus[0][0], chi_minus[1][0], chi_minus[2][0]))
                w_vec.append(1 / (2 * (na + lamb)))
            # Prior state
            x_k_km1 = w0m * x_aug[0:3]
            for (chi_p_k_km1, chi_m_k_km1, w) in zip(chi_plus_k_km1, chi_minus_k_km1, w_vec):
                x_k_km1 += w * chi_p_k_km1[0:3] + w * chi_m_k_km1[0:3]
            # Prior covariance
            P_k_km1 = w0c * (x_aug[0:3] - x_k_km1) @ (x_aug[0:3] - x_k_km1).T
            for (chi_p_k_km1, chi_m_k_km1, w) in zip(chi_plus_k_km1, chi_minus_k_km1, w_vec):
                P_k_km1 = w * (chi_p_k_km1[0:3] - x_k_km1) @ (chi_p_k_km1[0:3] - x_k_km1).T \
                        + w * (chi_m_k_km1[0:3] - x_k_km1) @ (chi_m_k_km1[0:3] - x_k_km1).T
            
            # Correction Step
            # Expected measurement
            y_k_est = w0m * np.sqrt(d**2 + x_k_km1[0][0]**2) 
            for (chi_p_k_km1, chi_m_k_km1, w) in zip(chi_plus_k_km1, chi_minus_k_km1, w_vec):
                y_k_est += w * np.sqrt(d**2 + chi_p_k_km1[0][0]**2) \
                           + w * np.sqrt(d**2 + chi_m_k_km1[0][0]**2)
            # Approximate innovation covariance
            P_y = w0c * (np.sqrt(d**2 + x_k_km1[0][0]**2) - y_k_est) \
                        * (np.sqrt(d**2 + x_k_km1[0][0]**2) - y_k_est).T
            # Approximate state-measurement cross-covariance
            P_xy = w0c * (x_aug[0:3] - x_k_km1) \
                         * (np.sqrt(d**2 + x_k_km1[0][0]**2) - y_k_est).T
            for (chi_p_k_km1, chi_m_k_km1, w) in zip(chi_plus_k_km1, chi_minus_k_km1, w_vec):
                P_y += w * (np.sqrt(d**2 + chi_p_k_km1[0][0]**2) - y_k_est) \
                        * (np.sqrt(d**2 + chi_p_k_km1[0][0]**2) - y_k_est).T
                P_y += w * (np.sqrt(d**2 + chi_m_k_km1[0][0]**2) - y_k_est) \
                        * (np.sqrt(d**2 + chi_m_k_km1[0][0]**2) - y_k_est).T
                P_xy += w * (chi_p_k_km1[0:3] - x_k_km1) \
                         * (np.sqrt(d**2 + chi_p_k_km1[0][0]**2) - y_k_est).T
                P_xy += w * (chi_m_k_km1[0:3] - x_k_km1) \
                         * (np.sqrt(d**2 + chi_m_k_km1[0][0]**2) - y_k_est).T
                pass
            K_k = P_xy * P_y**-1
            x_k_k = x_k_km1 + K_k * (y_k - y_k_est)
            P_k_k = P_k_km1 - K_k * P_y @ K_k.T

            # Saving and reseting values
            x_hist[i, :] = np.atleast_2d(x_k_k).T
            x_km1_km1 = x_k_k
            P_km1_km1 = P_k_k

    return x_hist

def make_pretty_plot(time, x_hist, h_k, s_k, Cb_k):
    '''
    Temp
    '''
    _, ax = plt.subplots((3))
    ax[0].plot(time, x_hist[:, 0], color='g', label='Estimate')
    ax[0].plot(time, h_k, color='y', label='True Value')
    ax[1].plot(time, x_hist[:, 1], color='g', label='Estimate')
    ax[1].plot(time, s_k, color='y', label='True Value')
    ax[2].plot(time, x_hist[:, 2], color='g', label='Estimate')
    ax[2].plot(time, Cb_k, color='y', label='True Value')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()
    return 0

def main():
    '''
    Temp
    '''
    # Load Data
    data = np.loadtxt('altimeter_data.csv', delimiter=',')
    time = data[:, 0]
    y_k = data[:, 1]                    # Current range measurement
    h_k = data[:, 2]                    # True altitude
    s_k = data[:, 3]                    # True vertical velocity
    Cb_k = data[:, 4]                   # True ballistic coefficient
    # Constants Definition
    global dt, rho0, g0, hp, RE, d
    dt = 0.5                                    # s
    rho0 = 0.0765                               # lb/ft^3
    g0 = 32.2                                   # ft/s^2
    hp = 30_000                                 # ft
    RE = 20_902_260                             # ft 
    d = 100_000                                 # ft
    # Matrices and Initial Conditions
    Q = np.diagflat((10**2, 10**2, 0.05**2))    # 
    R = 100**2                                  # ft^2
    x0 = np.array([[400_000], [-2_000], [20]])  # [[ft], [ft/s], [lb/ft^2]]
    P0 = np.diagflat((100**2, 10**2, 1**2))     # [ft^2, ft^2/s^2, lb^2/ft^2]

    # EKF_x_hist = EKF(y_k, x0, P0, Q, R)
    SP_UKF_x_hist = SP_UKF(y_k, x0, P0, Q, R)

    # make_pretty_plot(time, EKF_x_hist, h_k, s_k, Cb_k)
    make_pretty_plot(time, SP_UKF_x_hist, h_k, s_k, Cb_k)

    return 0

if __name__=='__main__':
    main()