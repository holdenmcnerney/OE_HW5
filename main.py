# Optimal Estimation - HW5 - Ballistic Vehicle Altimetry System Design

import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, uniform

def f_km1_build(h_km1, s_km1, Cb_km1):
    '''
    Builds nonlinear f result.
    '''
    f_km1 = np.array([[h_km1 + dt * s_km1], 
                     [s_km1 + dt * (rho0 * s_km1**2 / (2 * Cb_km1) * np.exp(-h_km1 / hp)
                                    - g0 * (RE / (RE + h_km1))**2)], 
                     [Cb_km1]])
    return f_km1

def F_km1_build(h_km1, s_km1, Cb_km1):
    '''
    Builds linear F matrix result.
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
    Builds nonlinear H result.
    '''
    H_km1 = np.array([h_km1 * (d**2 + h_km1**2)**(-1/2), 0, 0])
    return H_km1

def calc_sg_pts(x_aug, P_aug):
    '''
    Calculates sigma points.
    '''
    alpha = 1
    beta = 2
    kappa = 1
    na = P_aug.shape[0]
    lamb = alpha**2 * (na + kappa) - na
    sg_pts = []
    wm_vec = []
    wc_vec = []
    sg_pts.append(x_aug)
    wm_vec.append(lamb / (na + lamb))
    wc_vec.append(lamb / (na + lamb) + (1 - alpha**2 + beta))
    P_aug_sqrt = spla.sqrtm((na + lamb) * P_aug)
    for i in np.arange(0, na):
        P_j_col = np.atleast_2d(P_aug_sqrt[:, i]).T
        sg_p = x_aug + P_j_col
        sg_pts.append(sg_p)
        sg_m = x_aug - P_j_col
        sg_pts.append(sg_m)
        wm_vec.append(1 / (2 * (na + lamb)))
        wc_vec.append(1 / (2 * (na + lamb)))
        wm_vec.append(1 / (2 * (na + lamb)))
        wc_vec.append(1 / (2 * (na + lamb)))
    return sg_pts, wm_vec, wc_vec

def prop_sg_pts(sg_pts):
    '''
    Propogates sigma points through nonlinear dynamics equations
    '''
    p_sg_pts = []
    for sg_pt in sg_pts:
        h = sg_pt[0][0]
        s = sg_pt[1][0]
        Cb = sg_pt[2][0]
        p_sg_pts.append(f_km1_build(h, s, Cb))
    return p_sg_pts

def EKF(y, x0, P0, Q, R):
    '''
    Runs an extended kalman filter and output state trajectory history.
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
            P_k_k = P_k_km1 - K_k @ S_k * K_k.T
            # Saving and reseting values
            x_hist[i, :] = np.atleast_2d(x_k_k).T
            x_km1_km1 = x_k_k
            P_km1_km1 = P_k_k
    return x_hist

def SP_UKF(y, x0, P0, Q, R):
    '''
    Runs a sigma point unscented kalman filter and output state trajectory 
    history.
    '''
    x_hist = np.zeros((len(y), 3))
    x_km1_km1 = x0
    P_km1_km1 = P0
    for i, y_k in enumerate(y):
        if i == 0:
            x_hist[0, :] = np.atleast_2d(x0).T
        else:
            # Prediction Step
            x_aug = np.block([[x_km1_km1], [np.zeros((3, 1))], [0]])
            P_aug = np.block([[P_km1_km1, np.zeros((3, 4))], 
                              [np.zeros((3, 3)), Q, np.zeros((3, 1))], 
                              [np.zeros((1, 6)), R]])
            # Prior state sigma points
            sg_pts, wm_vec, wc_vec = calc_sg_pts(x_aug, P_aug)
            # Prior state
            x_k_km1 = 0
            for (sg_pt, w) in zip(sg_pts, wm_vec):
                x_k_km1 += w * np.real(sg_pt[0:3])
            # Propogate sigma point states
            p_sg_pts = prop_sg_pts(sg_pts)
            # Prior covariance
            P_k_km1 = np.zeros((3, 3))
            for (sg_pt, w) in zip(p_sg_pts, wc_vec):
                P_k_km1 += w * (sg_pt[0:3] - x_k_km1) @ (sg_pt[0:3] - x_k_km1).T
            # Correction Step
            x_aug_c = np.block([[x_k_km1], [np.zeros((3, 1))], [0]])
            P_aug_c = np.block([[P_k_km1, np.zeros((3, 4))], 
                                [np.zeros((3, 3)), Q, np.zeros((3, 1))], 
                                [np.zeros((1, 6)), R]])
            # Generate new sigma points
            sg_pts_c, wm_vec_c, wc_vec_c = calc_sg_pts(x_aug_c, P_aug_c)
            # Expected measurement
            y_k_est = 0
            for (sg_pt_c, w_c) in zip(sg_pts_c, wm_vec_c):
                y_k_est += w_c * np.sqrt(d**2 + sg_pt_c[0][0]**2)
            # Approximate innovation covariance
            P_y = 0
            # Approximate state-measurement cross-covariance
            P_xy = np.zeros((3, 1))
            for (sg_pt_c, w_c) in zip(sg_pts_c, wc_vec_c):
                P_y += w_c * (np.sqrt(d**2 + sg_pt_c[0][0]**2) - y_k_est) \
                        * (np.sqrt(d**2 + sg_pt_c[0][0]**2) - y_k_est).T
                P_xy += w_c * (sg_pt_c[0:3] - x_k_km1) \
                         * (np.sqrt(d**2 + sg_pt_c[0][0]**2) - y_k_est).T
                pass
            K_k = P_xy * P_y**-1
            x_k_k = x_k_km1 + K_k * (y_k - y_k_est)
            P_k_k = P_k_km1 - K_k * P_y @ K_k.T
            # Saving and reseting values
            x_hist[i, :] = np.atleast_2d(x_k_k).T
            x_km1_km1 = x_k_k
            P_km1_km1 = P_k_k
    return x_hist

def BPF(y, x0, P0, Q, R):
    '''
    Runs a bootstrap particle filter and output state trajectory history.
    '''
    x_hist = np.zeros((len(y), 3))
    x_km1_km1 = x0
    num_p = 10
    for i, y_k in enumerate(y):
        if i == 0:
            x_hist[0, :] = np.atleast_2d(x0).T
            w_vec = [1/num_p] * num_p
            x_km1_km1_vec = [x_km1_km1] * num_p
        else:
            # Sample
            wk_vec = []
            x_k_km1_vec = []
            for x_km1_km1 in x_km1_km1_vec:
                x_k_km1 = f_km1_build(x_km1_km1[0][0], x_km1_km1[1][0], x_km1_km1[2][0])
                x_k_km1_list = [x[0] for x in list(x_k_km1)]
                x_k_km1_samp = multivariate_normal.rvs(mean=x_k_km1_list, cov=Q)
                x_k_km1_vec.append(np.atleast_2d(x_k_km1_samp).T)
            # Compute weights
            for x_k_km1, w in zip(x_k_km1_vec, w_vec):
                y_diff = y_k - np.sqrt(d**2 + x_k_km1[0]**2)
                rand_samp = multivariate_normal.rvs(mean=y_diff, cov=R)
                wk_vec.append(rand_samp * w)
            # Normalize weights
            wk_sum = sum(wk_vec)
            wk_norm_vec = []
            for wk in wk_vec:
                wk_norm_vec.append(wk / wk_sum)
            # Resample
            x_k_k_vec = []
            for _ in range(0, num_p):
                ri = uniform.rvs()
                for j in range(0, num_p):
                    if sum(wk_norm_vec[0:j]) >= ri:
                        x_k_k_vec.append(x_k_km1_vec[j])
                        break
            w_vec = [1/num_p] * num_p
            # Fix to losing particles while resampling
            if len(x_k_k_vec) < num_p:
                for _ in range(0, num_p - len(x_k_k_vec)):
                    x_k_k_vec.append(x_k_km1_vec[0])
            # Output optimal state estimate
            x_PF = 1 / num_p * sum(x_k_k_vec)
            P_sum = np.zeros((3, 3))
            for x_k in x_k_k_vec:
                P_sum += (x_k - x_PF) @ (x_k - x_PF).T
            # P_PF = 1 / (num_p - 1) * P_sum
            # Saving and reseting values
            x_hist[i, :] = np.atleast_2d(x_PF).T
            x_km1_km1 = x_PF
            x_km1_km1_vec = x_k_k_vec
    return x_hist

def make_pretty_plot(filter, time, x_hist, h_k, s_k, Cb_k):
    '''
    Makes plots for comparing state trajectory history to true values.
    '''
    fig, ax = plt.subplots((3), figsize=(10, 8))
    fig.align_ylabels()
    fig.suptitle(filter)
    ax[0].set_ylabel(r'Altitude (h), $ft$')
    ax[0].plot(time, x_hist[:, 0], color='g', label=f'''Estimate''')
    ax[0].plot(time, h_k, color='y', label='True Value')
    ax[1].set_ylabel(r'Vertical Velocity (s), $ft/s$')
    ax[1].plot(time, x_hist[:, 1], color='g', label='Estimate')
    ax[1].plot(time, s_k, color='y', label='True Value')
    ax[2].set_ylabel(r'Ballistic Cofficient ($C_b$), $lb/ft^2$')
    ax[2].set_xlabel(r'Time, $s$')
    ax[2].plot(time, x_hist[:, 2], color='g', label='Estimate')
    ax[2].plot(time, Cb_k, color='y', label='True Value')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()
    return 0

def error_comp_plots(time, h_k, s_k, Cb_k, EKF_x_hist, SP_UKF_x_hist, BPF_x_hist):
    '''
    Makes plots for comparing state trajectory error history between filters.
    '''
    fig, ax = plt.subplots((3), figsize=(10, 8))
    fig.align_ylabels()
    fig.suptitle('State Error Across 3 Filters')
    ax[0].set_ylabel(r'Altitude Error (h), $ft$')
    ax[0].plot(time, EKF_x_hist[:, 0] - h_k, color='r', label='EFK Error')
    ax[0].plot(time, SP_UKF_x_hist[:, 0] - h_k, color='g', label='SPF Error')
    ax[0].plot(time, BPF_x_hist[:, 0] - h_k, color='y', label='BPF Error')
    ax[1].set_ylabel(r'Vertical Velocity Error (s), $ft/s$')
    ax[1].plot(time, EKF_x_hist[:, 1] - s_k, color='r', label='EFK Error')
    ax[1].plot(time, SP_UKF_x_hist[:, 1] - s_k, color='g', label='SPF Error')
    ax[1].plot(time, BPF_x_hist[:, 1] - s_k, color='y', label='BPF Error')
    ax[2].set_ylabel(r'Ballistic Cofficient Error ($C_b$), $lb/ft^2$')
    ax[2].set_xlabel(r'Time, $s$')
    ax[2].plot(time, EKF_x_hist[:, 2] - Cb_k, color='r', label='EFK Error')
    ax[2].plot(time, SP_UKF_x_hist[:, 2] - Cb_k, color='g', label='SPF Error')
    ax[2].plot(time, BPF_x_hist[:, 2] - Cb_k, color='y', label='BPF Error')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()
    return 0

def main():
    '''
    Main function for setting variables and running filters.
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
    # Generating state trajectories for all filters
    EKF_x_hist = EKF(y_k, x0, P0, Q, R)
    SP_UKF_x_hist = SP_UKF(y_k, x0, P0, Q, R)
    BPF_x_hist = BPF(y_k, x0, P0, Q, R)
    ekf = 'Extended Kalman Filter'
    spf = 'Sigma Point Filter'
    bpf = 'Bootstrap Particle Filter'
    make_pretty_plot(ekf, time, EKF_x_hist, h_k, s_k, Cb_k)
    make_pretty_plot(spf, time, SP_UKF_x_hist, h_k, s_k, Cb_k)
    make_pretty_plot(bpf, time, BPF_x_hist, h_k, s_k, Cb_k)
    error_comp_plots(time, h_k, s_k, Cb_k, EKF_x_hist, SP_UKF_x_hist, BPF_x_hist)
    return 0

if __name__=='__main__':
    main()