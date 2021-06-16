# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:11:01 2020

@author: jkenrick
"""
from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos
import math

##################### SPACECRAFT DYNAMICS AND CONTROL CAPSTONE

r_LMO=3796.19 #km
r_GMO=20424.2 #km
ascension_LMO=np.deg2rad(20) #rad
inlicination_LMO=np.deg2rad(30) #rad
latitude_t0_LMO=np.deg2rad(60) #rad

ascension_GMO=0 #rad
inlicination_GMO=0 #rad
latitude_t0_GMO=np.deg2rad(250) #rad

lat_dot_LMO=0.000884797 #rad/sec
lat_dot_GMO=0.0000709003 #rad/sec

def latitude(lat_t0,lat_dot,t):
    return lat_t0 + lat_dot*t

def tilde(x):
    x = np.squeeze(x)
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]
                     ])

def EA_LMO(t):
    return np.array([latitude(latitude_t0_LMO,lat_dot_LMO,t),inlicination_LMO,ascension_LMO])

def Euler3132C2(q):
    """
    Euler3132C

        C = Euler3132C(Q) returns the direction cosine
        matrix in terms of the 3-1-3 Euler angles.
        Input Q must be a 3x1 vector of Euler angles.
    """

    st1 = math.sin(q[0]);
    ct1 = math.cos(q[0]);
    st2 = math.sin(q[1]);
    ct2 = math.cos(q[1]);
    st3 = math.sin(q[2]);
    ct3 = math.cos(q[2]);

    C = np.array([[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]])
    C[0,0] = ct3*ct1-st3*ct2*st1
    C[0,1] = -(ct3*st1+st3*ct2*ct1)
    C[0,2] = st3*st2
    C[1,0] = -(-st3*ct1-ct3*ct2*st1)
    C[1,1] = -st3*st1+ct3*ct2*ct1
    C[1,2] = -(ct3*st2)
    C[2,0] = st2*st1
    C[2,1] = -(-st2*ct1)
    C[2,2] = ct2

    return C

M_PI = math.pi
D2R  = M_PI/180.
R2D  = 180./M_PI

def r_N_LMO(t):
    A=r_LMO*np.dot(Euler3132C2(EA_LMO(t)),np.array([[1],[0],[0]]))
    return A

def r_N_GMO(t): 
    A=r_GMO*np.array([[math.cos(latitude(latitude_t0_GMO,lat_dot_GMO,t))],[math.sin(latitude(latitude_t0_GMO,lat_dot_GMO,t))],[0]])
    return A

def r_dot_N_LMO(t):
    A=r_LMO*lat_dot_LMO*np.dot(Euler3132C2(EA_LMO(t)),np.array([[0],[1],[0]])) 
    return A

def r_dot_N_GMO(t):
    A=lat_dot_GMO*r_GMO*np.array([[-math.sin(latitude(latitude_t0_GMO,lat_dot_GMO,t))],[math.cos(latitude(latitude_t0_GMO,lat_dot_GMO,t))],[0]])  
    return A

#r_LMO fn too?
r_LMO*np.dot(np.transpose(HN_DCM(ti)),np.array([[1],[0],[0]]))

#spacecraft to inertial frame DCM
def HN_DCM(t):    
    A=np.dot(Euler3132C2(EA_LMO(t)),np.array([[1],[0],[0]]))
    B=np.dot(Euler3132C2(EA_LMO(t)),np.array([[0],[1],[0]]))
    C=np.cross(np.transpose(A),np.transpose(B)) 
    D=np.linalg.norm(C)
    ih=C/D
    ir=np.transpose(A/np.linalg.norm(A))
    i0=np.cross(ih,ir)
    dcm=np.array([[ir[0,0],ir[0,1],ir[0,2]],
                  [i0[0,0],i0[0,1],i0[0,2]],
                  [ih[0,0],ih[0,1],ih[0,2]]])
    return dcm

#sun pointing reference frames
Rs_N_DCM=np.array([[-1,0,0],[0,0,1],[0,1,0]])
omega_Rs_N=np.array([0,0,0])

#nadir poiting reference frame
def Rn_N_DCM(t):
    Rn=HN_DCM(t)*np.array([[-1,-1,-1],[1,1,1],[-1,-1,-1]])
    return Rn

#omega_Rn_N
def omega_Rn_N(t):
    return np.transpose(np.dot(np.transpose(Rn_N_DCM(t)),-np.array([0,0,lat_dot_LMO])))

#GMO pointing reference frame
def Rc_N_DCM(t):
    c1=-(r_N_GMO(t)-r_N_LMO(t))
    c1=np.transpose(c1/np.linalg.norm(c1))
    c2=np.cross(np.transpose((r_N_GMO(t)-r_N_LMO(t))),np.transpose(np.array([[0],[0],[1]])))
    c2=c2/np.linalg.norm(c2)
    c3=np.cross(c1,c2)
    rc=np.array([[c1[0,0],c1[0,1],c1[0,2]],
                  [c2[0,0],c2[0,1],c2[0,2]],
                  [c3[0,0],c3[0,1],c3[0,2]]])
    return rc

def omega_Rc_N(t):
    Rc_N_DCM_dot=(Rc_N_DCM(t)-Rc_N_DCM(t-0.01))/0.01
    omega_tilde=-np.dot(np.transpose(Rc_N_DCM(t)),Rc_N_DCM_dot)
    omega=np.array([omega_tilde[2,1],omega_tilde[0,2],omega_tilde[1,0]])
    return omega
sigma=mrp
def mrp_to_rotation_matrix(sigma):
#    if np.dot(sigma, sigma) > 1:
#        sigma = mrp_shadow(sigma)
    sigma_squared = np.inner(sigma, sigma)
    q0 = (1 - sigma_squared) / (1 + sigma_squared)
    q = [2 * sigma_i / ( 1 + sigma_squared) for sigma_i in sigma]
    q.extend([q0])
#    if 2*math.acos(q0)*R2D >=180:
#        q[0]=-q[0]
#        q[1]=-q[1]
#        q[2]=-q[2]
#        q[3]=-q[3]
    return Rotation.from_quat(q).as_matrix().T

def rotmat_to_mrp(matrix):
    zeta = np.sqrt(np.trace(matrix) + 1)
    constant = 1 / (zeta**2 + 2 * zeta)
    s1 = constant * (matrix[1, 2] - matrix[2, 1])
    s2 = constant * (matrix[2, 0] - matrix[0, 2])
    s3 = constant * (matrix[0, 1] - matrix[1, 0])
    mrp=np.array([s1, s2, s3])
#    if np.dot(mrp, mrp) > 1:
#        mrp = mrp_shadow(mrp)
    return mrp

sigma_B_N_t0=np.array([0.3,-0.4,0.5])
omega_B_N_t0=np.array([np.deg2rad(i) for i in [1,1.75,-2.2]]) #deg/s

#sigma_B_R
def attitude_tracking_error(t,reference,mrp):
    if reference=="nadir":
        RN_DCM=Rn_N_DCM(t)
    if reference=="gmo":
        RN_DCM=Rc_N_DCM(t)
    if reference=="sun":
        RN_DCM=Rs_N_DCM 
    sigma_B_R=rotmat_to_mrp(np.dot(mrp_to_rotation_matrix(mrp),np.transpose(RN_DCM)))
    return sigma_B_R

#w_B_R
t=0
def rates_tracking_error(t,reference,mrp,w):
    if reference=="nadir":
        omega_R_N=-np.array([0,0,lat_dot_LMO])
        RN_DCM=Rn_N_DCM(t)
    if reference=="gmo":
        omega_R_N=omega_Rc_N(t)
        RN_DCM=Rc_N_DCM(t)
    if reference=="sun":
        omega_R_N=omega_Rs_N
        RN_DCM=Rs_N_DCM 
    DCM_B_R=np.dot(mrp_to_rotation_matrix(mrp),np.transpose(RN_DCM))
    omega_B_R=w-np.dot(DCM_B_R,omega_R_N)
    return omega_B_R

#Task qs:
#attitude_tracking_error(0,"nadir",sigma_B_N_t0)
#attitude_tracking_error(0,"gmo",sigma_B_N_t0)
#attitude_tracking_error(0,"sun",sigma_B_N_t0)
#rates_tracking_error(0,"nadir",sigma_B_N_t0,omega_B_N_t0)
#rates_tracking_error(0,"gmo",sigma_B_N_t0,omega_B_N_t0)
#rates_tracking_error(0,"sun",sigma_B_N_t0,omega_B_N_t0)

### Integrator

I1 = 10
I2 = 5
I3 = 7.5 # kg m^2

I = np.array([[I1, 0, 0], [0, I2, 0],[0, 0, I3]])

P1 = 2*I1/120
P2 = 2*I2/120
P3 = 2*I3/120 # kg m^2
P=P1

K1=P**2/I1
K2=P**2/I2
K3=P**2/I3

K=K2
E1=P/(K*I1)**0.5
E2=P/(K*I2)**0.5
E3=P/(K*I3)**0.5

T1=2*I1/P
T2=2*I2/P
T3=2*I3/P

#if all Ts <=120 and all Es<=1 then the right P and K match have been selected
P=np.eye(3)*P

### Utility functions

def mrp_shadow(mrp):
    norm = np.linalg.norm(mrp) ** 2
    return np.array([-i / norm for i in mrp])

def mrp_dot(mrp, w):
    return 0.25 * np.dot(((1 - np.dot(mrp, mrp)) * np.eye(3) + 2 * tilde(mrp) + 2 * np.outer(mrp,  mrp)), w)

def mrp_dot_matrix(mrp):
    ss = np.dot(mrp, mrp)
    A = np.zeros((3,3))
    A[0, 0] = 1 - ss + 2 * mrp[0] **2
    A[1, 0] = 2*(mrp[1] * mrp[0] + mrp[2])
    A[2, 0] = 2*(mrp[2] * mrp[0] - mrp[1])
    A[0, 1] = 2*(mrp[0] * mrp[1] - mrp[2])
    A[1, 1] = 1 - ss + 2 * mrp[1] ** 2
    A[2, 1] = 2*(mrp[2] * mrp[1] + mrp[0])
    A[0, 2] = 2*(mrp[0] * mrp[2] + mrp[1])
    A[1, 2] = 2*(mrp[1] * mrp[2] - mrp[0])
    A[2, 2] = 1 - ss + 2 * mrp[2] ** 2
    return A

def control(t, mrp, w,reference):
    sigma_b_r = attitude_tracking_error(t,reference,mrp)
    w_b_r= rates_tracking_error(t,reference,mrp,w)
    if controlison:
        u = -K * sigma_b_r - np.transpose(np.dot(P, np.transpose(w_b_r)))    
    else:
        u = np.array([0.0,0.0,0.0]).T
    return u

def wdot(t, mrp, w,reference,u):
    w_dot = np.dot(np.linalg.inv(I), np.transpose((-np.cross(w, np.dot(I, w)) + u)))
    return w_dot

reference='sun'
controlison=True

#sigma_B_N
mrp = np.transpose(sigma_B_N_t0)
mrp0=mrp.copy()
mrp_history = [mrp]

#w_B_N
w = omega_B_N_t0
w0=w.copy()
w_history = [w]
h = 0.01
time = 6000
tvec = np.linspace(0, time, int(time/h + 1))
prev_t = 0
u_history=[control(0, mrp, w,reference)]
error_history = [attitude_tracking_error(0,reference,mrp)]
w_error_history = [rates_tracking_error(0,reference,mrp,w)]
reference_history=[reference]
for ti in tvec[1:]:
    theta=math.acos(np.dot(np.transpose(r_N_LMO(ti)),r_N_GMO(ti))/(np.linalg.norm(r_N_LMO(ti))*np.linalg.norm(r_N_GMO(ti))))*R2D
    if r_N_LMO(ti)[1]>=0:
        reference='sun'
    elif theta <= 35:
        reference='gmo'
    else:
        reference='nadir'
    dt = ti - prev_t
    prev_t = ti
    sigma_b_r_t0=sigma_b_r
    sigma_b_r = attitude_tracking_error(ti,reference,mrp)
    error_history.append(sigma_b_r)
    reference_history.append(reference)

    w_b_r = rates_tracking_error(ti,reference,mrp,w)
    w_error_history.append(w_b_r)
    # calculate and apply dots
    mrp = mrp + mrp_dot(mrp, w) * dt
    if np.dot(mrp, mrp) > 1:
        mrp = mrp_shadow(mrp)
    if ti ==0.01:
        u=control(0,mrp,w,reference)
    if ti % 1 ==0:
        u=control(ti,mrp,w,reference)
    w = w + wdot(ti, mrp, w,reference,u) * dt
    
    u_history.append([control(ti, mrp, w,reference)])
    mrp_history.append(mrp)
    w_history.append(w)
    
    if ti % 500 == 0:
        print("Simulated {} seconds".format(ti))

mrp_history = np.array(mrp_history)
mrp_norm = [np.dot(i, i) for i in mrp_history]
w_history = np.array(w_history)
error_history = np.array(error_history)
w_error_history = np.array(w_error_history)
u_history=np.array(u_history)
reference_history=np.array(reference_history)

#Task qs:
#H_B=np.dot(I,w)
#T=0.5*np.dot(np.transpose(w),H_B)
#mrp
#BN_DCM=mrp_to_rotation_matrix(mrp)
#H_N=np.dot(np.transpose(BN_DCM),H_B)

plt.figure(0)
plt.plot(tvec, mrp_history[:, 0], 'g')
#plt.plot(tvec, target_histories[:, 0], 'g--')
plt.plot(tvec, mrp_history[:, 1], 'b')
#plt.plot(tvec, target_histories[:, 1], 'b--')
plt.plot(tvec, mrp_history[:, 2], 'r')
#plt.plot(tvec, target_histories[:, 2], 'r--')
plt.plot(tvec, mrp_norm, 'k')
plt.title('Attitude (sigma) history')
plt.legend(["sigma_1", "sigma_2", "sigma_3", "norm^2"])
plt.grid()
plt.figure(1)
plt.plot(tvec, w_history[:, 0], 'b')
#plt.plot(tvec, target_rate_history[:, 0], 'b--')
plt.plot(tvec, w_history[:, 1], 'r')
#plt.plot(tvec, target_rate_history[:, 1], 'r--')
plt.plot(tvec, w_history[:, 2], 'g')
#plt.plot(tvec, target_rate_history[:, 2], 'g--')
plt.title("Rate (w) history")
plt.legend(["w_1", "w_2",  "w_3"])
plt.grid()
plt.figure(2)
plt.plot(tvec, reference_history, 'b')
plt.title("Reference")
plt.legend(["Ref"])
plt.grid()
plt.show()

mrp_history[300*100]
mrp_history[2100*100]
mrp_history[3400*100]
mrp_history[4400*100]
mrp_history[5600*100]


