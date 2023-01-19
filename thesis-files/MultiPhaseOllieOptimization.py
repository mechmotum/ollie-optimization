#!/usr/bin/env pyth_son3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 17:43:24 2022

@author: j.t.heinaen
"""
#from collections import namedtuple
#from matplotlib.animation import FuncAnimation
import sympy.physics.mechanics as me
import sympy as sm
import pycollo
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import random
import time
# import dill


current_time = time.ctime()[8:-5]

filepath = ''

testcase = 'l_wb'

l_wb, l_t, l_f, phi, d_tr, r_w, m_h, Ih = sm.symbols(
    'l_wb, l_t, l_f, phi, d_tr, r_w,m_h,Ih')


optimized_parameters= (#l_wb,
                      #l_t,
                      #l_f,
                      #phi,
                      #d_tr,
                      #r_w,
                      )

def dz(symbolic, numerical):
    return dict(zip(symbolic, numerical))

def fx(A,repl):
    return float(A.xreplace(repl)) 
def weight(width_deck, mass_bearing, mass_truck, height_truck, height_truck0, 
           width_wheel, n_ply, length_flat, length_tail, radius_wheel,
           rho_pu, rho_maple, rho_steel, m_glue, diameter_axle, d_veneer):

    mass_wheel = rho_pu * sm.pi * width_wheel * \
        ((2*radius_wheel)**2-diameter_axle**2) / 4  # V=pi*h*(D^2-d^2)/4

    mass_axle = sm.pi * (diameter_axle/2)**2 * width_deck * \
        rho_steel  # weight of axle, volume * steel * density

    #    _   _   ___________   _   _
    #  /  | | | |           | | | |  \
    # | 1 | |2| |     3     | |4| | 5 | 6=t1 7=w1 8=t2 9=w2
    #  \ _| |_| |___________| |_| |_ /

    #     1\                  /5
    #      2\_______3________/4
    #        6 \/         \/ 7
    #        8 O 9      10 O 11
    
    # Area of wooden components
    A1 = (1/2) * (1/4) * sm.pi * width_deck**2  # 1/4 pi d^2
    A2 = (length_tail - (width_deck/2)) * width_deck         # l * b
    A3 = length_flat * width_deck           # l * b
    A4 = A2
    A5 = A1
    
    thickness = n_ply*d_veneer
    dV = thickness*rho_maple+(m_glue/2*(n_ply-2))

    m1 = A1 * dV
    m2 = A2 * dV
    m3 = A3 * dV
    m4 = m2
    m5 = m1
    m6 = (mass_truck - mass_axle) * height_truck/height_truck0
    m7 = m6
    m8 = mass_axle
    m9 = 2*mass_wheel
    m10 = m8
    m11 = m9

    mass = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11]
    return mass

def COM(mass, com_points, reference_point):
    # Function gets vector position from reference_point
    # Then assigns COM location to point COM
    # Returns position vector of COM points relative to COM
    r_m = []
    for i, x in enumerate(com_points):
        r_m.append(x.pos_from(reference_point)*mass[i])
    COM_skateboard = me.Point('COM_skateboard')
    COM_skateboard.set_pos(reference_point, (sum(r_m)/sum(mass)))

    return COM_skateboard

def inertia(mass, com_points, com, majordim, shape):
    # Major dim:
    #   - (semi)cylinder;  diameter
    #   - cuboid: [l,h]
    #   - Triangle: [Base, Height]

    I_com = []
    I_steiner = []

    for i in range(len(mass)):
        if shape[i] == 'semicircle':
            I_com.append(((1/4)-(16/(9*sm.pi**2)))*mass[i]*(majordim[i]/2)**2)

        if shape[i] == 'cuboid':
            I_com.append((mass[i]/12) * (majordim[i]
                         [0]**2 + majordim[i][1]**2))

        if shape[i] == 'triangle':
            #I_com.append(0)
            s = sm.sqrt((majordim[i][0]/2)**2+majordim[i][1]**2)
            beta = 2*sm.asin((majordim[i][0]/2)/s)
            I_com.append((mass[i]/2)*s**2*(1-(2/3)*sm.sin(beta)))

        if shape[i] == 'cylinder':
            I_com.append((1/2)*mass[i]*(majordim[i]/2)**2)

        I_steiner.append(sm.trigsimp(mass[i]*d2s(sm.sqrt(com_points[i].pos_from(\
                com).dot(A.x)**2+com_points[i].pos_from(com).dot(A.y)**2)**2)))
    I_tot = sum(I_com)+sum(I_steiner)
    return I_tot, I_com, I_steiner

l_wb, l_t, l_f, phi, d_tr, r_w = sm.symbols('l_wb, l_t, l_f, phi, d_tr, r_w')
g, m_h, Ih, m_l, mu, d_d, m_b, m_t, d_w, n_ply, rho_pu, rho_maple, rho_steel, m_glue, d_axle, d_veneer,d_tr0 = sm.symbols('g, m_h, Ih, m_l, mu, d_d, m_b, m_t, d_w, n_ply, rho_pu, rho_maple, rho_steel, m_glue, d_axle, d_veneer,d_tr0')
g, m_l, mu, I_w = sm.symbols('g, m_l, mu, I_w')
d_d, m_b, m_t, d_w, n_ply = sm.symbols('d_d,m_b,m_t,d_w,n_ply')
s_1, s_2 = sm.symbols('s_1,s_2')

d_com_, m_s_, I_s_ = sm.symbols('d_com_, m_s_, I_s_')

rho_pu, rho_maple, rho_steel, m_glue, d_axle, d_veneer = sm.symbols(
    'rho_pu, rho_maple, rho_steel, m_glue, d_axle,d_veneer')
p = [l_wb, l_t, l_f, phi, d_tr, r_w,#m_s,I_s,
     g, m_h, mu, d_d, m_b, m_t, d_w, n_ply, rho_pu, rho_maple, rho_steel, m_glue, d_axle, d_veneer,d_tr0]
p_vals_jan       = np.array([0.444, 0.13, 0.83-2*0.13, np.deg2rad(20), 0.053, 0.049/2,# 3.6, 0.16, #0.3413428,
                             9.81, 80, 0.8, 0.21, 0.012, 0.366, 0.031, 7, 1130, 705, 7700, 0.210, 0.008, 0.0016,0.053])
p_vals_tobi      = np.array([0.433, 0.12, 0.81-2*0.12, np.deg2rad(18.5), 0.053, 0.052/2,# 3.6, 0.16, #0.3413428,
                             9.81, 80, 0.8, 0.205, 0.012, 0.366, 0.032, 7, 1130, 705, 7700, 0.210, 0.008, 0.0016,0.053])
inch = 0.0254
p_vals_longboard = np.array([23*inch, 5.70*inch, (35.8-2*5.07)*inch, np.deg2rad(10), 0.073, 0.06/2,# 3.6, 0.16, #0.3413428,
                             9.81, 80, 0.8, 9.6*inch, 0.012, 0.366, 0.041, 7, 1130, 705, 7700, 0.210, 0.008, 0.0016,0.053])
p_vals_penny     = np.array([14*inch, 5*inch, (22-5)*inch, np.deg2rad(10),0.065,0.059/2,
                             9.81, 80, 0.2, 5.9*inch, 0.012, 0.366,0.045,7,1130, 705, 7700, 0.210, 0.008, 0.0016,0.053])#https://www.wheelz4kids.com/nl/pennyboard-iron-man.html
#https://skateboardelite.com/what-is-penny-board/
p_jan = dict(zip(p, p_vals_jan))
#p_jan = dict(zip(p, p_vals_tobi))

#p_jan = dict(zip(p, p_vals_longboard))
#p_jan = dict(zip(p, p_vals_penny))
def rand_guess(bounds):
    rand_guess = []
    for b in bounds:
        rand_guess.append(lin_choice(b))
    return rand_guess

def lin_choice(b):
    return [round(random.choice(np.linspace(b[0],b[1],100)),2), round(random.choice(np.linspace(b[0],b[1],100)),2)]

parameter_bounds = []
parameter_guess = []
for i in optimized_parameters:
    if i == l_wb:
        parameter_bounds.append([0.05, 1])
        parameter_guess.append(p_jan[l_wb])
    if i == l_t:
        parameter_bounds.append([0.05, 0.3])
        parameter_guess.append(p_jan[l_t])
    if i == l_f:
        parameter_bounds.append([0.05, 1])
        parameter_guess.append(p_jan[l_f])
    if i == phi:
        parameter_bounds.append([np.deg2rad(0), np.deg2rad(38)])
        parameter_guess.append(p_jan[phi])
    if i == d_tr:
        parameter_bounds.append([0.045, 0.3])
        parameter_guess.append(p_jan[d_tr])
    if i == r_w:
        parameter_bounds.append([0.0125, 0.3])
        parameter_guess.append(p_jan[r_w])
    p_jan.pop(i)


#Dynamic symbols
x_s_, y_s_, th_s_, s_1_, s_2_,x_w_ = me.dynamicsymbols('x_s_,y_s_,th_s_,s_1_,s_2_,x_w_')
#Variables skateboard
x_s, y_s, th_s = sm.symbols('x_s,y_s,th_s')
dx_s, dy_s, dth_s = sm.symbols('dx_s,dy_s,dth_s')
ddx_s, ddy_s, ddth_s = sm.symbols('ddx_s,ddy_s,ddth_s')
#Variables human
x_h, y_h  = sm.symbols('x_h, y_h')
dx_h, dy_h = sm.symbols('dx_h, dy_h') 
ddx_h, ddy_h = sm.symbols('ddx_h, ddy_h')
#
s_1, s_2 = sm.symbols('s_1,s_2')
ds_1, ds_2 = sm.symbols('ds_1,ds_2')
dds_1, dds_2 = sm.symbols('dds_1,dds_2')

x_w,dx_w,ddx_w = sm.symbols('x_w,dx_w,ddx_w')
t = me.dynamicsymbols._t
q_ = sm.Matrix([x_s_, y_s_, th_s_, s_1_, s_2_,x_w_])
dq_ = q_.diff(t)
ddq_ = dq_.diff(t)

dynamic2symbolic = dict(zip(sm.flatten(q_)+sm.flatten(dq_)+sm.flatten(ddq_), [
                        x_s, y_s, th_s, s_1, s_2, x_w, dx_s, dy_s, dth_s, ds_1, ds_2, dx_w, ddx_s, ddy_s, ddth_s, dds_1, dds_2, ddx_w]))
def d2s(A):
    return A.xreplace(dynamic2symbolic)

N, A, B, C = sm.symbols('N, A, B, C', cls=me.ReferenceFrame)

A.orient_axis(N, N.z, th_s_)  # Bodyfixed frame
B.orient_axis(N, N.z, th_s_-phi)
C.orient_axis(N, N.z, th_s_+phi)

O = me.Point('O')
WC1_a = O.locatenew('contact back wheels', x_w_*N.x)
W1_a = WC1_a.locatenew('back wheels', r_w*N.y)
Tr1_a = W1_a.locatenew('back trucks at deck', d_tr*A.y)
mid_a= Tr1_a.locatenew('middle deck', l_wb/2*A.x)

com_a = mid_a.locatenew('centre of mass', -d_com_*A.y)
B0_a = mid_a.locatenew('back pocket', -l_f/2*A.x)
tail_a = B0_a.locatenew('tail', -l_t*B.x)
bf_a = tail_a.locatenew('back foot', s_1*B.x)
ff_a = B0_a.locatenew('front foot', s_2_*A.x)

fp1, fp2 = sm.symbols('fp1,fp2')

# When dx and N are positive, fw is positive and vice versa
# def logistic(N,dx,mu,k):
#     return N*mu*(2/(1+sm.exp(-k*(-dx)))-1)
def logistic(N,dx,mu,k):
     return N*mu*((1-sm.exp(dx/k))/(1+sm.exp(dx/k)))

# fw1 = logistic(fp1,ds_1,mu,10)
# fw2 = logistic(fp2,ds_2,mu,10)

# F1 = fp1*B.y + fw1*B.x
# F2 = fp2*A.y + fw2*A.x
l_xp1, l_xm1, g_k1,l_xp2, l_xm2, g_k2 = sm.symbols('l_xp1, l_xm1, g_k1,l_xp2, l_xm2, g_k2')
# Fs = F1 + F2
mu_c = sm.Symbol('mu_c')
fw1 = l_xp1 - l_xm1
fw2 = l_xp2 - l_xm2

F1 = -fp1*B.y - fw1*B.x
F2 = -fp2*A.y - fw2*A.x

Fs= F1 + F2



#%% TMT phaseA
q_a = sm.Matrix([x_w_,th_s_])
dq_a = q_a.diff(t)
ddq_a= dq_a.diff(t)

pos_a = sm.Matrix([com_a.pos_from(O).dot(N.x),
       com_a.pos_from(O).dot(N.y),
       th_s_])

dx_a = pos_a.diff(t)
Tij_a = dx_a.jacobian(dq_a)
Tji_a = Tij_a.transpose()
ddx_a = dx_a.diff(t)
gk_a  = dx_a.jacobian(q_a)*dq_a

Mij_a  = sm.diag(m_s_,m_s_,I_s_)
TMT_a = Tji_a*Mij_a*Tij_a

Fi_a = sm.Matrix([Fs.dot(N.x), Fs.dot(N.y)-m_s_*g, (bf_a.pos_from(com_a).cross(F1)+ff_a.pos_from(com_a).cross(F2)).dot(N.z)])

Fg_a = Tji_a*(Fi_a-Mij_a*gk_a)

ddq_eq_a = d2s(TMT_a.LUsolve(Fg_a))

accSA = {ddx_w:  d2s(ddq_eq_a[0]),
         ddth_s: d2s(ddq_eq_a[1]),
         }
#%% Free floating acc
#O = me.Point('O')
com_b = O.locatenew('centre of mass', x_s_*N.x+y_s_*N.y)
mid_b = com_b.locatenew('middle deck', d_com_*A.y)
Tr1_b = mid_b.locatenew('back trucks at deck', -l_wb/2*A.x)
Tr2_b = Tr1_b.locatenew('front trucks at deck', l_wb*A.x)

W1_b = Tr1_b.locatenew('back wheels', -d_tr*A.y)
WC1_b = W1_b.locatenew('contact back wheels', -r_w*N.y)
W2_b = Tr2_b.locatenew('front wheels', -d_tr*A.y)
WC2_b = W2_b.locatenew('contact back wheels', -r_w*N.y)


B0_b   = mid_b.locatenew('back pocket', -l_f/2*A.x)
tail_b = B0_b.locatenew('tail', -l_t*B.x)
bf_b   = tail_b.locatenew('back foot', s_1_*B.x)

ff_b = B0_b.locatenew('front foot', s_2_*A.x)
C0_b = B0_b.locatenew('front pocket', l_f*A.x)
nose_b = C0_b.locatenew('nose', l_t*C.x)
objective = WC1_b.locatenew('objective', l_wb/2*A.x)


O.set_vel(N,0)

#%% TMT phaseB
q_b = sm.Matrix([x_s_,y_s_,th_s_])
dq_b = q_b.diff(t)
ddq_b= dq_b.diff(t)

pos_b = sm.Matrix([com_b.pos_from(O).dot(N.x),
       com_b.pos_from(O).dot(N.y),
       th_s_])

dx_b = pos_b.diff(t)
Tij_b = dx_b.jacobian(dq_b)
Tji_b = Tij_b.transpose()
ddx_b = dx_b.diff(t)
gk_b  = dx_b.jacobian(q_b)*dq_b

Mij_b  = sm.diag(m_s_,m_s_,I_s_)
TMT_b = Tji_b*Mij_b*Tij_b
Fi_b = sm.Matrix([Fs.dot(N.x),Fs.dot(N.y)-m_s_*g,(bf_b.pos_from(com_b).cross(F1)+ff_b.pos_from(com_b).cross(F2)).dot(N.z)])
Fg_b = Tji_b*(Fi_b-Mij_b*gk_b)

#(Fi_b == Fg_b)

ddq_eq_b = d2s(TMT_b.LUsolve(Fg_b))

accSB = {ddx_s: ddq_eq_b[0],
         ddy_s: ddq_eq_b[1],
         ddth_s:ddq_eq_b[2],
         }

Va_bf_h = bf_a.vel(N) - (dx_h*N.x+dy_h*N.y)
Va_ff_h = ff_a.vel(N) - (dx_h*N.x+dy_h*N.y)
Vb_bf_h = bf_b.vel(N) - (dx_h*N.x+dy_h*N.y)
Vb_ff_h = ff_b.vel(N) - (dx_h*N.x+dy_h*N.y)

Wl1, Wl2, Wf1, Wf2, Wgs, Wgh = sm.symbols('Wl1, Wl2, Wf1, Wf2, Wgs, Wgh')
Pl1,Pl2,Pf1,Pf2, Pgs, Pgh = sm.symbols('Pl1,Pl2,Pf1,Pf2,Pgs,Pgh')
powerA = {Pl1:d2s(F1.dot(Va_bf_h)),
          Pl2:d2s(F2.dot(Va_ff_h)),   # Va/b = va - vb -> Vs/h = Vs - Vh
          Pf1:d2s(F1.dot(-ds_1*B.x)), # Va/b = Va - Vb -> Vs/f = Vs - (Vs+Vf)
          Pf2:d2s(F2.dot(-ds_2*A.x)),
          Pgs:d2s((-m_s_*g*N.y).dot(d2s(dx_a[1])*N.y)),
          Pgh:d2s((-m_h*g*N.y).dot(dy_h*N.y))}

powerB = {Pl1:d2s(F1.dot(Vb_bf_h)),
          Pl2:d2s(F2.dot(Vb_ff_h)),   # Va/b = va - vb -> Vs/h = Vs - Vh
          Pf1:d2s(F1.dot(-ds_1*B.x)), # Va/b = Va - Vb -> Vs/f = Vs - (Vs+Vf)
          Pf2:d2s(F2.dot(-ds_2*A.x)),
          Pgs:d2s((-m_s_*g*N.y).dot(dy_s*N.y)),
          Pgh:d2s((-m_h*g*N.y).dot(dy_h*N.y))}

#%%
skatemass = weight(d_d, m_b, m_t,d_tr,d_tr0, d_w, n_ply, l_f, l_t, r_w,
                    rho_pu, rho_maple, rho_steel, m_glue, d_axle, d_veneer)

m_s = sum(skatemass)

# COM's of pieces skateboard
# half a circle com is located at 4r/3pi
p1 = tail_b.locatenew('p1', (1-(4/(3*sm.pi))) * (d_d/2) * B.x)
p2 = B0_b.locatenew('p2', -(l_t-d_d/2)/2*B.x)
p3 = mid_b
p4 = C0_b.locatenew('p4', ((l_t-d_d/2)/2)*C.x)
p5 = nose_b.locatenew('p5', -(1 - (4/(3*sm.pi))) * (d_d/2) * C.x)
p6 = W1_b.locatenew('p6', (d_tr/3)*A.y)
p7 = W2_b.locatenew('p7', (d_tr/3)*A.y)
p8 = W1_b
p9 = W1_b
p10 = W2_b
p11 = W2_b
com_points = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]
COM_skateboard = COM(
    skatemass, [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11], mid_b)

shape = ['semicircle', 'cuboid', 'cuboid', 'cuboid', 'semicircle',
          'triangle', 'triangle', 'cylinder', 'cylinder', 'cylinder', 'cylinder']
h_d = n_ply*d_veneer
majordim = [d_d, [l_t-d_d/2, h_d], [l_f, h_d], [l_t-d_d/2, h_d],
            d_d, [0.053, d_tr], [0.053, d_tr], d_axle, 2*r_w, d_axle, 2*r_w]
I_s, I_com, I_steiner = inertia(skatemass, com_points, com_b, majordim, shape)

d_com = d2s(sm.trigsimp(mid_b.pos_from(COM_skateboard).dot(A.y)))

mid_b.set_pos(com_b, d_com*A.y)

I_s = I_s.xreplace({d_com_:d_com})

quicksolve = {d_com_: d_com, m_s_: d2s(m_s), I_s_: d2s(I_s)}



accH    = {ddx_h: d2s(-Fs.dot(N.x))/m_h,
            ddy_h: d2s(-Fs.dot(N.y)-m_h*g)/m_h}

#%%
th_impact = -sm.atan(tail_b.pos_from(WC1_b).dot(N.y)/tail_b.pos_from(WC1_b).dot(N.x))
th_impact = th_impact.xreplace({th_s_: 0})
dfp1,dfp2 = sm.symbols('dfp1,dfp2')

Vmax = [-50, 50]
Amax = [-10, 10]

Fmax = 32.61*m_h
Fmin = -Fmax
RFD_ecc = 196.41*m_h
RFD_unl = -41.8*m_h
concentric_displacement = 0.21
Pmax = 66.55*m_h

Reach_physical_max = np.array([0.466, 1.13]) 
Reach = Reach_physical_max[-1]+[-concentric_displacement,0]

#SKATEBOARD
#A
skate_variablesA    = [x_w , th_s , s_1, s_2, 
                      dx_w, dth_s, ds_1, ds_2]
skate_equationsA    = [dx_w, dth_s, ds_1, ds_2,
                      ddx_w, ddth_s, dds_1, dds_2]
As_init             = [[-1,0], 0, [0,1], [0,1],
                       0, 0, 0, 0]  
As_final            = [[-2,1], [0,np.pi/2], [0,1], [0,1],
                       Vmax  , Vmax  , Vmax, Vmax]
As_bounds           = [[-2,1], [0,np.pi/2], [0,1], [0,1],
                       Vmax  , [0,50], Vmax, Vmax]
#B
skate_variablesB    = [x_s , y_s , th_s , s_1 , s_2,
                       dx_s, dy_s, dth_s, ds_1 , ds_2]
skate_equationsB    = [dx_s , dy_s , dth_s , ds_1 , ds_2,
                       ddx_s, ddy_s, ddth_s, dds_1 , dds_2]
Bs_init             = [[-1,1], [0,2], [0,np.pi/2] , [0,1], [0,1],
                       Vmax  , Vmax , Vmax, Vmax  , Vmax]
Bs_final            = [[-1,1], [0,5], 0           , [0,1], [0,1],
                       Vmax  , 0    , Vmax, Vmax  , Vmax]
Bs_bounds           = [[-1,1], [0,5], [-np.pi/2,np.pi/4] , [0,1], [0,1],
                       Vmax  , Vmax , Vmax, Vmax  , Vmax]
Cs_init             = Bs_final
Cs_final            = [[-1,1], [0,1], [0,np.pi/6] , [0,1], [0,1],
                       Vmax  , Vmax , Vmax        , Vmax  , Vmax]
Cs_bounds           = [[-1,1], [0,5], [-np.pi/2,np.pi/4] , [0,1], [0,1],
                       Vmax  , Vmax , Vmax,               Vmax  , Vmax]

#HUMAN
human_variables     = [x_h, y_h ,dx_h, dy_h]
human_equations     = [dx_h,dy_h,ddx_h,ddy_h]
Ah_init             = [0, [0,2], 0, 0]
Ah_final            = [[-1,1], [0,2], Vmax,  Vmax]
Ah_bounds           = [[-1,1], [0,5], Vmax, Vmax]
Bh_init             = Ah_final
Bh_final            = [[-1,1], [0,5],  Vmax,  Vmax]
Bh_bounds           = [[-1,1], [0,5],  Vmax,  Vmax]
Ch_init             = Bh_final
Ch_final            = [[-1,1], [0,5],  Vmax, Vmax]
Ch_bounds           = [[-1,1], [0,5],  Vmax,  Vmax]

Fn,dFn = sm.symbols('Fn,dFn')
work_variables      = [Fn]
work_equations      = [dFn]
Aw_init             = [[Fmin,Fmax]]
Aw_final            = [[Fmin,Fmax]]
Aw_bounds           = [[Fmin,Fmax]]
Bw_init             = Aw_final
Bw_final            = [[Fmin,Fmax]]
Bw_bounds           = [[Fmin,Fmax]]
Cw_init             = Bw_final
Cw_final            = [[Fmin,Fmax]]
Cw_bounds           = [[Fmin,Fmax]]

#CONTROL
A_control           = [dFn, fp1, fp2, dds_1, dds_2]
A_control_bounds    = [[RFD_unl,RFD_ecc], [0,10000], [0,10000], Amax,Amax]
B_control           = A_control
B_control_bounds    = [[-10000,10000],[0,10000], [0,10000], Amax,Amax]
C_control           = B_control
C_control_bounds    = B_control_bounds

friction_control    = [l_xm1,l_xp1,g_k1,l_xm2,l_xp2,g_k2]
friction_bounds     = [[0,Fmax],[0,Fmax],[0,10], [0,Fmax],[0,Fmax],[0,10]]

#Time
A_time_init         = 0
A_time_final        = [0, 1.5]
B_time_init         = A_time_final
B_time_final        = [0, 2.5]
C_time_init         = B_time_final
C_time_final        = [0, 3.5]


#%%
# Phase A
problem = pycollo.OptimalControlProblem(
    name="ollie",
)
phase_A = problem.new_phase(name="A")
phase_A.state_variables                   = skate_variablesA + human_variables + work_variables
phase_A.bounds.initial_state_constraints  = As_init + Ah_init + Aw_init
phase_A.bounds.final_state_constraints    = As_final + Ah_final + Aw_final
phase_A.bounds.state_variables            = As_bounds + Ah_bounds + Aw_bounds
phase_A.state_equations                   = skate_equationsA + human_equations + work_equations
phase_A.control_variables                 = A_control + friction_control
phase_A.bounds.control_variables          = A_control_bounds + friction_bounds

#phase_A.integrand_functions               = [powerA[Pl1]+powerA[Pl2]]
#phase_A.bounds.integral_variables         = [[-3.04*m_h,(8.04-3.04)*m_h]]
#phase_A.guess.integral_variables          = [0]

phase_A.guess.state_variables             = len(phase_A.state_variables)*[[0.01, 0.01]]
phase_A.guess.control_variables           = len(phase_A.control_variables)*[[0.01, 0.01]]

phase_A.bounds.initial_time               = A_time_init
phase_A.bounds.final_time                 = A_time_final
phase_A.guess.time                        = np.array(A_time_final)
phase_A.auxiliary_data                    = accSA|accH#|powerA

phase_B = problem.new_phase(name="B")
phase_B.state_variables                   = skate_variablesB + human_variables + work_variables  # [:-2]
phase_B.bounds.initial_state_constraints  = Bs_init + Bh_init + Bw_init
phase_B.bounds.final_state_constraints    = Bs_final + Bh_final + Bw_final
phase_B.bounds.state_variables            = Bs_bounds + Bh_bounds + Bw_bounds
phase_B.state_equations                   = skate_equationsB + human_equations + work_equations# [:-2]
phase_B.control_variables                 = B_control + friction_control
phase_B.bounds.control_variables          = B_control_bounds + friction_bounds

phase_B.guess.state_variables             = len(phase_B.state_variables)*[[0.01, 0.01]]
phase_B.guess.control_variables           = len(phase_B.control_variables)*[[0.01, 0.01]]

phase_B.bounds.initial_time               = B_time_init
phase_B.bounds.final_time                 = B_time_final
phase_B.guess.time                        = np.array(B_time_final)
phase_B.auxiliary_data                    = accSB|accH#|powerB

phase_C = problem.new_phase(name="C")
phase_C.state_variables                   = skate_variablesB+ human_variables + work_variables
phase_C.bounds.initial_state_constraints  = Cs_init + Ch_init + Cw_init
phase_C.bounds.final_state_constraints    = Cs_final + Ch_final + Cw_final
phase_C.bounds.state_variables            = Cs_bounds + Ch_bounds + Cw_bounds
phase_C.state_equations                   = skate_equationsB + human_equations + work_variables
phase_C.control_variables                 = C_control + friction_control
phase_C.bounds.control_variables          = C_control_bounds + friction_bounds

phase_C.guess.state_variables             = len(phase_C.state_variables)*[[0.01, 0.01]]
phase_C.guess.control_variables           = len(phase_C.control_variables)*[[0.01, 0.01]]

phase_C.bounds.initial_time               = C_time_init
phase_C.bounds.final_time                 = C_time_final
phase_C.guess.time                        = np.array(C_time_final)
phase_C.auxiliary_data                    = accSB|accH#|powerB
def KE_A(S):
    V = (1/2)*m_s*S.x_w**2 + (1/2)* (I_s+m_s*W1_a.pos_from(com_a).magnitude()**2)*S.dth_s**2 + (1/2)*m_h*sm.sqrt(S.dy_h**2 +  S.dx_h**2)**2 
    return V.xreplace(quicksolve)
def KE_B(S):
    V = (1/2)*m_s*sm.sqrt(S.dx_s**2 + S.dy_s**2)**2 + (1/2) * I_s* S.dth_s**2 + \
        (1/2)*m_h*sm.sqrt(S.dy_h**2+  S.dx_h**2)**2 
    return V
def PE_A(S):
    y_sa = l_wb*sm.sin(S.th_s)/2 + r_w + (-d_com_ + d_tr)*sm.cos(S.th_s)
    P = m_h*g*S.y_h + m_s*g*y_sa
    return P
def PE_B(S):
    P = m_h*g*S.y_h + m_s*g*S.y_s    
    return P

#%%
Ai = phase_A.initial_state_variables
Af = phase_A.final_state_variables
Bi = phase_B.initial_state_variables
Cf = phase_C.final_state_variables

eps = 1

impact_loss = KE_B(Bi)-KE_A(Af)
phase_A.path_constraints                  = (#Human distances
                                             y_h-d2s(bf_a.pos_from(O).dot(N.y)),    
                                             y_h-d2s(ff_a.pos_from(O).dot(N.y)),
                                             x_h-d2s(com_a.pos_from(O).dot(N.x)),
                                             s_1-l_t, s_2-l_f, #l_f-l_wb,# d_tr-r_w, 
                                             d2s(sm.trigsimp(ff_a.pos_from(bf_a).magnitude())),
                                             
                                             #Forces
                                             d2s(F1.dot(N.x)),                      
                                             d2s(F2.dot(N.x)),
                                             d2s(Fs.dot(N.y)+Fn),
                                             
                                             #Power
                                             powerA[Pl1]+powerA[Pl2], 
                                             powerA[Pl1]-powerA[Pl2], 
                                             powerA[Pl2]-powerA[Pl1], 
                                             #Wl1+Wl2,
                                             
                                             #Friction
                                             mu*fp1-l_xp1-l_xm1,                    
                                             g_k1+ds_1,
                                             g_k1-ds_1,
                                             (mu*fp1-l_xp1-l_xm1)*g_k1,
                                             (g_k1+ds_1)*l_xp1,
                                             (g_k1-ds_1)*l_xm1,
                                             
                                             mu*fp2-l_xp2-l_xm2,
                                             g_k2+ds_2,
                                             g_k2-ds_2,
                                             (mu*fp2-l_xp2-l_xm2)*g_k2,
                                             (g_k2+ds_2)*l_xp2,
                                             (g_k2-ds_2)*l_xm2,
                                             )

phase_A.bounds.path_constraints           = (#Human distances
                                             Reach,                    
                                             Reach,
                                             [-0.3,0.3],
                                             [-2,0],[-2,0], #[0,10], #[0,10],
                                             [0.1,1],
                                             
                                             #Forces            
                                             [-200,200],                            
                                             [-200,200],
                                             0,
                                             
                                             #Power
                                             [-54.62*m_h,54.62*m_h],
                                             [-54.62*m_h,54.62*m_h],
                                             [-54.62*m_h,54.62*m_h],

                                             #[-3.04*m_h,(8.04-3.04)*m_h],
                                             
                                             #Friction
                                             [0,10000],
                                             [0,10000],
                                             [0,10000],    
                                             [-0.1,0.1],[-eps,eps],[-eps,eps],
                                             
                                             [0,10000],
                                             [0,10000],
                                             [0,10000],              
                                             [-0.1,0.1],[-eps,eps],[-eps,eps],
                                             )
#%%
phase_B.path_constraints                  = tuple()
phase_B.bounds.path_constraints           = tuple()
phase_C.path_constraints                  = tuple()
phase_C.bounds.path_constraints           = tuple()

extra_path_constraints                    = (#Human distance
                                             y_h-d2s(bf_b.pos_from(O).dot(N.y)),
                                             y_h-d2s(ff_b.pos_from(O).dot(N.y)),
                                             x_h-x_s,
                                             s_1-l_t, s_2-l_f, #l_f-l_wb,#d_tr-r_w, 
                                             d2s(sm.trigsimp(ff_b.pos_from(bf_b).magnitude())),
                                             
                                             #Skateboard distance
                                             d2s(WC2_b.pos_from(O).dot(N.y)),
                                             d2s(nose_b.pos_from(O).dot(N.y)),
                                             
                                             #Forces
                                             d2s(F1.dot(N.x)),
                                             d2s(F2.dot(N.x)),
                                             d2s(Fs.dot(N.y)+Fn),

                                             #Power

                                             #Friction                                                   
                                             mu*fp1-l_xp1-l_xm1,
                                             g_k1+ds_1,
                                             g_k1-ds_1,
                                             (mu*fp1-l_xp1-l_xm1)*g_k1,
                                             (g_k1+ds_1)*l_xp1,
                                             (g_k1-ds_1)*l_xm1,
                                             
                                             mu*fp2-l_xp2-l_xm2,
                                             g_k2+ds_2,
                                             g_k2-ds_2,
                                             (mu*fp2-l_xp2-l_xm2)*g_k2,
                                             (g_k2+ds_2)*l_xp2,
                                             (g_k2-ds_2)*l_xm2,
                                             )
extra_path_bound                          = (#Human distances
                                             Reach_physical_max,
                                             Reach_physical_max,
                                             [-0.3,0.3],
                                             [-2,0],[-2,0], #[0,10], #[0,10],
                                             [0.1,1],
                                             
                                             #Skateboard distances
                                             [0,10],
                                             [0,10],
                                             
                                             #Forces
                                             [-200,200],
                                             [-200,200],
                                             0,
                                             
                                             #Power
                                             
                                             #Friction
                                             [0,10000],
                                             [0,10000],
                                             [0,10000],               
                                             [-0.1,0.1],[-eps,eps],[-eps,eps],
                                             
                                             [0,10000],
                                             [0,10000],
                                             [0,10000],             
                                             [-0.1,0.1],[-eps,eps],[-eps,eps],
                                             )
for i in [phase_B,phase_C]:
    i.path_constraints = i.path_constraints+extra_path_constraints
    i.bounds.path_constraints = i.bounds.path_constraints+extra_path_bound

#%%
equalendpoints1 = []
for finalA, initB in zip(phase_A.final_state_variables[1:4]+phase_A.final_state_variables[6:-1],phase_B.initial_state_variables[2:5]+phase_B.initial_state_variables[8:-1]):
    equalendpoints1.append(finalA-initB)
    
equalendpoints2 = []
for finalB, initC in zip(phase_B.final_state_variables[:-1],phase_C.initial_state_variables[:-1]):
    equalendpoints2.append(finalB-initC)
#%%


Mij  = sm.diag(m_s_,m_s_,I_s_)
# Impact at vertical wall at 1
Cw   = sm.Matrix([tail_b.pos_from(O).dot(N.y)])#.xreplace()
Cwi  = Cw.jacobian([x_s_,y_s_,th_s_]).xreplace({th_s_:Af.th_s})
Cwj  = Cwi.transpose()

rho_c,e = sm.symbols('rho_c,e')
dx,dy,dth,th = sm.symbols('dx,dy,dth,th')
Vimp_lhs = sm.Matrix([[Mij, Cwj],[Cwi,sm.zeros(1)]])*sm.Matrix([Bi.dx_s,Bi.dy_s,Bi.dth_s,rho_c])

dictA_Af = dict(zip(phase_A.state_variables,phase_A.final_state_variables))

v_com_Af = sm.Matrix([d2s(dx_a[0]).xreplace(dictA_Af),
                          d2s(dx_a[1]).xreplace(dictA_Af),
                          Af.dth_s])

Vimp_rhs = sm.Matrix([Mij*v_com_Af,-e*Cwi*v_com_Af])
impact_constraint = Vimp_lhs-Vimp_rhs

#%%
problem.endpoint_constraints = [phase_A.final_time_variable-phase_B.initial_time_variable,         
                                phase_B.final_time_variable-phase_C.initial_time_variable,          

                                phase_A.final_state_variables.th_s - th_impact,

                                phase_A.initial_state_variables.x_w+l_wb/2,                  #y_0 calculated when wheel touches ground

                                impact_constraint[0],
                                impact_constraint[1],
                                impact_constraint[2],
                                impact_constraint[3],
                                
                                phase_A.initial_state_variables.Fn - m_h*g,
                                #m_s*g*(l_wb*sm.sin(Ai.th_s)/2 + r_w + (-d_com_ + d_tr)*sm.cos(Ai.th_s))+m_h*g*Ai.y_h - 600,
                                
                                d2s(com_a.pos_from(O).dot(N.x)).xreplace(dictA_Af) - Bi.x_s,
                                d2s(com_a.pos_from(O).dot(N.y)).xreplace(dictA_Af) - Bi.y_s,

                                d2s(WC1_b.pos_from(O).dot(N.y)).xreplace({th_s:Cf.th_s,y_s:Cf.y_s})
                                ]+equalendpoints1+equalendpoints2

problem.bounds.endpoint_constraints = len(problem.endpoint_constraints)*[0]


# data = pickle.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results2/Base/tryout.p','rb'))

# #%%

# for idx, i in enumerate([phase_A,phase_B,phase_C]):    
#     i.guess.state_variables = data['state'][idx]
#     i.guess.control_variables = data['control'][idx]
#     i.guess.time = data['time'][idx]


#%%

problem.parameter_variables         = optimized_parameters + (rho_c,)
problem.bounds.parameter_variables  = parameter_bounds + [[-10000,0]]
problem.guess.parameter_variables   = parameter_guess + [0]

problem.objective_function              = -(phase_B.final_state_variables.y_s+d_com-d_tr-r_w)#-d2s(objective.pos_from(O).dot(N.y)).xreplace(dict(zip(phase_B.state_variables,phase_B.final_state_variables)))
problem.auxiliary_data                  = p_jan|quicksolve|{e:0.8}

problem.settings.nlp_tolerance          = 1e-8
problem.settings.mesh_tolerance         = 1e-3

phase_A.mesh.number_mesh_sections = 30
phase_B.mesh.number_mesh_sections = 30
phase_C.mesh.number_mesh_sections = 10

problem.settings.max_nlp_iterations = 10000

# problem.settings.collocation_points_min = 2
# problem.settings.collocation_points_max = 3
# phase_A.mesh.number_mesh_section_nodes  = 2
# phase_B.mesh.number_mesh_section_nodes  = 2
# phase_C.mesh.number_mesh_section_nodes  = 2


#problem.settings.update_scaling = False
#problem.settings.max_mesh_iterations    = 2

problem.initialise()
problem.solve()
    
#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 23:51:44 2022

@author: j.t.heinen
"""
from collections import namedtuple
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pickle

def plotterr(x, y, title, xlabel, ylabel, doubley, legend, save):
    #plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()
    data1 = []
    data2 = []
    lol = 0
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    for idx, data in enumerate(y):
        if doubley[idx] == 0:
            data1.append(data)
            ax.plot(x, data, label=r'$'+legend[idx]+'$', color=colors[idx])
        if doubley[idx] == 1:
            if lol == 0:
                ax2 = ax.twinx()
            data2.append(data)
            ax2.plot(x, data, label=r'$'+legend[idx]+'$', color=colors[idx])
            lol = lol+1


    ax.set_title(title)
    ax.set_ylabel(ylabel[0])
    ax.set_xlabel(xlabel)

    ax.set_ylim(auto=True)
    if bool(data2) == True:
        #ax2.set_ylim(auto=True)
        ax2.set_ylabel(ylabel[1])
        align_yaxis(ax, ax2)

    for i in problem.solution._time:
        ax.axvline(x=i[-1], linestyle='--', linewidth=1, c='grey')

    legendd = []
    for i in legend:
        legendd.append(r'$'+i+'$')

    fig.legend()

    if save == 'yes':
        plt.savefig('/Users/j.t.heinen/Documents/Master Thesis/Results/rw/{}.png'.format(title))
    return plt.show


def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [list(reversed(l))
                               for l in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])

# Combine arrays of seperate phase_As to one array per state, x phase_A b is found from previous section
sol = problem.solution

if bool(problem.parameter_variables) == True:
    p = np.append(p, problem.parameter_variables)
    p_opt = p_jan | dz(problem.parameter_variables, problem.solution.parameter)
    p_vals_opt = p_opt.values()  # np.append(p_vals_jan,sol.parameter)
if bool(problem.parameter_variables) == False:
    p_opt = p_jan
    p_vals_opt = p_vals_jan

for i in range(len(sol._time_)):
    if i == 0:
        t_s = (sol._time_[i],)
        z1  = np.zeros([1,len(t_s[0])])
        o1 = np.ones([1,len(t_s[0])])
        x_a_sol = sm.lambdify([phase_A.state_variables],d2s(pos_a[0].xreplace(quicksolve)).xreplace(p_opt))(sol.state[i])
        y_a_sol = sm.lambdify([phase_A.state_variables],d2s(pos_a[1].xreplace(quicksolve)).xreplace(p_opt))(sol.state[i])
        dx_a_sol = sm.lambdify([phase_A.state_variables],d2s(dx_a[0].xreplace(quicksolve)).xreplace(p_opt))(sol.state[i])
        dy_a_sol = sm.lambdify([phase_A.state_variables],d2s(dx_a[1].xreplace(quicksolve)).xreplace(p_opt))(sol.state[i])
        
        sol_states = np.vstack((sol.state[i][0].T,
                                np.array([x_a_sol,y_a_sol]),
                                sol.state[i][1:5],
                                np.array([dx_a_sol,dy_a_sol]),
                                sol.state[i][5:],
                                ))
        sol_control= (sol.control[i],)
    else:
        t_s = t_s + (sol._time_[i],)
        z2  = np.zeros([1,len(sol._time_[i])])
        o2 = np.ones([1,len(sol._time_[i])])
        
        addstate = np.vstack((z2,
                              sol.state[i][:5],
                              z2,
                              sol.state[i][5:],
                                ))
        sol_states = np.hstack((sol_states, addstate))
        
        sol_control= sol_control + (sol.control[i],)
        
t_s = np.hstack(t_s)
#sol_states = np.hstack(sol_states)
sol_control = np.hstack(sol_control)

all_vars = (x_w,) + phase_B.state_variables[:5] + (dx_w,) + phase_B.state_variables[5:]
all_cont = phase_A.control_variables
ns = namedtuple('states', all_vars)
nc = namedtuple('control', all_cont)

for idx, i in enumerate(all_vars):
    setattr(ns, '%s' % i, sol_states[idx])
for idx, i in enumerate(all_cont):
    setattr(nc, '%s' % i, sol_control[idx])



# %%Energy
def Energy_plot():    
    #y_bf = d2s(bf.pos_from(O).dot(N.y))
    #v_bf = d2s(bf.vel(N).magnitude())
    #y_ff = d2s(ff.pos_from(O).dot(N.y))
    #v_ff = d2s(ff.vel(N).magnitude())
    
    P = y_s * m_s * g + y_h * m_h * g #+ (y_bf+y_ff)*m_l*g
    V = (1/2)*m_s*sm.sqrt(dx_s**2 + dy_s**2)**2 + (1/2) * I_s* dth_s**2 + \
        (1/2)*m_h*sm.sqrt(dy_h**2+  dx_h**2)**2 
#(1/2)*m_l*(v_bf**2 + v_ff**2)
    return V+P


def KEs_A():
    V = (1/2)*m_s*x_w**2 + (1/2)* (I_s+m_s*W1_a.pos_from(com_a).magnitude()**2)*dth_s**2 + (1/2)*m_h*sm.sqrt(dy_h**2+  dx_h**2)**2 
    return V.xreplace(quicksolve)
def KEs_B(): 
    V = (1/2)*m_s*sm.sqrt(dx_s**2 + dy_s**2)**2 + (1/2) * I_s* dth_s**2 + (1/2)*m_h*sm.sqrt(dy_h**2+  dx_h**2)**2 
    return V
def KEh_AB():
    V = (1/2)*m_h*sm.sqrt(dy_h**2+  dx_h**2)**2 
    return V
# %%plot

    
p_opt= p_opt
p = p_opt.keys()
p_vals_opt = p_opt.values()
q = all_vars
u = all_cont

Pl_eval = sm.lambdify([q,u,p], d2s(powerA[Pl1]+powerA[Pl2]))
Pl_sol  = Pl_eval(sol_states, sol_control, p_vals_opt)
fw1_eval  = sm.lambdify([q, u, p], d2s(-fw1))
fw2_eval  = sm.lambdify([q, u, p], d2s(-fw2))

fhx1_eval = sm.lambdify([q, u, p], d2s(-F1.dot(N.x)))
fhy1_eval = sm.lambdify([q, u, p], d2s(-F1.dot(N.y)))
fhx2_eval = sm.lambdify([q, u, p], d2s(-F2.dot(N.x)))
fhy2_eval = sm.lambdify([q, u, p], d2s(-F2.dot(N.y)))

KE_evalA = sm.lambdify([phase_A.state_variables,phase_A.control_variables,p], KEs_A())
KE_evalB = sm.lambdify([phase_B.state_variables,phase_B.control_variables,p], KEs_B())

KE_solA = KE_evalA(sol.state[0],sol.control[0],p_vals_opt)
KE_solB = KE_evalB(sol.state[1],sol.control[1],p_vals_opt)
KE_solC = KE_evalB(sol.state[2],sol.control[2],p_vals_opt)

KE_sol = np.hstack((KE_solA,KE_solB,KE_solC))

fw1_sol   = fw1_eval(sol_states, sol_control, p_vals_opt)
fw2_sol   = fw2_eval(sol_states, sol_control, p_vals_opt)

fhx1_sol  = fhx1_eval(sol_states, sol_control, p_vals_opt)
fhy1_sol  = fhy1_eval(sol_states, sol_control, p_vals_opt)
fhx2_sol  = fhx2_eval(sol_states, sol_control, p_vals_opt)
fhy2_sol  = fhy2_eval(sol_states, sol_control, p_vals_opt)


#plotterr(t_s, [ns.Wl1,ns.Wl2,KE_sol],'Work', 'Time [s]', ['Joule [J]', '[rad]'], [
#           0, 0, 0, 0, 0, 0, 0, 0], ['W_{legL}', ' W_{legR}', 'E_{kinetic}'], 'no')
# plotterr(t_s, [ns.Wl1,ns.Wl2,ns.Wf1,ns.Wf2,ns.Wgs,ns.Wgh, ns.Wl1+ns.Wl2+ns.Wf1+ns.Wf2+ns.Wgs+ns.Wgh,KE_sol],'Work', 'Time [s]', ['Joule [J]', '[rad]'], [
#            0, 0, 0, 0, 0, 0, 0, 0], ['W_{legL}', ' W_{legR}', 'W_{frictionL}', 'W_{frictionR}', 'W_{gravity S}', 'W_{gravity H}', '\sum W','E_{kinetic}'], 'no')

#%%
positions = []
for i in ['x_s', 'y_s', 'th_s', 's_1', 's_2', 'x_h', 'y_h']:
    positions.append(getattr(ns, i))
speeds = []
for i in ['dx_s','dy_s','dth_s', 'ds_1', 'ds_2', 'dx_h', 'dy_h']:
    speeds.append(getattr(ns, i))
    
plotterr(t_s, positions, 'Positions', 'Time [s]', ['Distance [m]', 'Angle [rad]'], [
          0, 0, 1, 0, 0, 0, 0], ['x_s', 'y_s', '\\theta_s', 's_1', 's_2', 'x_h', 'y_h'], 'no')
#%%
plotterr(t_s, speeds, 'Speeds', 'Time [s]', ['Velocity [m/s]', 'Angular velocity [rad/s]'], [0, 0, 1, 0, 0, 0, 0], [
          '\dot x_s', '\dot y_s','\dot \\theta_s', '\dot s_1', '\dot s_2','\dot x_h', '\dot y_h'], 'no')
#%%
plotterr(t_s, [fhy1_sol,fhy2_sol,fhx1_sol,fhx2_sol,ns.Fn], 'Human Forces', 'Time [s]', [['Force [N]']], [
        0, 0, 0, 0, 0], ['F_{leg L}', 'F_{leg R}', 'F_{abduction L}', 'F_{abduction R}','sum'], 'no')
#%%
plotterr(t_s, [nc.fp1,nc.fp2,fw1_sol,fw2_sol], 'Skateboard Force', 'Time [s]', ['Force [N]'], [
          0, 0, 0, 0], ['F_{p1}', 'F_{p2}', 'F_{w1}', 'F_{w2}'], 'no')

#%%animate
# %matplotlib qt
sol_statesT = np.transpose(sol_states)
points = [C0_b, nose_b, C0_b, ff_b, Tr2_b, W2_b, Tr2_b, Tr1_b, W1_b, Tr1_b, B0_b, bf_b, tail_b]

# Evaluate coordinates


coordinates = points[0].pos_from(O).to_matrix(N)
for point in points[1:]:
    coordinates = d2s(coordinates.row_join(point.pos_from(O).to_matrix(N)).xreplace(quicksolve))

#coordinates= coordinates.xreplace(p_opt)
eval_skate_coords = sm.lambdify((q, p), coordinates)

coords = []
for xi in sol_statesT:
    coords.append(eval_skate_coords(xi[:len(q)], p_vals_opt))
coords = np.array(coords)

# Animation figure
fig, ax = plt.subplots()
ax.set_aspect('equal')
title_template = 'Time = {:1.2f} s'
title_text = ax.set_title(title_template.format(t_s[0]))
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')


coors = coords.transpose()

fp1_eval = sm.lambdify([q, u, p], d2s(-fp1))
fp2_eval = sm.lambdify([q, u, p], d2s(-fp2))
fp1_sol = fp1_eval(sol_states, sol_control, p_vals_opt)
fp2_sol = fp2_eval(sol_states, sol_control, p_vals_opt)

Fscale = 100

nFp2a = np.array([-fp2_sol*np.sin(ns.th_s),
                  fp2_sol*np.cos(ns.th_s)])/Fscale
nFw2a = np.array([fw2_sol*np.cos(ns.th_s),
                  fw2_sol*np.sin(ns.th_s)])/Fscale
nFp1a = np.array([fp1_sol*np.sin(p_opt[phi] - ns.th_s),
                  fp1_sol*np.cos(p_opt[phi] - ns.th_s)])/Fscale
nFw1a = np.array([fw1_sol*np.cos(p_opt[phi] - ns.th_s), 
                  -fw1_sol*np.sin(p_opt[phi] - ns.th_s)])/Fscale
# ff y
arrow1 = plt.Arrow(coors[3][0][0], coors[3][1][0],
                    nFp2a[0][0], nFp2a[1][0], width=0.1)
# ff x
arrow2 = plt.Arrow(coors[3][0][0], coors[3][1][0],
                    nFw2a[0][0], nFw2a[1][0], width=0.1)
# bf y
arrow3 = plt.Arrow(coors[11][0][0], coors[11][1][0],
                    nFp1a[0][0], nFp1a[1][0], width=0.1)
# bf x
arrow4 = plt.Arrow(coors[11][0][0], coors[11][1][0],
                    nFw1a[0][0], nFw1a[1][0], width=0.1)
# Fh1
arrow5 = plt.Arrow(ns.x_h[0]-0.1, ns.y_h[0],
                    fhx1_sol[0]/Fscale            , fhy1_sol[0]/Fscale        , width=0.1)
# Fh2
arrow6 = plt.Arrow(ns.x_h[0]+0.1, ns.y_h[0],
                    fhx2_sol[0]/Fscale            ,fhy2_sol[0]/Fscale         , width=0.1)
  
wheel_back = plt.Circle((coords[0, 0, 5], coords[0, 1, 5]), p_opt[r_w])
wheel_front = plt.Circle((coords[0, 0, 8], coords[0, 1, 8]), p_opt[r_w])


def init():
    wheel_back.center = (coords[0, 0, 5], coords[0, 1, 5])
    wheel_front.center = (coords[0, 0, 8], coords[0, 1, 8])
    ax.add_patch(wheel_back)
    ax.add_patch(wheel_front)
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    ax.add_patch(arrow3)
    ax.add_patch(arrow4)
    ax.add_patch(arrow5)
    ax.add_patch(arrow6)

#points1, = ax.plot(sol_states[0][0],sol_states[1][0],marker='o', markerfacecolor='blue', markersize=5)

# Lines


lines, = ax.plot(coords[0, 0, :], coords[0, 1, :], color='black',
                 marker='o', markerfacecolor='blue', markersize=1)
lines1 = ax.plot([-10, 10], [0, 0])
points1, = ax.plot(coors[3][0][0], coors[3][1][0],
                   marker='o', markerfacecolor='blue', markersize=5)
points2, = ax.plot(coors[11][0][0], coors[11][1][0],
                   marker='o', markerfacecolor='blue', markersize=5)
points3, = ax.plot(ns.x_h[0], ns.y_h[0],
                   marker='o', markerfacecolor='blue', markersize=5)


def animate(i):
    global arrow1, arrow2, arrow3, arrow4, arrow5, arrow6
    title_text.set_text(title_template.format(t_s[i]))
    lines.set_data(coords[i, 0, :], coords[i, 1, :])

    wheel_back.center = (coords[i, 0, 5], coords[i, 1, 5])
    wheel_front.center = (coords[i, 0, 8], coords[i, 1, 8])

    points1.set_data(coors[3][0][i], coors[3][1][i])
    points2.set_data(coors[11][0][i], coors[11][1][i])
    points3.set_data(ns.x_h[i], ns.y_h[i])
    ax.patches.remove(arrow1)
    ax.patches.remove(arrow2)
    ax.patches.remove(arrow3)
    ax.patches.remove(arrow4)
    ax.patches.remove(arrow5)
    ax.patches.remove(arrow6)
    #points1.set_data(sol_states[0][i], sol_states[1][i])

    # ff y
    arrow1 = plt.Arrow(coors[3][0][i], coors[3][1][i],
                       nFp2a[0][i]   , nFp2a[1][i]   , width=0.1)
    # ff x
    arrow2 = plt.Arrow(coors[3][0][i], coors[3][1][i],
                       nFw2a[0][i]   , nFw2a[1][i]   , width=0.1)
    # bf y
    arrow3 = plt.Arrow(coors[11][0][i], coors[11][1][i],
                       nFp1a[0][i]    , nFp1a[1][i]    , width=0.1)
    # bf x
    arrow4 = plt.Arrow(coors[11][0][i], coors[11][1][i],
                       nFw1a[0][i]    , nFw1a[1][i]    , width=0.1)
    # Fh1
    arrow5 = plt.Arrow(ns.x_h[i]-0.1, ns.y_h[i],
                       fhx1_sol[i]/Fscale            , fhy1_sol[i]/Fscale, width=0.1)
    # Fh2
    arrow6 = plt.Arrow(ns.x_h[i]+0.1, ns.y_h[i],
                       fhx2_sol[i]/Fscale            , fhy2_sol[i]/Fscale, width=0.1)
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    ax.add_patch(arrow3)
    ax.add_patch(arrow4)
    ax.add_patch(arrow5)
    ax.add_patch(arrow6)


ani = FuncAnimation(fig, animate, len(t_s), init_func=init())
# #%%
# f = '/Users/j.t.heinen/Documents/Master Thesis/Results/'
# ani.save(f+'animation.gif',writer='Pillow')
# #%%
# filename = 'state_cont_time_obj_par_mesh.p'
# outfile = open(f+filename,'wb')
# pickle.dump([sol_states,sol_control,sol._time_,sol.objective,sol.parameter,[problem.solution.mesh_refinement.ph_mesh.tau, problem.solution.mesh_refinement.relative_mesh_errors]],outfile)
# #%%
# for i in [0,1,12]:
#     sol_states[i] = sol_states[i] + np.arange(0,5*len(t_s),5)/len(t_s)
# #%%
# a= 0.2
# count = 0
# fig, ax = plt.subplots()
# time = sol._time_
# step = t_s[-1]/9
# normalize = len(t_s)/t_s[-1]
# first = (np.arange(0,time[0][-1],step*1.4) * normalize).astype(int)
# second= (np.arange(time[0][-1],time[1][-1], step) *normalize).astype(int)
# third = (np.arange(time[1][-1],time[2][-1], step/2) * normalize).astype(int)

# for i in np.hstack((first,len(time[0]),second[1:],len(time[0])+len(time[1]),third,len(t_s)-1)):
#     #global arrow1, arrow2, arrow3, arrow4, arrow5, arrow6
#     # Animation figure
#     a += 0.6/(len(np.append(np.arange(0,len(t_s),round(len(t_s)/9)-1),len(t_s))))
#     ax.set_aspect('equal')
#     text = 't = %s s'%round(t_s[i],3)
#     #ax.set_title("Objective={} [m],   no parameter optimization".format(round(-sol.objective,3)))
#     ax.set_title("Objective={} [m],   {} = {} [m] ".format(round(-sol.objective,3),optimized_parameters[0],round(sol.parameter[0],3)))
#     ax.set_xlim(-0.5, 5.7)
#     ax.set_ylim(-0.1, 1.75)
#     ax.set_xlabel('$x$ [m]')
#     ax.set_ylabel('$y$ [m]')

#     sol_statesT = np.transpose(sol_states)
#     if  count in [0, 2, 6, 11]:
#         plt.text(ns.x_h[i],1.6,text)
#     count += 1

#     coords = []
#     for xi in sol_statesT:
#         coords.append(eval_skate_coords(xi[:len(q)], p_vals_opt))
#     coords = np.array(coords)


#     coors = coords.transpose()

#     coors = coords.transpose()

     
#     wheel_back = plt.Circle((coords[0, 0, 5], coords[0, 1, 5]), p_opt[r_w],alpha = a, color='orange')
#     wheel_front = plt.Circle((coords[0, 0, 8], coords[0, 1, 8]), p_opt[r_w],alpha = a, color='orange')


#     lines, = ax.plot(coords[0, 0, :], coords[0, 1, :], color='black',
#                      marker='o', markerfacecolor='blue', markersize=1,alpha = a)
#     lines1 = ax.plot([-10, 10], [0, 0])
#     points1, = ax.plot(coors[3][0][0], coors[3][1][0], color='black',
#                        marker='o', markerfacecolor='cyan', markersize=5,alpha = a)
#     points2, = ax.plot(coors[11][0][0], coors[11][1][0], color='black',
#                        marker='o', markerfacecolor='blue', markersize=5,alpha = a)
#     points3, = ax.plot(ns.x_h[0], ns.y_h[0], color='black',
#                        marker='o', markerfacecolor='maroon', markersize=5,alpha = a)

    
#     title_text.set_text(title_template.format(t_s[i]))
#     lines.set_data(coords[i, 0, :], coords[i, 1, :])

#     wheel_back.center = (coords[i, 0, 5], coords[i, 1, 5])
#     wheel_front.center = (coords[i, 0, 8], coords[i, 1, 8])
    
#     ax.add_patch(wheel_back)
#     ax.add_patch(wheel_front)
    
#     points1.set_data(coors[3][0][i], coors[3][1][i])
#     points2.set_data(coors[11][0][i], coors[11][1][i])
#     points3.set_data(ns.x_h[i], ns.y_h[i])

#     arrow1 = plt.Arrow(coors[3][0][i], coors[3][1][i],
#                        nFp2a[0][i]   , nFp2a[1][i]   , width=0.1, alpha=a)
#     # ff x
#     arrow2 = plt.Arrow(coors[3][0][i], coors[3][1][i],
#                        nFw2a[0][i]   , nFw2a[1][i]   , width=0.1, alpha=a)
#     # bf y
#     arrow3 = plt.Arrow(coors[11][0][i], coors[11][1][i],
#                        nFp1a[0][i]    , nFp1a[1][i]    , width=0.1, alpha=a)
#     # bf x
#     arrow4 = plt.Arrow(coors[11][0][i], coors[11][1][i],
#                        nFw1a[0][i]    , nFw1a[1][i]    , width=0.1, alpha=a)
#     # Fh1
#     arrow5 = plt.Arrow(ns.x_h[i]-0.1, ns.y_h[i],
#                        fhx1_sol[i]/Fscale            , fhy1_sol[i]/Fscale, width=0.1, alpha=a)
#     # Fh2
#     arrow6 = plt.Arrow(ns.x_h[i]+0.1, ns.y_h[i],
#                        fhx2_sol[i]/Fscale            , fhy2_sol[i]/Fscale, width=0.1, alpha=a)


# #%%
# fig,ax = plt.subplots()
# ax.plot(t_s,nc.g_k2)
# ax.plot(t_s,ns.ds_2)
# #ax.plot(t_s,nc.g_k1)
# #ax.plot(t_s,ns.ds_1)
# #%%

# def extract_sol(problem):
#     sol = problem.solution
#     mesh = sol.mesh_refinement.__dict__.copy()
#     ph_mesh = sol.mesh_refinement.ph_mesh.__dict__.copy()
#     for i in ['backend','dy_ph_callables','next_iter_mesh','ph_mesh','it','sol','ocp']:
#         mesh.pop(i)
#     for i in ['backend','settings','quadrature','sI_matrix','sA_matrix','p']:
#         ph_mesh.pop(i)
        
        
#     dumpdict = {'state':sol.state,
#                 'control':sol.control,
#                 'integral':sol.integral,
#                 'parameter':sol.parameter,
#                 'time':sol._time_,
#                 'mesh':mesh,
#                 'ph_mesh':ph_mesh,
#                 'objective':sol.objective,
#                 'guess':problem.guess.parameter_variables,
#                 'p_opt':p_opt,
#                 'sol_states':sol_states,
#                 'sol_control':sol_control,
#                 't_s':t_s,
#                 'ns':ns,
#                 'nc':nc,
#                 'all_vars':all_vars,
#                 'all_cont':all_cont._asdict().values(),
#                 'coordinates':coordinates,
#                 'Astate':phase_A.state_variables._asdict().values(),
#                 'Bstate':phase_B.state_variables._asdict().values(),
#                 }
#     return dumpdict
#                 #%%

# dill.dump(extract_sol(problem),open('/Users/j.t.heinen/Documents/Master Thesis/Results3/prescribed/data_penny_02.p','wb'))

# #%%
# dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results/dill/tryout.p','rb'))

