#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 23:00:54 2022

@author: j.t.heinen
"""

from collections import namedtuple
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import dill
import sympy.physics.mechanics as me
import sympy as sm
import numpy as np
import matplotlib.ticker as ticker

path = '/Users/j.t.heinen/Documents/Master Thesis/Results3/all/'
testcase = 'data_no_tail'
data = dill.load(open(path+testcase+'.p','rb'))


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

phi = sm.Symbol('phi')
A.orient_axis(N, N.z, th_s_)  # Bodyfixed frame
B.orient_axis(N, N.z, th_s_-phi)
C.orient_axis(N, N.z, th_s_+phi)


fp1, fp2 = sm.symbols('fp1,fp2')

def logistic(N,dx,mu,k):
     return N*mu*((1-sm.exp(dx/k))/(1+sm.exp(dx/k)))

l_xp1, l_xm1, g_k1,l_xp2, l_xm2, g_k2 = sm.symbols('l_xp1, l_xm1, g_k1,l_xp2, l_xm2, g_k2')

fw1 = l_xp1 - l_xm1
fw2 = l_xp2 - l_xm2

F1 = -fp1*B.y - fw1*B.x
F2 = -fp2*A.y - fw2*A.x

Fs= F1 + F2



def plotterr(x, y, title, xlabel, ylabel, doubley, legend, save, no):
    #plt.rcParams['text.usetex'] = True
    ## COSTUMIZE
    # twindex = np.where(np.array(doubley)==1)[0][0]
    # notwin = y[:twindex]+y[(twindex+1):]
    # ax[no].set_ylim(np.min(notwin),np.max(notwin))
    ax[no].set_ylim(auto=True)

    data1 = []
    data2 = []
    lol = 0
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    for idx, data in enumerate(y):
        if doubley[idx] == 0:
            data1.append(data)
            ax[no].plot(x, data, label=r'$'+legend[idx]+'$', color=colors[idx])
        if doubley[idx] == 2:
            ax[no].set_ylim(auto=False)
            ax[no].plot(x, data, label=r'$'+legend[idx]+'$', color=colors[idx])
        if doubley[idx] == 1:
            if lol == 0:
                ax2 = ax[no].twinx()
            data2.append(data)
            ax2.plot(x, data, label=r'$'+legend[idx]+'$', color=colors[idx])
            lol = lol+1

    ax[no].set_xlim(min(x)+(max(x)-max(x)*1.04),max(x)*1.04)

    ax[no].set_title(title,loc='left')
    ax[no].set_ylabel(ylabel[0])
    if no ==3:
        ax[no].set_xlabel(xlabel)
    ax[no].xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax[no].xaxis.set_minor_locator(ticker.MaxNLocator(40))
    ax[no].yaxis.set_major_locator(ticker.MaxNLocator(6))


    ax[no].xaxis.set_minor_locator(ticker.LinearLocator(9*4))

    if bool(data2) == True:
        ax2.set_ylim(auto=True)
        ax2.set_ylabel(ylabel[1])
        align_yaxis(ax[no], ax2)

    for idx,i in enumerate(time):
        ax[no].axvline(x=i[-1], linestyle='--', linewidth=1, c='blue')
        if no == 0:
            if idx==0:
                ax[no].text(i[-1],np.max(y)*1.3,'Impact',horizontalalignment='center')
            if idx==1:
                ax[no].text(i[-1],np.max(y)*1.3,'Highest point',horizontalalignment='center')
            if idx==2:
                ax[no].text(i[-1],np.max(y)*1.3,'Landing',horizontalalignment='center')
    legendd = []
    for i in legend:
        legendd.append(r'$'+i+'$')

    ax[no].legend(loc='best')
    ax[no].grid()
    ax[no].grid(visible=True, which='minor', color='grey', alpha=0.3)
    if 1 in doubley:
        ax[no].legend(loc='upper left')
        ax2.legend(loc='upper right')
        
    if no < 3:
        ax[no].xaxis.set_ticklabels([])
    

    
    if save == 'yes':
        plt.savefig('/Users/j.t.heinen/Documents/Master Thesis/Results2/'+path+testcase+'{}.png'.format(title))
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
    #axes[1].yaxis.set_major_locator(ticker.MaxNLocator(6))
    l = axes[0].get_ylim()
    l2 = axes[1].get_ylim()
    f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
    ticks = f(axes[0].get_yticks())
    axes[1].yaxis.set_major_locator(ticker.FixedLocator(ticks))    
    #axes[1].yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))

    #if no ==

state = data['state']
p_opt = data['p_opt']
sol_control = data['sol_control']
sol_states = data['sol_states']
control = data['control']
all_vars = data['all_vars']
all_cont = data['all_cont']
coordinates = data['coordinates']
t_s = data['t_s']
ns = data['ns']
time = data['time']
objective = data['objective']
parameter = data['parameter']

ns = namedtuple('states', all_vars)
nc = namedtuple('control', all_cont)
for idx, i in enumerate(all_vars):
    setattr(ns, '%s' % i, sol_states[idx])
for idx, i in enumerate(all_cont):
    setattr(nc, '%s' % i, sol_control[idx])

p_vals_opt = p_opt.values()

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
p = p_opt.keys()
p_vals_opt = p_opt.values()
q = all_vars
u = all_cont

from scipy.ndimage import gaussian_filter1d
def gf(S):
    return gaussian_filter1d(S,sigma=0.7)

#Pl_eval = sm.lambdify([q,u,p], d2s(powerA[Pl1]+powerA[Pl2]))
#Pl_sol  = Pl_eval(sol_states, sol_control, p_vals_opt)
fw1_eval  = sm.lambdify([q, u, p], d2s(-fw1))
fw2_eval  = sm.lambdify([q, u, p], d2s(-fw2))

fhx1_eval = sm.lambdify([q, u, p], d2s(-F1.dot(N.x)))
fhy1_eval = sm.lambdify([q, u, p], d2s(-F1.dot(N.y)))
fhx2_eval = sm.lambdify([q, u, p], d2s(-F2.dot(N.x)))
fhy2_eval = sm.lambdify([q, u, p], d2s(-F2.dot(N.y)))

# KE_evalA = sm.lambdify([A_state,phase_A.control_variables,p], KEs_A())
# KE_evalB = sm.lambdify([phase_B.state_variables,phase_B.control_variables,p], KEs_B())

# KE_solA = KE_evalA(state[0],sol.control[0],p_vals_opt)
# KE_solB = KE_evalB(state[1],sol.control[1],p_vals_opt)
# KE_solC = KE_evalB(state[2],sol.control[2],p_vals_opt)

# KE_sol = np.hstack((KE_solA,KE_solB,KE_solC))

fw1_sol   = gf(fw1_eval(sol_states, sol_control, p_vals_opt))
fw2_sol   = gf(fw2_eval(sol_states, sol_control, p_vals_opt))

fhx1_sol  = gf(fhx1_eval(sol_states, sol_control, p_vals_opt))
fhy1_sol  = gf(fhy1_eval(sol_states, sol_control, p_vals_opt))
fhx2_sol  = gf(fhx2_eval(sol_states, sol_control, p_vals_opt))
fhy2_sol  = gf(fhy2_eval(sol_states, sol_control, p_vals_opt))
#%%
# positions = []
# for i in ['x_s', 'y_s', 'th_s', 's_1', 's_2', 'x_h', 'y_h']:
#     positions.append(getattr(ns, i))
# speeds = []
# for i in ['dx_s','dy_s','dth_s', 'ds_1', 'ds_2', 'dx_h', 'dy_h']:
#     speeds.append(getattr(ns, i))

# fig, ax = plt.subplots(3,1)
# fig.set_figwidth(15)
# fig.set_figheight(7.5)
# plotterr(t_s, positions, 'Positions', 'Time [s]', ['Distance [m]', 'Angle [rad]'], [
#           0, 0, 1, 0, 0, 0, 0], ['x_s', 'y_s', '\\theta_s', 's_1', 's_2', 'x_h', 'y_h'], 'yes',0)
# plotterr(t_s, speeds, 'Speeds', 'Time [s]', ['Velocity [m/s]', 'Angular velocity [rad/s]'], [0, 0, 1, 0, 0, 0, 0], [
#           '\dot x_s', '\dot y_s','\dot \\theta_s', '\dot s_1', '\dot s_2','\dot x_h', '\dot y_h'], 'yes',1)
# plotterr(t_s, [fhy1_sol,fhy2_sol,fhx1_sol,fhx2_sol,ns.Fn], 'Human Forces', 'Time [s]', ['Force [N]',[]], [
#         0, 0, 0, 0, 0], ['F_{leg L}', 'F_{leg R}', 'F_{abduction L}', 'F_{abduction R}','\sum F_{vertical}'], 'yes',2)

# timeticks = []
# for i in range(8):
#     timeticks.append(ax[0].get_xticklabels(which='major')[i].get_position()[0])
# # plotterr(t_s, [nc.fp1,nc.fp2,fw1_sol,fw2_sol], 'Skateboard Force', 'Time [s]', ['Force [N]'], [
# #           0, 0, 0, 0], ['F_{p1}', 'F_{p2}', 'F_{w1}', 'F_{w2}'], 'yes',3)

#%%animate
#%matplotlib qt


sol_statesT = np.transpose(sol_states)

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

fp1_sol = -nc.fp1
fp2_sol = -nc.fp2

Fscale = 500

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
  
wheel_back = plt.Circle((coords[0, 0, 5], coords[0, 1, 5]), p_opt[sm.Symbol('r_w')])
wheel_front = plt.Circle((coords[0, 0, 8], coords[0, 1, 8]), p_opt[sm.Symbol('r_w')])


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

fig.set_size_inches(16, 9, True)

ani = FuncAnimation(fig, animate, len(t_s), init_func=init())
# #%%
#ani.save(path+testcase+'animation.gif',writer='Pillow',dpi)
#%%
# filename = 'state_cont_time_obj_par_mesh.p'
# outfile = open(f+filename,'wb')
# pickle.dump([sol_states,sol_control,sol._time_,sol.objective,sol.parameter,[problem.solution.mesh_refinement.ph_mesh.tau, problem.solution.mesh_refinement.relative_mesh_errors]],outfile)
#%%
fig, ax = plt.subplots(4,1)
fig.set_figwidth(15)
fig.set_figheight(12)
fig.tight_layout

#fig.set_figheight(15)
positions = []
for i in ['x_s', 'y_s', 'th_s', 's_1', 's_2', 'x_h', 'y_h']:
    if i == 'y_s':
        positions.append(getattr(ns, i)-getattr(ns, i)[0])    
    else: 
        positions.append(getattr(ns, i))
speeds = []
for i in ['dx_s','dy_s','dth_s', 'ds_1', 'ds_2', 'dx_h', 'dy_h']:
    speeds.append(getattr(ns, i))
    
plotterr(t_s, positions, 'Positions', 'Time [s]', ['Distance [m]', 'Angle [rad]'], [
          0, 0, 1, 0, 0, 0, 0], ['x_s', '\Delta y_s', '\\theta_s', 's_1', 's_2', 'x_h', 'y_h'], 'no',1)
plotterr(t_s, speeds, 'Speeds', 'Time [s]', ['Velocity [m/s]', 'Angular velocity [rad/s]'], [0, 2, 1, 0, 0, 0, 0], [
          '\dot x_s', '\dot y_s','\dot \\theta_s', '\dot s_1', '\dot s_2','\dot x_h', '\dot y_h'], 'no',2)
filt = 0.7
plotterr(t_s, [gaussian_filter1d(-fhy1_sol,sigma=filt),gaussian_filter1d(-fhy2_sol,sigma=filt),gaussian_filter1d(-fhx1_sol,sigma=filt),gaussian_filter1d(-fhx2_sol,sigma=filt),-ns.Fn], 'Forces exerted by human on skateboard in N frame', 'Time [s]', ['Force [N]',[]], [
        0, 0, 0, 0, 0], ['F_{extension L}', 'F_{extension R}', 'F_{abduction L}', 'F_{abduction R}','\sum F_{vertical}'], 'no',3)

timeticks = []
for i in range(len(ax[1].get_xticklabels())-1):
    timeticks.append(ax[1].get_xticklabels(which='major')[i].get_position()[0])

stretch = 10
sol_states_stretch = sol_states.copy()
sol_states_stretch[12] = t_s*stretch
sol_states_stretch[0] = sol_states_stretch[0] + t_s*stretch
sol_states_stretch[1] = sol_states_stretch[1] + t_s*stretch

a= 0.2
count = 0

def closest_value(input_list, input_value):
  arr = np.asarray(input_list)
  i = (np.abs(arr - input_value)).argmin() 
  return arr[i]

timeticksindex = [np.where(t_s==closest_value(t_s,i))[0][0] for i in np.arange(0,t_s[-1],timeticks[4]-timeticks[3])]
timeevents = [len(time[0])-1,len(time[0])+len(time[1])-2,len(t_s)-1]
#extra = [np.where(t_s==closest_value(t_s,1.3))[0][0]]#,np.where(t_s==closest_value(t_s,1.125))[0][0]]
index = timeticksindex+timeevents#+extra
index = np.sort(index)

eval_skate_coords = sm.lambdify((q, p), coordinates)

coords = []
for xi in sol_statesT:
    coords.append(eval_skate_coords(xi[:len(q)], p_vals_opt))
coords = np.array(coords)

for idx, i in enumerate(time):
    ax[0].axvline(x=i[-1]*stretch, linestyle='--', linewidth=1, c='blue')
    if idx == 0:
        ax[0].text(i[-1]*stretch,1.6*1.2,'Impact',horizontalalignment='center',color='blue')
    if idx == 1:
        ax[0].text(i[-1]*stretch,1.6*1.2,'Highest point',horizontalalignment='center',color='blue')
    if idx ==2:
        ax[0].text(i[-1]*stretch,1.6*1.2,'Landing',horizontalalignment='center',color='blue')
coors = coords.transpose()

ax[0].grid(visible=True)
ax[0].grid(visible=True, which='minor', color='grey', alpha=0.3)
ax[0].xaxis.tick_top()

a = 1
ax[0].set_aspect('equal')
ax[0].set_title('Trajectory',loc='left')
ax[0].set_xlim(min(t_s)+(max(t_s)*stretch-max(t_s)*1.04*stretch),max(t_s)*1.04*stretch)
ax[0].set_ylim(-0.01, 1.6)
ax[0].set_ylabel('Height [m]')

for i in range(4):
    ax[i].xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax[i].xaxis.set_minor_locator(ticker.MaxNLocator(40))

ax[0].yaxis.set_major_locator(ticker.MaxNLocator(6))
ax[0].set_xticklabels(["%.2f" % float(i) for i in timeticks])
    
sol_statesT = np.transpose(sol_states_stretch)

coords = []
for xi in sol_statesT:
    coords.append(eval_skate_coords(xi[:len(q)], p_vals_opt))
coords = np.array(coords)
coors = coords.transpose()
count = 0
#index  = np.delete(index,8)
#index  = np.delete(index,4)

for i in index:
#     #BASE PLOT
# index[6] = index[6] + 20
# index[8] = index[8] - 10
# index[11] = index[11] - 5
# for i in index:
#     a = 1
#     if count == 4: #or count == 7:
#         a = 0.3
#     count += 1

    #all notrrw index  = np.delete(index,5)
    a = 1
    if count == 4:# or count == 5:
        a= 0.3
    count += 1
    wheel_back = plt.Circle((coords[0, 0, 5], coords[0, 1, 5]), p_opt[sm.Symbol('r_w')],alpha = a, color='orange')
    wheel_front = plt.Circle((coords[0, 0, 8], coords[0, 1, 8]), p_opt[sm.Symbol('r_w')],alpha = a, color='orange')
    
    
    lines, = ax[0].plot(coords[0, 0, :], coords[0, 1, :], color='black',
                     marker='o', markerfacecolor='blue', markersize=1,alpha = a)
    lines1 = ax[0].plot([-20, 20], [0, 0], color = 'olive')
    points1, = ax[0].plot(coors[3][0][0], coors[3][1][0], color='black',
                       marker='o', markerfacecolor='cyan', markersize=5,alpha = a)
    points2, = ax[0].plot(coors[11][0][0], coors[11][1][0], color='black',
                       marker='o', markerfacecolor='blue', markersize=5,alpha = a)
    points3, = ax[0].plot(ns.x_h[0], ns.y_h[0], color='black',
                       marker='o', markerfacecolor='maroon', markersize=5)#,alpha = a)

   
    #title_text.set_text(title_template.format(t_s[i]))
    lines.set_data(coords[i, 0, :], coords[i, 1, :])

    wheel_back.center = (coords[i, 0, 5], coords[i, 1, 5])
    wheel_front.center = (coords[i, 0, 8], coords[i, 1, 8])
    
    ax[0].add_patch(wheel_back)
    ax[0].add_patch(wheel_front)
    #ax[0].grid()
    points1.set_data(coors[3][0][i], coors[3][1][i])
    points2.set_data(coors[11][0][i], coors[11][1][i])
    points3.set_data(sol_states_stretch[12][i], ns.y_h[i])



#plt.savefig(path+testcase+'{}.png'.format('dpi600'),dpi=600)

#%%#%%
fig, ax = plt.subplots(4,1)
fig.set_figwidth(15)
fig.set_figheight(12)
fig.tight_layout

#fig.set_figheight(15)
positions = []
for i in ['x_s', 'y_s', 'th_s', 's_1', 's_2', 'x_h', 'y_h']:
    if i == 'y_s':
        positions.append(getattr(ns, i)-getattr(ns, i)[0])    
    else: 
        positions.append(getattr(ns, i))
speeds = []
for i in ['dx_s','dy_s','dth_s', 'ds_1', 'ds_2', 'dx_h', 'dy_h']:
    speeds.append(getattr(ns, i))
    
plotterr(t_s, positions, 'Positions', 'Time [s]', ['Distance [m]', 'Angle [rad]'], [
          0, 0, 1, 0, 0, 0, 0], ['x_s', '\Delta y_s', '\\theta_s', 's_1', 's_2', 'x_h', 'y_h'], 'no',1)
plotterr(t_s, speeds, 'Speeds', 'Time [s]', ['Velocity [m/s]', 'Angular velocity [rad/s]'], [0, 2, 1, 0, 0, 0, 0], [
          '\dot x_s', '\dot y_s','\dot \\theta_s', '\dot s_1', '\dot s_2','\dot x_h', '\dot y_h'], 'no',2)
filt = 0.7
plotterr(t_s, [gaussian_filter1d(-fhy1_sol,sigma=filt),gaussian_filter1d(-fhy2_sol,sigma=filt),gaussian_filter1d(-fhx1_sol,sigma=filt),gaussian_filter1d(-fhx2_sol,sigma=filt),-ns.Fn], 'Forces exerted by human on skateboard in N frame', 'Time [s]', ['Force [N]',[]], [
        0, 0, 0, 0, 0], ['F_{extension L}', 'F_{extension R}', 'F_{abduction L}', 'F_{abduction R}','\sum F_{vertical}'], 'no',3)

timeticks = []
for i in range(len(ax[1].get_xticklabels())-1):
    timeticks.append(ax[1].get_xticklabels(which='major')[i].get_position()[0])

stretch = 10
sol_states_stretch = sol_states.copy()
sol_states_stretch[12] = t_s*stretch
sol_states_stretch[0] = sol_states_stretch[0] + t_s*stretch
sol_states_stretch[1] = sol_states_stretch[1] + t_s*stretch

a= 0.2
count = 0

def closest_value(input_list, input_value):
  arr = np.asarray(input_list)
  i = (np.abs(arr - input_value)).argmin() 
  return arr[i]

timeticksindex = [np.where(t_s==closest_value(t_s,i))[0][0] for i in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.4]]
timeevents = [len(time[0])-1,len(time[0])+len(time[1])-2,len(t_s)-1]
#extra = [np.where(t_s==closest_value(t_s,1.3))[0][0]]#,np.where(t_s==closest_value(t_s,1.125))[0][0]]
index = timeticksindex+timeevents#+extra
index = np.sort(index)

eval_skate_coords = sm.lambdify((q, p), coordinates)

coords = []
for xi in sol_statesT:
    coords.append(eval_skate_coords(xi[:len(q)], p_vals_opt))
coords = np.array(coords)

for idx, i in enumerate(time):
    ax[0].axvline(x=i[-1]*stretch, linestyle='--', linewidth=1, c='blue')
    if idx == 0:
        ax[0].text(i[-1]*stretch,1.6*1.2,'Impact',horizontalalignment='center',color='blue')
    if idx == 1:
        ax[0].text(i[-1]*stretch,1.6*1.2,'Highest point',horizontalalignment='center',color='blue')
    if idx ==2:
        ax[0].text(i[-1]*stretch,1.6*1.2,'Landing',horizontalalignment='center',color='blue')
coors = coords.transpose()

ax[0].grid(visible=True)
ax[0].grid(visible=True, which='minor', color='grey', alpha=0.3)
ax[0].xaxis.tick_top()

a = 1
ax[0].set_aspect('equal')
ax[0].set_title('Trajectory',loc='left')
ax[0].set_xlim(min(t_s)+(max(t_s)*stretch-max(t_s)*1.04*stretch),max(t_s)*1.04*stretch)
ax[0].set_ylim(-0.01, 1.6)
ax[0].set_ylabel('Height [m]')

for i in range(4):
    ax[i].xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax[i].xaxis.set_minor_locator(ticker.MaxNLocator(40))
    if i == 0:
        ax[i].set_xlim(left = 0.15*stretch)
    else:
        ax[i].set_xlim(left=0.15)

ax[0].yaxis.set_major_locator(ticker.MaxNLocator(6))
ax[0].set_xticklabels(["%.2f" % float(i) for i in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.4]])
    
sol_statesT = np.transpose(sol_states_stretch)

coords = []
for xi in sol_statesT:
    coords.append(eval_skate_coords(xi[:len(q)], p_vals_opt))
coords = np.array(coords)
coors = coords.transpose()
count = 0
#index  = np.delete(index,8)
#index  = np.delete(index,4)

for i in index:
#     #BASE PLOT
# index[6] = index[6] + 20
# index[8] = index[8] - 10
# index[11] = index[11] - 5
# for i in index:
#     a = 1
#     if count == 4: #or count == 7:
#         a = 0.3
#     count += 1

    #all notrrw index  = np.delete(index,5)
    # a = 1
    # if count == 5:# or count == 5:
    #     a= 0.3
    # count += 1
    wheel_back = plt.Circle((coords[0, 0, 5], coords[0, 1, 5]), p_opt[sm.Symbol('r_w')],alpha = a, color='orange')
    wheel_front = plt.Circle((coords[0, 0, 8], coords[0, 1, 8]), p_opt[sm.Symbol('r_w')],alpha = a, color='orange')
    
    
    lines, = ax[0].plot(coords[0, 0, :], coords[0, 1, :], color='black',
                     marker='o', markerfacecolor='blue', markersize=1,alpha = a)
    lines1 = ax[0].plot([-20, 20], [0, 0], color = 'olive')
    points1, = ax[0].plot(coors[3][0][0], coors[3][1][0], color='black',
                       marker='o', markerfacecolor='cyan', markersize=5,alpha = a)
    points2, = ax[0].plot(coors[11][0][0], coors[11][1][0], color='black',
                       marker='o', markerfacecolor='blue', markersize=5,alpha = a)
    points3, = ax[0].plot(ns.x_h[0], ns.y_h[0], color='black',
                       marker='o', markerfacecolor='maroon', markersize=5)#,alpha = a)

   
    #title_text.set_text(title_template.format(t_s[i]))
    lines.set_data(coords[i, 0, :], coords[i, 1, :])

    wheel_back.center = (coords[i, 0, 5], coords[i, 1, 5])
    wheel_front.center = (coords[i, 0, 8], coords[i, 1, 8])
    
    ax[0].add_patch(wheel_back)
    ax[0].add_patch(wheel_front)
    #ax[0].grid()
    points1.set_data(coors[3][0][i], coors[3][1][i])
    points2.set_data(coors[11][0][i], coors[11][1][i])
    points3.set_data(sol_states_stretch[12][i], ns.y_h[i])



plt.savefig(path+testcase+'{}.png'.format('dpi600'),dpi=600)

#%%
fig, ax = plt.subplots(4,1)
fig.set_figwidth(20)
fig.set_figheight(12)
fig.tight_layout

#fig.set_figheight(15)
positions = []
for i in ['x_s', 'y_s', 'th_s', 's_1', 's_2', 'x_h', 'y_h']:
    if i == 'y_s':
        positions.append(getattr(ns, i)-getattr(ns, i)[0])    
    else: 
        positions.append(getattr(ns, i))
speeds = []
for i in ['dx_s','dy_s','dth_s', 'ds_1', 'ds_2', 'dx_h', 'dy_h']:
    speeds.append(getattr(ns, i))
    
plotterr(t_s, positions, 'Positions', 'Time [s]', ['Distance [m]', 'Angle [rad]'], [
          0, 0, 1, 0, 0, 0, 0], ['x_s', '\Delta y_s', '\\theta_s', 's_1', 's_2', 'x_h', 'y_h'], 'no',1)
plotterr(t_s, speeds, 'Speeds', 'Time [s]', ['Velocity [m/s]', 'Angular velocity [rad/s]'], [0, 2, 1, 0, 0, 0, 0], [
          '\dot x_s', '\dot y_s','\dot \\theta_s', '\dot s_1', '\dot s_2','\dot x_h', '\dot y_h'], 'no',2)
filt = 0.7
plotterr(t_s, [gaussian_filter1d(-fhy1_sol,sigma=filt),gaussian_filter1d(-fhy2_sol,sigma=filt),gaussian_filter1d(-fhx1_sol,sigma=filt),gaussian_filter1d(-fhx2_sol,sigma=filt),-ns.Fn], 'Forces exerted by human on skateboard in N frame', 'Time [s]', ['Force [N]',[]], [
        0, 0, 0, 0, 0], ['F_{extension L}', 'F_{extension R}', 'F_{abduction L}', 'F_{abduction R}','\sum F_{vertical}'], 'no',3)

timeticks = []
for i in range(len(ax[1].get_xticklabels())-1):
    timeticks.append(ax[1].get_xticklabels(which='major')[i].get_position()[0])

stretch = 10
sol_states_stretch = sol_states.copy()
sol_states_stretch[12] = t_s*stretch
sol_states_stretch[0] = sol_states_stretch[0] + t_s*stretch
sol_states_stretch[1] = sol_states_stretch[1] + t_s*stretch

a= 0.2
count = 0

def closest_value(input_list, input_value):
  arr = np.asarray(input_list)
  i = (np.abs(arr - input_value)).argmin() 
  return arr[i]

timeticksindex = [np.where(t_s==closest_value(t_s,i))[0][0] for i in [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]]
timeevents = [len(time[0])-1,len(time[0])+len(time[1])-2,len(t_s)-1]
#extra = [np.where(t_s==closest_value(t_s,1.3))[0][0]]#,np.where(t_s==closest_value(t_s,1.125))[0][0]]
index = timeticksindex+timeevents#+extra
index = np.sort(index)

eval_skate_coords = sm.lambdify((q, p), coordinates)

coords = []
for xi in sol_statesT:
    coords.append(eval_skate_coords(xi[:len(q)], p_vals_opt))
coords = np.array(coords)

for idx, i in enumerate(time):
    ax[0].axvline(x=i[-1]*stretch, linestyle='--', linewidth=1, c='blue')
    if idx == 0:
        ax[0].text(i[-1]*stretch,1.6*1.2,'Impact',horizontalalignment='center',color='blue')
    if idx == 1:
        ax[0].text(i[-1]*stretch,1.6*1.2,'Highest point',horizontalalignment='center',color='blue')
    if idx ==2:
        ax[0].text(i[-1]*stretch,1.6*1.2,'Landing',horizontalalignment='center',color='blue')
coors = coords.transpose()

ax[0].grid(visible=True)
ax[0].grid(visible=True, which='minor', color='grey', alpha=0.3)
ax[0].xaxis.tick_top()

a = 1
ax[0].set_aspect('equal')
ax[0].set_title('Trajectory',loc='left')
ax[0].set_xlim(min(t_s)+(max(t_s)*stretch-max(t_s)*1.04*stretch),max(t_s)*1.04*stretch)
ax[0].set_ylim(-0.01, 1.6)
ax[0].set_ylabel('Height [m]')

for i in range(4):
    ax[i].xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax[i].xaxis.set_minor_locator(ticker.MaxNLocator(40))
    if i == 0:
        ax[i].set_xlim(left = 0.5*stretch)
    else:
        ax[i].set_xlim(left=0.5)

ax[0].yaxis.set_major_locator(ticker.MaxNLocator(6))
ax[0].set_xticklabels(["%.2f" % float(i) for i in [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0]])
    
sol_statesT = np.transpose(sol_states_stretch)

coords = []
for xi in sol_statesT:
    coords.append(eval_skate_coords(xi[:len(q)], p_vals_opt))
coords = np.array(coords)
coors = coords.transpose()
count = 0
#index  = np.delete(index,8)
#index  = np.delete(index,4)

for i in index:
#     #BASE PLOT
# index[6] = index[6] + 20
# index[8] = index[8] - 10
# index[11] = index[11] - 5
# for i in index:
#     a = 1
#     if count == 4: #or count == 7:
#         a = 0.3
#     count += 1

    #all notrrw index  = np.delete(index,5)
    a = 1
    if count == 5:# or count == 5:
        a= 0.3
    count += 1
    wheel_back = plt.Circle((coords[0, 0, 5], coords[0, 1, 5]), p_opt[sm.Symbol('r_w')],alpha = a, color='orange')
    wheel_front = plt.Circle((coords[0, 0, 8], coords[0, 1, 8]), p_opt[sm.Symbol('r_w')],alpha = a, color='orange')
    
    
    lines, = ax[0].plot(coords[0, 0, :], coords[0, 1, :], color='black',
                     marker='o', markerfacecolor='blue', markersize=1,alpha = a)
    lines1 = ax[0].plot([-20, 20], [0, 0], color = 'olive')
    points1, = ax[0].plot(coors[3][0][0], coors[3][1][0], color='black',
                       marker='o', markerfacecolor='cyan', markersize=5,alpha = a)
    points2, = ax[0].plot(coors[11][0][0], coors[11][1][0], color='black',
                       marker='o', markerfacecolor='blue', markersize=5,alpha = a)
    points3, = ax[0].plot(ns.x_h[0], ns.y_h[0], color='black',
                       marker='o', markerfacecolor='maroon', markersize=5)#,alpha = a)

   
    #title_text.set_text(title_template.format(t_s[i]))
    lines.set_data(coords[i, 0, :], coords[i, 1, :])

    wheel_back.center = (coords[i, 0, 5], coords[i, 1, 5])
    wheel_front.center = (coords[i, 0, 8], coords[i, 1, 8])
    
    ax[0].add_patch(wheel_back)
    ax[0].add_patch(wheel_front)
    #ax[0].grid()
    points1.set_data(coors[3][0][i], coors[3][1][i])
    points2.set_data(coors[11][0][i], coors[11][1][i])
    points3.set_data(sol_states_stretch[12][i], ns.y_h[i])



plt.savefig(path+testcase+'{}.png'.format('dpi600'),dpi=600)

