#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 22:52:16 2022

@author: j.t.heinen
"""
import dill
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm

def plot_sb(p_opt,all_vars,sol_states,coordinates,x_pos,ax):
    for i in [0,1,12]:
        sol_states[i] = x_pos
    p = p_opt.keys()
    p_vals_opt = p_opt.values()
    q = all_vars
    sol_statesT = np.transpose(sol_states)
    eval_skate_coords = sm.lambdify((q, p), coordinates)
    coords = []
    for xi in sol_statesT:
        coords.append(eval_skate_coords(xi[:len(q)], p_vals_opt))
    coords = np.array(coords)
    coors = coords.transpose()
    wheel_back = plt.Circle((coords[0, 0, 5], coords[0, 1, 5]), p_opt[sm.Symbol('r_w')])
    wheel_front = plt.Circle((coords[0, 0, 8], coords[0, 1, 8]), p_opt[sm.Symbol('r_w')])
    wheel_back.center = (coords[0, 0, 5], coords[0, 1, 5])
    wheel_front.center = (coords[0, 0, 8], coords[0, 1, 8])
    ax.add_patch(wheel_back)
    ax.add_patch(wheel_front)
    lines, = ax.plot(coords[0, 0, :], coords[0, 1, :], color='black',
                     marker='o', markerfacecolor='blue', markersize=1)

wb = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/l_wb/data_l_wb.p','rb'))
base = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/Base/data_base.p','rb'))
dtr = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/d_tr/data_d_tr.p','rb'))
lf = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/l_f/data_l_f.p','rb'))
phi_ = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/phi/data_phi.p','rb'))
long = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/prescribed/data_longboard_init_longer_tail_opt.p','rb'))
penn02 = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/prescribed/data_penny_02.p','rb'))
penn08 = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/prescribed/data_penny_08.p','rb'))
lt = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/l_t/data_l_t.p','rb'))
rw = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/r_w/data_r_w.p','rb'))
all_no_tail = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/all/data_no_tail.p','rb'))
all_notrrw = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/all/data_notrrw.p','rb'))
all_notrrwlt = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/all/data_notrrwlt.p','rb'))
all_ = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/all/data_all.p','rb'))
all_nomass = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/all/data_all_nomass_truckswheels.p','rb'))
name = ['Base','Wheelbase','Truck height', 'Flat length', 'Tail inclination', 'Longboard', 'Plastic penny', 'Griptape penny', 'Wheel radius', 'All but tail', 'Full deck', 'Deck without tail length']
#%%
columns = ('1. Base','2. Longboard',f'3. Penny plastic ($\mu=0.2$)', f'4. Penny griptape ($\mu=0.8$)','empty')
rows = ['Dimensions','Max. height SB [m]','Max. speed SB [m/s]', 'Max. angular speed SB [rad/s]', 'Weight [kg]', 'Inertia [kg m^2]','Impact loss [J]', 'Max. height H [m]','Jump height [m]', 'Max. speed H [m/s]']

variables = [base,long,penn02,penn08]

fig, ax = plt.subplots()
fig.set_figwidth(15)
ax.set_xlim(-0.6464290098462666, 4.6174883113239025)
ax.set_ylim(-0.011666392205821278, 0.24499423632224634)
#fig.set_figheight(0.4)
ax.set_aspect('equal')
count = 0
plt.axis('off')


max_height_sb = len(columns)*[0]
max_speed_sb =len(columns)*[0]
max_rot_sb = len(columns)*[0]
max_height_h = len(columns)*[0]
max_speed_h = len(columns)*[0]
jump_height = len(columns)*[0]
weight = len(columns)*[0]
inertia = len(columns)*[0]
impact_loss = len(columns)*[0]
dimensions = len(columns)*[0]

for idx, i in enumerate(variables):
#for i in [base,wb,d_tr,l_f,phi]:
#for i in [base,wb,d_tr,l_f,phi]:
    max_height_sb[idx]=(-i['objective'])              
    max_speed_sb[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('dy_s'))]))
    max_rot_sb[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('dth_s'))]))
    max_height_h[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('y_h'))]))
    max_speed_h[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('dy_h'))]))
    idxAtf = len(i['time'][0]-1)
    idxBtf = idxAtf + len(i['time'][1]-1)
    jump_height[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('y_h'))])-i['sol_states'][i['all_vars'].index(sm.Symbol('y_h'))][idxAtf])
    plot_sb(i['p_opt'],i['all_vars'],i['sol_states'],i['coordinates'],count,ax)
    plt.plot(count, (r_w+d_tr-d_com).xreplace(i['p_opt']), color='black',
                    marker='o', markerfacecolor="None", markersize=7,alpha = 0.6)
    plt.plot(count, (r_w+d_tr-d_com).xreplace(i['p_opt']), color='black',
                   marker='x', markersize=4,alpha = 0.6)
    
    count += 1
    weight[idx]=(float(m_s.xreplace(i['p_opt'])))
    inertia[idx]=(float(I_s.xreplace(i['p_opt'])))
    KE_evalA = sm.lambdify([i['Astate'],i['p_opt'].keys()], KEs_A())
    KE_evalB = sm.lambdify([i['Bstate'],i['p_opt'].keys()], KEs_B())
    KE_solA = KE_evalA(i['state'][0],i['p_opt'].values())
    KE_solB = KE_evalB(i['state'][1],i['p_opt'].values())
    impact_loss[idx]=(KE_solB[0]-KE_solA[-1])
    
    dim = i['p_opt']
    dimensions[idx] = f'$l_{{wb}} =$%a'%np.round(dim[l_wb],2)+\
                      f', $l_d =$%a'%np.round(dim[l_f],2)+\
                      f', $l_t =$%a'%np.round(dim[l_t],2)+'\n'+\
                      f'$\phi =$%a'%np.round(np.rad2deg(dim[phi]),3)+\
                      f', $d_{{tr}} =$%a'%np.round(dim[d_tr],3)+\
                      f', $r_w =$%a'%np.round(dim[r_w],3)

    
data = [max_height_sb,
        max_speed_sb,
        max_rot_sb,
        weight,
        inertia,
        impact_loss,
        max_height_h,
        jump_height,
        max_speed_h,

        ]



# Get some pastel shades for the colors
#colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

# Initialize the vertical-offset for the stacked bar chart.
# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    if row == range(n_rows)[0]:
        #cell_text.append(dimensions)    
        cell_text.append(dimensions)
        
    cell_text.append(['%1.3f' %x for x in data[row]])

# Reverse colors and text labels to display the last value at the top.
the_table = ax.table(cellText=cell_text,
                      rowLabels=rows,
                      #rowColours=colors,
                      colLabels=columns,
                      loc='bottom',
                      fontsize = 'xx-small',
                      )
the_table.scale(1,11)
#the_table.FONTSIZE(8)
ax.set_xticks([])
ax.set_yticks([])
plt.subplots_adjust(bottom=0.5)
fig.tight_layout()
plt.savefig('/Users/j.t.heinen/Documents/Master Thesis/Results3/no_optimization_table_{}.png'.format('dpi600'),dpi=600)

#%%
columns = (f'1. Wheel radius',f'2. Wheelbase ', f'3. Truck height ', f'4. Deck length ',f'5. Tail length ',f'6. Tail inclination' )
rows = ['Optimized parameters','Dimensions','Max. height SB [m]','Max. speed SB [m/s]', 'Max. angular speed SB [rad/s]', 'Weight [kg]', 'Inertia [kg m^2]','Impact loss [J]', 'Max. height H [m]','Jump height [m]', 'Max. speed H [m/s]']

variables = [rw,wb,dtr,lf,lt,phi_]
OP = [ f'($r_w$)',f'($l_{{wb}}$)',f'($d_{{tr}}$)',f'($l_d$)',f'($l_t$)',f'($\phi$)']
fig, ax = plt.subplots()
fig.set_figwidth(15)
ax.set_xlim(-0.6464290098462666, 4.6174883113239025)
ax.set_ylim(-0.011666392205821278, 0.24499423632224634)
#fig.set_figheight(0.4)
ax.set_aspect('equal')
count = -0.2

plt.axis('off')

max_height_sb = len(columns)*[0]
max_speed_sb =len(columns)*[0]
max_rot_sb = len(columns)*[0]
max_height_h = len(columns)*[0]
max_speed_h = len(columns)*[0]
jump_height = len(columns)*[0]
weight = len(columns)*[0]
inertia = len(columns)*[0]
impact_loss = len(columns)*[0]
dimensions = len(columns)*[0]

for idx, i in enumerate(variables):
#for i in [base,wb,d_tr,l_f,phi]:
#for i in [base,wb,d_tr,l_f,phi]:
    max_height_sb[idx]=(-i['objective'])              
    max_speed_sb[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('dy_s'))]))
    max_rot_sb[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('dth_s'))]))
    max_height_h[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('y_h'))]))
    max_speed_h[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('dy_h'))]))
    idxAtf = len(i['time'][0]-1)
    idxBtf = idxAtf + len(i['time'][1]-1)
    jump_height[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('y_h'))])-i['sol_states'][i['all_vars'].index(sm.Symbol('y_h'))][idxAtf])
    plot_sb(i['p_opt'],i['all_vars'],i['sol_states'],i['coordinates'],count,ax)
    plt.plot(count, (r_w+d_tr-d_com).xreplace(i['p_opt']), color='black',
                    marker='o', markerfacecolor="None", markersize=7,alpha = 0.6)
    plt.plot(count, (r_w+d_tr-d_com).xreplace(i['p_opt']), color='black',
                   marker='x', markersize=4,alpha = 0.6)
    
    count += 0.867
    weight[idx]=(float(m_s.xreplace(i['p_opt'])))
    inertia[idx]=(float(I_s.xreplace(i['p_opt'])))
    KE_evalA = sm.lambdify([i['Astate'],i['p_opt'].keys()], KEs_A())
    KE_evalB = sm.lambdify([i['Bstate'],i['p_opt'].keys()], KEs_B())
    KE_solA = KE_evalA(i['state'][0],i['p_opt'].values())
    KE_solB = KE_evalB(i['state'][1],i['p_opt'].values())
    impact_loss[idx]=(KE_solB[0]-KE_solA[-1])
    
    dim = i['p_opt']
    dimensions[idx] = f'$l_{{wb}} =$%a'%np.round(dim[l_wb],2)+\
                      f', $l_d =$%a'%np.round(dim[l_f],2)+\
                      f', $l_t =$%a'%np.round(dim[l_t],2)+'\n'+\
                      f'$\phi =$%a'%np.round(np.rad2deg(dim[phi]),3)+\
                      f', $d_{{tr}} =$%a'%np.round(dim[d_tr],3)+\
                      f', $r_w =$%a'%np.round(dim[r_w],3)
    
data = [max_height_sb,
        max_speed_sb,
        max_rot_sb,
        weight,
        inertia,
        impact_loss,
        max_height_h,
        jump_height,
        max_speed_h,
        ]



# Get some pastel shades for the colors
#colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

# Initialize the vertical-offset for the stacked bar chart.
# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    if row == range(n_rows)[0]:
        cell_text.append(OP)
        cell_text.append(dimensions)
    cell_text.append(['%1.3f' %x for x in data[row]])
# Reverse colors and text labels to display the last value at the top.

the_table = ax.table(cellText=cell_text,
                      rowLabels=rows,
                      #rowColours=colors,
                      colLabels=columns,
                      loc='bottom',
                      fontsize = 'xx-small',
                      )
the_table.scale(1,11)
#the_table.FONTSIZE(8)
ax.set_xticks([])
ax.set_yticks([])
plt.subplots_adjust(bottom=0.6)
fig.tight_layout()
plt.savefig('/Users/j.t.heinen/Documents/Master Thesis/Results3/single_optimization_table_{}.png'.format('dpi600'),dpi=600)

#%%
columns = (f'1. All',f'2. No $l_t$', f'3. Deck and tail', f'4. Deck no $l_t$ ',f'5. All no mass')
variables = [all_,all_no_tail,all_notrrw,all_notrrwlt,all_nomass]
rows = ['Optimized parameters','Dimensions','Max. height SB [m]','Max. speed SB [m/s]', 'Max. angular speed SB [rad/s]', 'Weight [kg]', 'Inertia [kg m^2]','Impact loss [J]', 'Max. height H [m]','Jump height [m]', 'Max. speed H [m/s]']
OP = [ f'($l_{{wb}},l_d,l_t,\phi,d_{{tr}},r_w$)',
      f'($l_{{wb}},l_d,\phi,d_{{tr}},r_w$)',
      f'($l_{{wb}},l_d,l_t,\phi$)',
      f'($l_{{wb}},l_d,\phi$)',
      f'($l_{{wb}},l_d,l_t,\phi,d_{{tr}},r_w$)',
      ]

fig, ax = plt.subplots()
fig.set_figwidth(15)
ax.set_xlim(-0.6464290098462666, 4.6174883113239025)
ax.set_ylim(-0.011666392205821278, 0.24499423632224634)
#fig.set_figheight(0.4)
ax.set_aspect('equal')
count = 0
plt.axis('off')


max_height_sb = len(columns)*[0]
max_speed_sb =len(columns)*[0]
max_rot_sb = len(columns)*[0]
max_height_h = len(columns)*[0]
max_speed_h = len(columns)*[0]
jump_height = len(columns)*[0]
weight = len(columns)*[0]
inertia = len(columns)*[0]
impact_loss = len(columns)*[0]
dimensions = len(columns)*[0]

for idx, i in enumerate(variables):
#for i in [base,wb,d_tr,l_f,phi]:
#for i in [base,wb,d_tr,l_f,phi]:
    max_height_sb[idx]=(-i['objective'])              
    max_speed_sb[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('dy_s'))]))
    max_rot_sb[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('dth_s'))]))
    max_height_h[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('y_h'))]))
    max_speed_h[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('dy_h'))]))
    idxAtf = len(i['time'][0]-1)
    idxBtf = idxAtf + len(i['time'][1]-1)
    jump_height[idx]=(max(i['sol_states'][i['all_vars'].index(sm.Symbol('y_h'))])-i['sol_states'][i['all_vars'].index(sm.Symbol('y_h'))][idxAtf])
    plot_sb(i['p_opt'],i['all_vars'],i['sol_states'],i['coordinates'],count,ax)
    if idx < 4:     
        plt.plot(count, (r_w+d_tr-d_com).xreplace(i['p_opt']), color='black',
                        marker='o', markerfacecolor="None", markersize=7,alpha = 0.6)
        plt.plot(count, (r_w+d_tr-d_com).xreplace(i['p_opt']), color='black',
                       marker='x', markersize=4,alpha = 0.6)
    count += 1
    weight[idx]=(float(m_s.xreplace(i['p_opt'])))
    inertia[idx]=(float(I_s.xreplace(i['p_opt'])))
    KE_evalA = sm.lambdify([i['Astate'],i['p_opt'].keys()], KEs_A())
    KE_evalB = sm.lambdify([i['Bstate'],i['p_opt'].keys()], KEs_B())
    KE_solA = KE_evalA(i['state'][0],i['p_opt'].values())
    KE_solB = KE_evalB(i['state'][1],i['p_opt'].values())
    impact_loss[idx]=(KE_solB[0]-KE_solA[-1])
    
    dim = i['p_opt']
    dimensions[idx] = f'$l_{{wb}} =$%a'%np.round(dim[l_wb],2)+\
                      f', $l_d =$%a'%np.round(dim[l_f],2)+\
                      f', $l_t =$%a'%np.round(dim[l_t],2)+'\n'+\
                      f'$\phi =$%a'%np.round(np.rad2deg(dim[phi]),3)+\
                      f', $d_{{tr}} =$%a'%np.round(dim[d_tr],3)+\
                      f', $r_w =$%a'%np.round(dim[r_w],3)
    
nomass = dill.load(open('/Users/j.t.heinen/Documents/Master Thesis/Results3/all/nomass_IsMs.p','rb'))

def KEs_Anm():
    #V = (1/2)*nomass[1]*x_w**2 + (1/2)* (nomass[0]+nomass[1]*W1_a.pos_from(com_a).magnitude()**2)*dth_s**2 + (1/2)*m_h*sm.sqrt(dy_h**2+  dx_h**2)**2 
    V = nomass[1]*d2s(com_a.vel(N).magnitude())**2/2 + nomass[0] * dth_s**2/2 + m_h*sm.sqrt(dy_h**2 + dx_h**2)**2/2
    return V.xreplace(nomass[2])
def KEs_Bnm(): 
    #V = (1/2)*nomass[1]*sm.sqrt(dx_s**2 + dy_s**2) + (1/2) * nomass[0]* dth_s**2 + (1/2)*m_h*sm.sqrt(dy_h**2+  dx_h**2)**2 
    V = nomass[1]*d2s(com_b.vel(N).magnitude())**2/2 + nomass[0] * dth_s**2/2 + m_h*sm.sqrt(dy_h**2 + dx_h**2)**2/2
    return V
i = variables[-1]
weight[4]=(float(nomass[1].xreplace(i['p_opt'])))
inertia[4]=(float(nomass[0].xreplace(i['p_opt'])))
KE_evalAnm = sm.lambdify([i['Astate'],i['p_opt'].keys()], KEs_Anm())
KE_evalBnm = sm.lambdify([i['Bstate'],i['p_opt'].keys()], KEs_Bnm())
KE_solA = KE_evalAnm(i['state'][0],i['p_opt'].values())
KE_solB = KE_evalBnm(i['state'][1],i['p_opt'].values())
impact_loss[4]=(KE_solB[0]-KE_solA[-1])
plt.plot(4, (r_w+d_tr-nomass[2][d_com_]).xreplace(i['p_opt']), color='black',
                marker='o', markerfacecolor="None", markersize=7,alpha = 0.6)
plt.plot(4, (r_w+d_tr-nomass[2][d_com_]).xreplace(i['p_opt']), color='black',
               marker='x', markersize=4,alpha = 0.6)

data = [max_height_sb,
        max_speed_sb,
        max_rot_sb,
        weight,
        inertia,
        impact_loss,
        max_height_h,
        jump_height,
        max_speed_h,
]



# Get some pastel shades for the colors
#colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

# Initialize the vertical-offset for the stacked bar chart.
# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    if row == range(n_rows)[0]:
        cell_text.append(OP)
        cell_text.append(dimensions)
    cell_text.append(['%1.3f' %x for x in data[row]])

# Reverse colors and text labels to display the last value at the top.

the_table = ax.table(cellText=cell_text,
                      rowLabels=rows,
                      #rowColours=colors,
                      colLabels=columns,
                      loc='bottom',
                      fontsize = 'xx-small',
                      )
the_table.scale(1,11)
#the_table.FONTSIZE(8)
ax.set_xticks([])
ax.set_yticks([])
plt.subplots_adjust(bottom=0.6)
fig.tight_layout()
plt.savefig('/Users/j.t.heinen/Documents/Master Thesis/Results3/multi_optimize_table_{}.png'.format('dpi600'),dpi=600)

#%%
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
import dill


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
def weight(width_deck, mass_bearing, mass_truck, height_truck, height_truck0, width_wheel, n_ply, length_flat, length_tail, radius_wheel,
           rho_pu, rho_maple, rho_steel, m_glue, diameter_axle, d_veneer):

    mass_wheel = rho_pu * sm.pi * width_wheel * \
        ((2*radius_wheel)**2-diameter_axle**2) / 4  # V=pi*h*(D^2-d^2)/4

    mass_axle = sm.pi * (diameter_axle/2)**2 * width_deck * \
        rho_steel  # weight of axle, volume * steel 7700

    # rho_maple = 705; # kg/m3 https://www.wood-database.com/hard-maple/

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

        I_steiner.append(sm.trigsimp(mass[i]*d2s(sm.sqrt(com_points[i].pos_from(com).dot(A.x)**2+com_points[i].pos_from(com).dot(A.y)**2)**2)))
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
#p_jan = dict(zip(p, p_vals_jan))
#p_jan = dict(zip(p, p_vals_tobi))

#p_jan = dict(zip(p, p_vals_longboard))
p_jan = dict(zip(p, p_vals_penny))
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

def KEs_A():
    #V = (1/2)*m_s*x_w**2 + (1/2)* (I_s+m_s*W1_a.pos_from(com_a).magnitude()**2)*dth_s**2 + (1/2)*m_h*sm.sqrt(dy_h**2+  dx_h**2)**2 
    V = m_s*d2s(com_a.vel(N).magnitude())**2/2 + I_s * dth_s**2/2 + m_h*sm.sqrt(dy_h**2 + dx_h**2)**2/2
    return V.xreplace(quicksolve)
def KEs_B(): 
    #V = (1/2)*m_s*sm.sqrt(dx_s**2 + dy_s**2) + (1/2) * I_s* dth_s**2 + (1/2)*m_h*sm.sqrt(dy_h**2+  dx_h**2)**2 
    V = m_s*d2s(com_b.vel(N).magnitude())**2/2 + I_s * dth_s**2/2 + m_h*sm.sqrt(dy_h**2 + dx_h**2)**2/2
    return V
def dKE(pre):
    V = m_s*(x_s**2+y_s**2)/2 
    
