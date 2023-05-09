"""
@author: j.t.heinen
"""

def Mass_model(width_deck, mass_bearing, mass_truck, height_truck, height_truck0, width_wheel, n_ply, length_flat, length_tail, radius_wheel,
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

def Inertia_model(mass, com_points, com, majordim, shape):
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
            s = sm.sqrt((majordim[i][0]/2)**2+majordim[i][1]**2)
            beta = 2*sm.asin((majordim[i][0]/2)/s)
            I_com.append((mass[i]/2)*s**2*(1-(2/3)*sm.sin(beta)))

        if shape[i] == 'cylinder':
            I_com.append((1/2)*mass[i]*(majordim[i]/2)**2)
        #Trigsimp was sometimes necesarry due to a theta still being in there
        I_steiner.append(sm.trigsimp(mass[i]*d2s(sm.sqrt(com_points[i].pos_from(com).dot(A.x)**2+com_points[i].pos_from(com).dot(A.y)**2)**2)))
    I_tot = sum(I_com)+sum(I_steiner)
    return I_tot, I_com, I_steiner