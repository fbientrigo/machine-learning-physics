import argparse
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import csv


#========================================================
#       Integrator for 2nd Order Equation
#   uses RK5 order with error bound assuming RK4
#   generates a CSV with the name given to the program
#
#   python integrate_data.py -name "name.csv"
#========================================================

#   configurations for the global model

dt = 5e-2


# =========force function here============== <<<<
g = 9.81
m = 1.0

input_ = np.linspace(0, 250, 500) # limits

def force_function(x):
    """ this function """
    return  ( (x-1) * (x-11)**2 * (x-23)**2 / 8000 ) - 0.7

# ===========================================


def f(t, z):
    """function in the differential equation"""
    y, v = z

    # ======== UPDATE HERE IF NEW force_function ================= <<<<
    # dzdt = [v, force_function(y,v)]
    dzdt = [v, force_function(v) ]
    # ============================================================
    return dzdt


def export_force_function():
    """
    plots the force function and export a csv with the real force data
    used for training
    """

    force_array = force_function(input_)

    plt.title(f'Force function Force')
    plt.plot( input_, force_array )
    plt.grid()
    plt.savefig(f'gen_data/force.png')
    plt.show()
    

    data = np.vstack((input_, force_array)).T

    with open(f'gen_data/force.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['input', 'force'])
        writer.writerows(data)


def main(args):
    
    # ================ change parameters here   =========
    N = args.N
    z0 = [args.xi, args.vi]  # initial condition: y=1, v=0
    t0 = 0.0


    # ===================================================
    # harcodding the dt so is always fixed for a NN dt=1e-3
    # dont change unless you change your NN and regenerate all data

    
    tf = t0 + N * dt
    print(f't0: {t0}; tf: {tf}, dt: {dt}')

    t_span = [t0,tf]
    t_eval = np.arange(start= t0, stop=tf, step=dt)



    sol = solve_ivp(fun=lambda t, z: f(t, z), t_span=t_span, 
        y0=z0, t_eval=t_eval, vectorized=True)


    if sol.status == 0:
        y = sol.y[0] # position
        v = sol.y[1] # velocity
        data = np.vstack((sol.t, y, v)).T

        with open(f'gen_data/{args.name}_dt1e-3.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time', 'position', 'velocity'])
            writer.writerows(data)

        plt.title(f'{args.name}')
        plt.plot(sol.t, y, label='rk4 - position')
        plt.plot(sol.t, v, label='rk4 - velocity')
        plt.xlabel('Time')
        plt.grid()
        plt.legend()
        plt.savefig(f'gen_data/{args.name}_dt1e-3.png')
        plt.show()

    else:
        print('Solver failed to converge!')
        print(sol.message)

    # ======= writting the data
    with open(f'gen_data/{args.name}_parameters.txt', 'w', newline='') as file:
        file.write(f'N: {N} \n')
        file.write(f'dt: {dt} \n')
        file.write(f't0: {t0}, tf= {tf} \n')
        file.write(f'z0: {z0} \n')
        file.write(f'position_min: {np.min(y)}, position_max: {np.max(y)} \n')
        file.write(f'velocity_min: {np.min(v)}, velocity_max: {np.max(v)} \n')
        # file.write(f'tolerance: {tol} \n')
    # ======



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrator for 2nd Order Differential Equation, it creates plot images and txt file with parameters')
    parser.add_argument('-name', type=str, help='Name of the files, recommended the standard [force_name]_[v_speed_initial], because we always use dt=1e-3 the name will include dt1e-3 at the end')
    parser.add_argument('--force', help='exports and plots force data', action='store_true')
    parser.add_argument('-xi',type=float, help='specifies initial position')
    parser.add_argument('-vi',type=float, help='specifies initial velocity')
    parser.add_argument('-N',type=int, help='specifies ammounts of jump steps, recommended around 1e3 ')
    args = parser.parse_args()
    
    # if -force is used, only execute the export_force_function()
    if args.force:
        export_force_function()
    
    if args.name:
        main(args)
