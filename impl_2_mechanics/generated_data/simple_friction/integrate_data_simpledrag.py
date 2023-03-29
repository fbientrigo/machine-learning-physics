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

#   Modify this function to generate new simulations


# =========force function here============== <<<<
g = 9.81
m = 1.0

input_ = np.linspace(0, 10, 100) # limits

def force_function(v):
    """ In some cases is purely a function of position, other on velocity """
    return - g + v**2 / m

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
    plt.savefig(f'data/force.png')
    plt.show()
    

    data = np.vstack((input_, force_array)).T

    with open(f'data/force.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['input', 'force'])
        writer.writerows(data)

def main(args):
    # the end time is controlled by the number of points
    N = 1000000
    # harcodding the dt so is always fixed for a NN
    dt = 1e-5

    z0 = [100.0, 0.0]  # initial condition: y=1, v=0
    
    # tol = 1e-5  # tolerance for error control

    t0 = 0.0
    tf = t0 + N * dt
    print(f't0: {t0}; tf: {tf}')

    t_span = [t0,tf]
    t_eval = np.arange(start= t0, stop=tf, step=dt)

    # ======= writting the data
    with open(f'data/{args.name}_parameters.txt', 'w', newline='') as file:
        file.write(f'N: {N} \n')
        file.write(f'dt: {dt} \n')
        file.write(f't0: {t0}, tf= {tf} \n')
        file.write(f'z0: {z0} \n')
        # file.write(f'tolerance: {tol} \n')
    # ======

    sol = solve_ivp(fun=lambda t, z: f(t, z), t_span=t_span, 
        y0=z0, t_eval=t_eval, vectorized=True)


    if sol.status == 0:
        y = sol.y[0] # position
        v = sol.y[1] # velocity
        data = np.vstack((sol.t, y, v)).T

        with open(f'data/{args.name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time', 'position', 'velocity'])
            writer.writerows(data)

        plt.title(f'{args.name}')
        plt.plot(sol.t, y, label='rk4 - position')
        plt.plot(sol.t, v, label='rk4 - velocity')
        plt.xlabel('Time')
        plt.legend()
        plt.savefig(f'data/{args.name}.png')
        plt.show()

    else:
        print('Solver failed to converge!')
        print(sol.message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrator for 2nd Order Differential Equation')
    parser.add_argument('-name', type=str, help='Name of the CSV file')
    parser.add_argument('--force', help='exports and plots force data', action='store_true')
    args = parser.parse_args()
    
    # if -force is used, only execute the export_force_function()
    if args.force:
        export_force_function()
    
    if args.name:
        main(args)
