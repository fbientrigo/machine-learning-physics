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
#
#


# =========force function here============== <<<<
g = 9.81
m = 1.0

def force_function(v):
    """ In some cases is purely a function of position, other on velocity """
    return -g - v**2/m

# ===========================================


def f(t, z, omega):
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

    force_array = force_function(input)

    plt.title(f'Force function F({input})')
    plt.plot( input, force_array )
    plt.grid()
    plt.show()
    plt.savefig(f'data/force.png')

    data = np.vstack((input, force_array)).T

def main(args):
    N = 1000
    t0 = 0.0  # initial time
    tf = 10.0  # final time
    z0 = [1.0, 0.0]  # initial condition: y=1, v=0
    omega = 2.0  # frequency of oscillation
    tol = 1e-6  # tolerance for error control

    # ==================================

    t_span = [t0, tf]
    t_eval = np.linspace(t0, tf, 1000)

    sol = solve_ivp(fun=lambda t, z: f(t, z, omega), t_span=t_span, y0=z0, t_eval=t_eval, rtol=tol, atol=tol, vectorized=True)

    if sol.status == 0:
        y = sol.y[0] # position
        v = sol.y[1] # velocity
        data = np.vstack((sol.t, y, v)).T

        with open(f'data/{args.name}', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time', 'position', 'velocity'])
            writer.writerows(data)

        plt.title(f'{args.name}')
        plt.plot(sol.t, y, label='rk4 - position')
        plt.plot(sol.t, v, label='rk4 - velocity')

        # You can plot the known solution if you wish
        #
        plt.plot(sol.t, np.cos(omega*sol.t), 'r--',label='analytical position', alpha=0.5)
        #

        plt.xlabel('Time')
        plt.legend()
        plt.show()

    else:
        print('Solver failed to converge!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrator for 2nd Order Differential Equation')
    parser.add_argument('-name', type=str, required=True, help='Name of the CSV file')
    parser.add_argument('-force', type=bool, default=False, help='exports and plots force data')
    args = parser.parse_args()

    # if -force is used, only execute the export_force_function()

    # if -foce is not used, run main
    main(args)
