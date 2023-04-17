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

# global parameters of tolerance for integrator

absolute_tolerance = 1e-6

relative_tolerance = 1e-6



# =========force function here============== <<<<

input_ = np.linspace(0, 250, 500) # limits

def force_function(x):
    """ this function """
    return  ( (x-1) * (x-11)**2 * (x-23)**2 / 8000 ) - 0.7





# ==================================================================
# Inner workings of code:
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


def compile_data_to_csv(data, args):


    N = args.N
    z0 = [args.xi, args.vi]  # initial condition: y=1, v=0
    t0 = 0.0
    dt = args.dt

    tf = t0 + N * dt

    t_span = [t0,tf]
    t_eval = np.arange(start= t0, stop=tf, step=dt)

    with open(f'gen_data/{args.name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id_conditions','time', 'position', 'velocity'])
        writer.writerows(data)


    # ======= writting the data configuration
    with open(f'gen_data/{args.name}_parameters.txt', 'w', newline='') as file:
        file.write(f'N: {N} \n')
        file.write(f'dt: {dt} \n')
        file.write(f't0: {t0}, tf= {tf} \n')
        file.write(f'z0: {z0} \n')
        # file.write(f'tolerance: {tol} \n')
    # ======



def run_integrator(args):
      
    # ================ change parameters here   =========
    N = args.N
    z0 = [args.xi, args.vi]  # initial condition: y=1, v=0
    t0 = 0.0
    dt = args.dt

    # id for saving
    data_id = f"xi_{args.xi}_vi_{args.vi}" # for every initial condition


    # ===================================================
    # harcodding the dt so is always fixed for a NN dt=1e-3
    # dont change unless you change your NN and regenerate all data

    
    tf = t0 + N * dt
    print(f't0: {t0}; tf: {tf}, dt: {dt}')

    t_span = [t0,tf]
    t_eval = np.arange(start= t0, stop=tf, step=dt)

    # Integrator
    sol = solve_ivp(fun=lambda t, z: f(t, z), t_span=t_span, 
        y0=z0, t_eval=t_eval, vectorized=True, atol=absolute_tolerance, rtol=relative_tolerance)


    # Given state of solution
    if sol.status == 0:
        y = sol.y[0] # position
        v = sol.y[1] # velocity
        data_id = np.array([data_id]*len(sol.t)).reshape(1,-1)

        data = np.vstack((data_id, sol.t, y, v)).T

        return sol.status, data

    else:
        print('Solver failed to converge!')
        print(sol.message)

        return sol.status, np.array([np.nan, np.nan, np.nan, np.nan])



def main(args):

    solution_status, data = run_integrator(args)

    if solution_status == 0:
        # compile data sections
        print("Compilation function goes here")
        compile_data_to_csv(data, args)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrator for 2nd Order Differential Equation, it creates plot images and txt file with parameters')
    parser.add_argument('-name', type=str, help='Name of the files, recommended the standard [force_name]_[v_speed_initial], because we always use dt=1e-3 the name will include dt1e-3 at the end')
    parser.add_argument('--force', help='exports and plots force data', action='store_true')
    parser.add_argument('-xi',type=float, help='specifies initial position')
    parser.add_argument('-vi',type=float, help='specifies initial velocity')
    parser.add_argument('-N',type=int, help='specifies ammounts of jump steps, recommended around 1e3 ')
    parser.add_argument('-dt',type=float, help='specifies discrete jumps of time in integrator, recommended 5e-2')
    args = parser.parse_args()
    
    # if -force is used, only execute the export_force_function()
    if args.force:
        export_force_function()
    
    if args.name:
        main(args)
