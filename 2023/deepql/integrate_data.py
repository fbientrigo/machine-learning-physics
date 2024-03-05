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


# =========force function here============== <<<<

def f_function(v):
    """ example function """
    return  - v


# ==================================================================
# Inner workings of code:
# ===========================================


def compile_data_to_csv(data, name, include_id=False):
    """save data to the folder gen_data"""
    with open(f'gen_data/{name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        if include_id:
            writer.writerow(['id_conditions','time', 'position', 'velocity'])
        else:
            writer.writerow(['time', 'position', 'velocity'])
        writer.writerows(data)



def command_line_compile_to_csv(data, args):
    N = args.N
    z0 = [args.xi, args.vi]  # initial condition: y=1, v=0
    t0 = 0.0
    dt = args.dtime
    name = args.name
    
    compile_data_to_csv(data, name, include_id=False)

    # ======= writting the data configuration
    with open(f'gen_data/{args.name}_parameters.txt', 'w', newline='') as file:
        file.write(f'N: {N} \n')
        file.write(f'dt: {dt} \n')
        file.write(f't0: {t0}, tf= {tf} \n')
        file.write(f'z0: {z0} \n')
        # file.write(f'tolerance: {tol} \n')
    # ======

#     The main function here
def integrate_diffeq(N, xi, vi, dt, debug=False, method="RK45",
    force_function=f_function, force_type="speed", xtol=1e-3, atol=1e-2,
    vectorize=True):
    """
    uses scipy.solve_ivp, with the RK45; if the integrator converge
    it gives out an array: (data_id, t, y, v)
    - data_id are the initial conditions as string
    - t is the time of every data point
    - y is the position at time t
    - v is the speed at time t

    - force_type: "position" or "speed" or "both"
    """

    z0 = [xi, vi]  # initial condition: y=1, v=0
    t0 = 0.0    
    tf = t0 + N * dt

    if debug:
        print(f't0: {t0}; tf: {tf}, dt: {dt}')

    t_span = [t0,tf]
    t_eval = np.arange(start= t0, stop=tf, step=dt)

    def vec_function(t, z):
        """creates the function in a vectorized way"""
        y, v = z
        if force_type == "position":
            dzdt = [v, force_function(y) ]
        elif force_type == "speed":
            dzdt = [v, force_function(v) ]
        elif force_type == "both":
            dzdt = [v, force_function(y,v) ]
        else:
            raise ValueError(f"Unknown force_type: {force_type}")
        return dzdt

    # Integrator
    sol = solve_ivp(fun=lambda t, z: vec_function(t, z), 
        method= method,
        t_span=t_span, y0=z0, t_eval=t_eval, vectorized=True, 
        atol=atol, rtol=xtol)


    # Given state of solution
    if sol.status == 0:
        
        y = np.array(sol.y[0], dtype='float64') # position
        v = np.array(sol.y[1], dtype='float64') # velocity

        # this should be a float
        data = np.vstack((sol.t, y, v)).T

        return f"x{xi}-v{vi}", sol.status, data

    else:
        print('Solver failed to converge!')
        if debug:
            print(sol.message)

        return sol.status, np.array([np.nan, np.nan, np.nan, np.nan])


def make_integrator_args(name, xi, vi, N, dt):
    """
    Compiles in a pandas series all of the essential arguments for integrator
    >> arguments = pd.Series({'name': name, 'xi': xi, 'vi':vi, 'N':N, 'dt':dt })
    >> solution_status, data = run_integrator( arguments )
    """

    import pandas as pd

    integrator_args = pd.Series(
        {'name': name, 'xi': xi, 'vi':vi, 'N':N, 'dtime':dt }
    )
    return integrator_args

def run_integrator(args, debug=False, force_function=f_function):
    """
    it needs input as a pandas Series
    >> arguments = pd.Series({'name': name, 'xi': xi, 'vi':vi, 'N':N, 'dtime':dt })
    >> solution_status, data = run_integrator( arguments )

    by default it used the function defined in the script
    """
      
    # ================ change parameters here   =========
    N = args.N
    xi = args.xi
    vi = args.vi
    dt = args.dtime

    return integrate_diffeq(N, xi, vi, dt, 
        debug=False, force_function=force_function)




def main(args):

    data_id, solution_status, data = run_integrator(args)

    if solution_status == 0:
        # compile data sections
        print("Compilation function goes here")
        command_line_compile_to_csv(data, args)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrator for 2nd Order Differential Equation, it creates plot images and txt file with parameters')
    parser.add_argument('-name', type=str, help='Name of the files, recommended the standard [force_name]_[v_speed_initial], because we always use dt=1e-3 the name will include dt1e-3 at the end')
    parser.add_argument('-xi',type=float, help='specifies initial position')
    parser.add_argument('-vi',type=float, help='specifies initial velocity')
    parser.add_argument('-N',type=int, help='specifies ammounts of jump steps, recommended around 1e3 ')
    parser.add_argument('-dtime',type=float, help='specifies discrete jumps of time in integrator, recommended 5e-2')
    args = parser.parse_args()
    
    
    if args.name:
        main(args)
