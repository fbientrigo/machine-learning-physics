import integrate_data as intd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import torch

#input of global program (data about our network)
N_layers = 50
dt = 5e-2
name = "exp_random_data"
relative_tolerance = 1e-3

#comparission for post processing
xreal, epsilon = 4.0, 5e-1

# function for the differential equation
alpha = 1
def force_function(x):
    # return - alpha * x * x
    return (1/8000)*((x-1) * (x-11)**2 * (x-23)**2) - 0.7
    

def generate_data(force_function, xlow=-10.0, xhigh=20.0, ylow=-5.0, yhigh=5.0, N_samples=1000):
    # ===== inner definitions of the program ==================
    xlimits = [xlow, xhigh]
    vlimits = [vlow, vhigh]
    # the first tuple is low limits for both
    limits = list(zip(xlimits, vlimits))

    # where we save the data information
    global_data = []

    #starting the generation of random data
    random_xv = np.random.uniform(low=limits[0], high=limits[1], 
        size=(N_samples,2))


    # defining the function for simulation
    def f_dottz(t, z):
        """function in the differential equation"""
        x, v = z
        dzdt = [v, force_function(x)]
        return dzdt


    # ===== inner definitions of the program ==================
    xlimits = [xlow, xhigh]
    vlimits = [vlow, vhigh]
    # the first tuple is low limits for both
    limits = list(zip(xlimits, vlimits))


    # where we save the data information
    global_data = []

    #starting the generation of random data
    random_xv = np.random.uniform(low=limits[0], high=limits[1], 
        size=(N_samples,2))


    # defining the function for simulation
    def f_dottz(t, z):
        """function in the differential equation"""
        x, v = z
        dzdt = [v, force_function(x)]
        return dzdt


    # ====== Generation of data ================
    for k_data in range(N_samples):
        # generate the series with the inputs
        xi = random_xv[k_data][0]
        vi = random_xv[k_data][1]

        arguments = intd.make_integrator_args(name,xi,vi,N_layers, dt)

        try:
            data_id, sol_estatus, data_generated = intd.run_integrator(
                arguments, vec_function=f_dottz, relative_tolerance=relative_tolerance)
            # we use the last data point
            data_point = [xi,vi,data_generated[-1][1], data_generated[-1][2] ]

        except ValueError:
            data_point = [xi, vi, np.nan, np.nan]

        global_data.append(data_point)


    total_data_frame = pd.DataFrame(global_data, columns=["xi", "vi", "xf", "vf"])
    
    return total_data_frame


def main():
    pass






# Plots together for obtaining data
def run_plots(save_path=None):
    """ runs inside of maain function, """

    # plot of force function
    x_plot = np.linspace(xlow, xhigh, 400)
    plt.plot(x_plot, force_function(x_plot))
    plt.title("Force function used")
    plt.xlabel('x')
    plt.show()

    # plotting of the distribution
    plt.scatter(random_xv[:,0], random_xv[:,1])
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title("Random data generation")
    plt.show()

    # data and xreal
    plt.plot(total_data_frame.xf)
    plt.axhline(y=xreal, color='r', linestyle='-')

    if save_path != None:
        plt.savefig(save_path)

    plt.show()






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrator for 2nd Order Differential Equation, it creates plot images and txt file with parameters')
    parser.add_argument('-name', type=str, help='Name of the files, recommended the standard [force_name]_[v_speed_initial], because we always use dt=1e-3 the name will include dt1e-3 at the end')
    parser.add_argument('--force', help='exports and plots force data', action='store_true')
    parser.add_argument('-xi',type=float, help='specifies initial position')
    parser.add_argument('-vi',type=float, help='specifies initial velocity')
    parser.add_argument('-N',type=int, help='specifies ammounts of jump steps, recommended around 1e3 ')
    parser.add_argument('-dtime',type=float, help='specifies discrete jumps of time in integrator, recommended 5e-2')
    args = parser.parse_args()

    
    if args.name:
        main(args)
