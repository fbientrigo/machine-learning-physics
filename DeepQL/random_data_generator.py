from integrate_data import make_integrator_args, run_integrator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse



v=0 
def force_function(x,v):
    return (1/8000)*((x-1) * (x-11)**2 * (x-23)**2) - 0.7


def random_data_generation( Nsamples,Nlayers ,dtime ,xtol ,force_function, xlim=(-1,1), vlim=(-1,1)):
    """
    main function, generates an ammount of Nsamples with random initial conditions
    and runs a scipy time integrator; the ammount of steps is controlled by Nlayers
    referencing the Neural Network the data is for

    returns a pandas dataframe with the data
    with columns columns=["xi", "vi", "xf", "vf"]
    """
    
    xlow, xhigh = xlim[0], xlim[1]
    vlow, vhigh = vlim[0], vlim[1]

    xlimits = [xlow, xhigh]
    vlimits = [vlow, vhigh]
    # the first tuple is low limits for both
    limits = list(zip(xlimits, vlimits))

    # where we save the data information
    global_data = []

    #starting the generation of random data
    random_xv = np.random.uniform(low=limits[0], high=limits[1], 
        size=(Nsamples,2))


    # defining the function for simulation
    def f_dottz(t, z):
        """function in the differential equation"""
        x, v = z
        dzdt = [v, force_function(x,v)]
        return dzdt


    # ====== Generation of data ================
    for k_data in range(Nsamples):
        # generate the series with the inputs
        xi = random_xv[k_data][0]
        vi = random_xv[k_data][1]

        # reconstruc the arguments with the ones that change
        name = "_"
        arguments = make_integrator_args(name,xi,vi,Nlayers, dtime)

        try:
            data_id, sol_estatus, data_generated = run_integrator(
                arguments, vec_function=f_dottz, relative_tolerance=xtol)
            # we use the last data point
            data_point = [xi,vi,data_generated[-1][1], data_generated[-1][2] ]

        except ValueError:
            data_point = [xi, vi, np.nan, np.nan]

        global_data.append(data_point)


    total_data_frame = pd.DataFrame(global_data, columns=["xi", "vi", "xf", "vf"])
    
    return total_data_frame


def generate_data(args, force_function, xlow=-10.0, xhigh=20.0, vlow=-5.0, vhigh=5.0):
    """ for using as command line, prefer random_data_generation for scripts """

    Nsamples = args.Nsamples
    Nlayers = args.Nlayers
    dtime = args.dtime
    xtol = args.xtol 

    data = random_data_generation(Nsamples, Nlayers, 
        dtime, xtol,force_function, xlim=(xlow,xhigh), vlim=(vlow,vhigh))

    return data




def main(args, xlow=-10.0, xhigh=20.0, vlow=-5.0, vhigh=5.0):

    total_data_frame = generate_data(args, force_function, 
        xlow, xhigh, vlow, vhigh)

    total_data_frame.to_csv(f"./{args.name}")

    return total_data_frame



# Plots together for obtaining data
def run_plots(data, xreal, epsilon, xlow, xhigh, vlow, vhigh, save_path=None):
    """ runs inside of main function, used for soft debugging your data after generation """

    # apply K for quick view
    data["K"] = data.apply(
        lambda row: 0 if abs(row["xf"] - xreal) <= epsilon else 
                    -1 if np.isnan(row["xf"]) else 1,
        axis=1
    )

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,10))

    ax[0][0].set_title("Force function used")
    x_plot = np.linspace(xlow, xhigh, 400)
    ax[0][0].plot(x_plot, force_function(x_plot,v))
    ax[0][0].set_xlabel('x')

    ax[0][1].set_title("Random data generation")
    ax[0][1].scatter(data.xi, data.vi)
    ax[0][1].set_xlabel('x')
    ax[0][1].set_ylabel('v')

    ax[1][0].set_title("Cross of xreal data")
    ax[1][0].plot(data.xf)
    ax[1][0].axhline(y=xreal, color='r', linestyle='-')


    ax[1][1].set_title("Classification of data")
    colors = {-1: 'red', 0: 'green', 1: 'blue'}
    labels = {-1: 'K=-1', 0: 'K=0', 1: 'K=1'}
    ax[1][1].scatter(data['xi'], data['vi'], 
        c=data['K'].apply(lambda x: colors[x]),
        label=data['K'].apply(lambda x: labels[x]))
    ax[1][1].set_xlabel('xi')
    ax[1][1].set_ylabel('vi')
    legend_elements = [(labels[x], colors[x]) for x in data['K'].unique()]
    ax[1][1].legend(handles=[plt.Line2D([], [], marker='o', color='w', label=label, markerfacecolor=color, markersize=8) for label, color in legend_elements], loc='best')

    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrator for 2nd Order Differential Equation, it creates plot images and txt file with parameters')
    parser.add_argument('-name', type=str, help='Name for saving the file, example: file.csv')
    parser.add_argument('-Nsamples',type=int, help='specifies ammounts of jump steps, recommended around 1e3 ')
    parser.add_argument('-dtime',type=float, default=1e-3, help='specifies discrete jumps of time in integrator, recommended 5e-2')
    parser.add_argument('-xtol',type=float, default=1e-2, help='tolerance for integrator, recommended 5e-2')
    parser.add_argument('-Nlayers',type=float, default=50,help='Number of layers for NN')
    parser.add_argument('-xreal',type=float, help='Controls when a data is considered positive, K=0, if close to xreal')
    parser.add_argument('-epsilon',type=float, default=1e-1,help='bigger means more points are considered closer to xreal')
    
    args = parser.parse_args()

    # fixed for command line generation, change them here
    xlow=-10.0
    xhigh=20.0
    vlow=-5.0
    vhigh=5.0
    
    if args.name:
        main(args, xlow, xhigh, vlow, vhigh)

    if args.xreal:
        xreal = args.xreal
        epsilon = args.epsilon

        total_data_frame = main(args, xlow, xhigh, vlow, vhigh)
        run_plots(total_data_frame, xreal, epsilon, xlow, xhigh, vlow, vhigh, save_path=".")
