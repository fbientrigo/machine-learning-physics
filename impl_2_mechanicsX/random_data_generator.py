import integrate_data as intd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


#comparission for post processing
xreal, epsilon = 4.0, 5e-1


v=0 
def force_function(x,v):
    return (1/8000)*((x-1) * (x-11)**2 * (x-23)**2) - 0.7
    

def generate_data(args, force_function, xlow=-10.0, xhigh=20.0, vlow=-5.0, vhigh=5.0):
    # ===== inner definitions of the program ==================
    xlimits = [xlow, xhigh]
    vlimits = [vlow, vhigh]
    # the first tuple is low limits for both
    limits = list(zip(xlimits, vlimits))

    # where we save the data information
    global_data = []

    #starting the generation of random data
    random_xv = np.random.uniform(low=limits[0], high=limits[1], 
        size=(args.Nsamples,2))


    # defining the function for simulation
    def f_dottz(t, z):
        """function in the differential equation"""
        x, v = z
        dzdt = [v, force_function(x,v)]
        return dzdt


    # ====== Generation of data ================
    for k_data in range(args.Nsamples):
        # generate the series with the inputs
        xi = random_xv[k_data][0]
        vi = random_xv[k_data][1]

        # reconstruc the arguments with the ones that change
        arguments = intd.make_integrator_args(args.name,xi,vi,args.Nlayers, args.dtime)

        try:
            data_id, sol_estatus, data_generated = intd.run_integrator(
                arguments, vec_function=f_dottz, relative_tolerance=args.xtol)
            # we use the last data point
            data_point = [xi,vi,data_generated[-1][1], data_generated[-1][2] ]

        except ValueError:
            data_point = [xi, vi, np.nan, np.nan]

        global_data.append(data_point)


    total_data_frame = pd.DataFrame(global_data, columns=["xi", "vi", "xf", "vf"])
    
    return total_data_frame


def main(args, xlow=-10.0, xhigh=20.0, vlow=-5.0, vhigh=5.0):

    total_data_frame = generate_data(args, force_function, 
        xlow, xhigh, vlow, vhigh)




    # apply K
    total_data_frame["K"] = total_data_frame.apply(
        lambda row: 0 if abs(row["xf"] - xreal) <= epsilon else 
                    -1 if np.isnan(row["xf"]) else 1,
        axis=1
    )

    # data in the range where xf is close to xreal (normally set to 0)
    print(f"Total positive data K=0: {sum(total_data_frame['K'] == 0)} data points")

    total_data_frame.to_csv(f"./{args.name}")

    run_plots(total_data_frame, xlow, xhigh, vlow, vhigh, save_path=".")



# Plots together for obtaining data
def run_plots(data, xlow, xhigh, vlow, vhigh, save_path=None):
    """ runs inside of maain function, """
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
    parser.add_argument('-xtol',type=float, help='tolerance for integrator, recommended 5e-2')
    parser.add_argument('-Nlayers',type=float, default=50,help='Number of layers for NN')


    args = parser.parse_args()

    # for generation
    xlow=-10.0
    xhigh=20.0
    vlow=-5.0
    vhigh=5.0
    
    if args.name:
        main(args, xlow, xhigh, vlow, vhigh)
