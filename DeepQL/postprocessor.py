import integrate_data as intd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def narrow_gaussian(x, mu=0, sigma=0.01):
    """approximation of dirac delta, use a small sigma"""
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return pdf / pdf.max()


def post_process_data(data, xreal, epsilon):
    """reads data and runs postprocessor"""

    new_data = data.copy()

    # apply K
    new_data["K"] = data.apply(
        lambda row: 0 if abs(row["xf"] - xreal) <= epsilon else 
                    -1 if np.isnan(row["xf"]) else 1,
        axis=1)

    # apply continuous K
    new_data['c_K'] = narrow_gaussian(
        data['xf'], mu=xreal, sigma=epsilon)
    #data['c_K'] = data['c_K'].fillna(-1) # optional

    #print("data frame:")
    #print(data.head()[['xf', 'K', 'c_K']])

    return new_data




# Plots together for obtaining data
def p_run_plots(data, xreal, epsilon, force_function, save_path=None):
    """ runs inside of main function, used for soft debugging your data after generation
    requires a force_function(x,v)
    include v even if its fixed """

    xlow = data['xi'].min()
    xhigh = data['xi'].max()


    # apply K for quick view
    data["K"] = data.apply(
        lambda row: 0 if abs(row["xf"] - xreal) <= epsilon else 
                    -1 if np.isnan(row["xf"]) else 1,
        axis=1
    )

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,10))

    ax[0][0].set_title("Force function used")
    x_plot = np.linspace(xlow, xhigh, 400)
    ax[0][0].plot(x_plot, force_function(x_plot,v=0))
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



def main(args):
    name = args.name
    xreal = args.xreal
    epsilon = args.epsilon

    data = pd.read_csv(f"./{name}.csv")

    data_post = post_process_data(data, xreal, epsilon)

    data_post.to_csv(f"./{name}.csv")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrator for 2nd Order Differential Equation, it creates plot images and txt file with parameters')
    parser.add_argument('-name', type=str, help='Name of file to take as input, should be a .csv')
    # post processing arguments
    parser.add_argument('-xreal',type=float, default=0.0,help='Controls when a data is considered positive, K=0, if close to xreal')
    parser.add_argument('-epsilon',type=float, default=1e-1,help='bigger means more points are considered closer to xreal')
    
    args = parser.parse_args()

    
    if args.name:
        main(args)

