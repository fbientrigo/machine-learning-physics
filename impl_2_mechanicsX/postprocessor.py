import integrate_data as intd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


# To write


def main(args, xlow=-10.0, xhigh=20.0, vlow=-5.0, vhigh=5.0):

    # reads csv and runs post processor

    data = generate_data(args, force_function, 
        xlow, xhigh, vlow, vhigh)

    data = pd.read_csv(f"./{args.name}.csv")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrator for 2nd Order Differential Equation, it creates plot images and txt file with parameters')
    parser.add_argument('-name', type=str, help='Name of the files, recommended the standard [force_name]_[v_speed_initial], because we always use dt=1e-3 the name will include dt1e-3 at the end')
    parser.add_argument('-Nsamples',type=int, help='specifies ammounts of jump steps, recommended around 1e3 ')
    parser.add_argument('-dtime',type=float, default=1e-3, help='specifies discrete jumps of time in integrator, recommended 5e-2')
    parser.add_argument('-xtol',type=float, help='tolerance for integrator, recommended 5e-2')
    parser.add_argument('-Nlayers',type=float, default=50,help='specifies discrete jumps of time in integrator, recommended 5e-2')





    args = parser.parse_args()

    
    if args.name:
        main(args)
