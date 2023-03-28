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




def main(args):
    
    
    # save training parameters and others
    with open(f'data/{args.name}_parameters.txt', 'w', newline='') as file:
        file.write(f'N: {N} \n')
        file.write(f'dt: {dt} \n')
        file.write(f't0: {t0}, tf= {tf} \n')
        file.write(f'z0: {z0} \n')
        # file.write(f'tolerance: {tol} \n')


        # with open(f'data/{args.name}.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['time', 'position', 'velocity'])
        #     writer.writerows(data)

        # plt.title(f'{args.name}')
        # plt.plot(sol.t, y, label='rk4 - position')
        # plt.plot(sol.t, v, label='rk4 - velocity')
        # plt.xlabel('Time')
        # plt.legend()
        # plt.savefig(f'data/{args.name}.png')
        # plt.show()


if __name__ == '__main__':
    # argparsing
    parser = argparse.ArgumentParser(description='Integrator for 2nd Order Differential Equation')
    parser.add_argument('-name', type=str, help='Name of the CSV file')
    parser.add_argument('--force', help='exports and plots force data', action='store_true')
    args = parser.parse_args()
    
    # if -force is used, only execute the export_force_function()
    if args.force:
        export_force_function()
    
    if args.name:
        main(args)
