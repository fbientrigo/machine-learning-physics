import integrate_data as intd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def narrow_gaussian(x, mu=0, sigma=0.01):
    """approximation of dirac delta, use a small sigma"""
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return pdf / pdf.max()



def main(args):
    # reads csv and runs post processor
    data = pd.read_csv(f"./{args.name}.csv")

    total_data_frame['c_K'] = narrow_gaussian(
        total_data_frame['xf'], mu=xreal, sigma=args.epsilon)
    #total_data_frame['c_K'] = total_data_frame['c_K'].fillna(-1) # optional

    print("data frame:")
    print(data.head()[['xf', 'K', 'c_K']])




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrator for 2nd Order Differential Equation, it creates plot images and txt file with parameters')
    parser.add_argument('-name', type=str, help='Name of file to take as input, should be a .csv')
    # post processing arguments
    parser.add_argument('-xreal',type=float, default=0.0,help='Controls when a data is considered positive, K=0, if close to xreal')
    parser.add_argument('-epsilon',type=float, default=1e-1,help='bigger means more points are considered closer to xreal')
    


    args = parser.parse_args()

    
    if args.name:
        main(args)

