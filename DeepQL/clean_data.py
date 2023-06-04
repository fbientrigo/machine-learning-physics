import pandas as pd
import argparse
import glob

#   
#   Library for reordering the data from integrate_data.py
#   it is able to multiply the ammount of data points
#   from a long simulation
#


def separate_data(df, run_length=50, xmax=None, vmax=None,
    position='position', velocity='velocity',
    spacing=1,
    x0='x_initial', xf='x_step',
    v0='v_initial', vf='v_step'):
    """
    given a long simulation, cut it into run_length pieces
    long simulation: [1,2,3,4,5...]
    will end up in:
        [1,2,3,4,5...]
        [2,3,4,5...  ]
        [3,4,5...    ]
        ...

    its recommended to be the depth of your network
    
    using the information on position and velocity

    spacing=1 will produce most data
    spacing=10 will only add every 10th point
    """

    # filter data:
    if vmax:
        print(f"len of dataframe before vmax: {len(df)}")
        bool_mask = df[velocity] <= vmax
        df = df[ bool_mask ].reset_index(drop=True)
        # its important to reset index or will get errors when trying to acces non existen indexs
        print(f"len of dataframe after vmax: {len(df)}")
        
    if xmax:
        print(f"len of dataframe before xmax: {len(df)}")
        bool_mask = df[position] <= xmax
        df = df[ bool_mask ].reset_index(drop=True)
        print(f"len of dataframe after xmax: {len(df)}")

    # re order in time steps for NN
    data_worked = pd.DataFrame()
    x_initial = []
    v_initial = []
    x_steps = [[] for _ in range(run_length//10)]
    v_steps = [[] for _ in range(run_length//10)]

    for index in range(run_length, len(df) - 1, spacing):
        x_initial.append(df[position][index - run_length])
        v_initial.append(df[velocity][index - run_length])
        for step in range(run_length//10):
            x_steps[step].append(df[position][index - run_length + (1+step)*10])
            v_steps[step].append(df[velocity][index - run_length + (1+step)*10])

    data_worked[x0] = x_initial
    data_worked[v0] = v_initial
    for step in range(run_length//10):
        step_num = (step+1)*10
        data_worked[xf+f'{step_num:02d}'] = x_steps[step]
        data_worked[vf+f'{step_num:02d}'] = v_steps[step]


    return data_worked


def output_data(args, df):
    output_name = args.output
    output_path = './data/' + output_name +'.csv'
    df.to_csv(output_path)

def rework_data(args, df, run_length=50):
    """command line version; 
    reworks data in place and replaces the file with its improved version"""

    return separate_data(df, run_length, xmax=args.xmax, vmax=args.vmax)


def main(args):
    # generate the data
    input_path = args.input

    print(f"input path: {input_path}")
    print(f"output path: {args.output}")

    # all the names that fit this name
    # excepting a force.csv that we may have previously exported
    csv_files = [file for file in glob.glob(input_path+"*.csv") if "force.csv" not in file]


    print(f"to read: {csv_files}")

    # empty list for the data frames
    dfs = []
    for filename in csv_files:
        print(f"reading {filename} with {len(filename)} data points")
        df = pd.read_csv(filename)
        # rework data and save inplace
        df = rework_data(args, df)
        dfs.append(df)

    # combine all data frames into 1
    combined_df = pd.concat(dfs, ignore_index=True)

    # rework the data & save it
    output_data(args, combined_df)

    print(f"ammount of data so far: {len(combined_df)}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data cleaner for 2nd order differential equation made with integrate_data.py; works by creating strips of inital values and final values pairing each 10 time steps, until 50, creating far more data points for training')
    parser.add_argument('-input', type=str, help='Path of the input CSV file, use path to gen_data for generated data')
    parser.add_argument('-output', type=str, help='Name of the output CSV file to save on data/')
    parser.add_argument('-xmax', type=float, required=False, help='if provided, the data will be cut till the max value of position specified')
    parser.add_argument('-vmax', type=float, required=False, help='if provided, the data will be cut till the max value of velocity specified')
    args = parser.parse_args()

    if args.input:
        main(args)