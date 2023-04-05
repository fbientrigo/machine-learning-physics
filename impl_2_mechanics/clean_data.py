import pandas as pd
import argparse
import glob




def rework_data(args, df):
    output_name = args.output

    output_path = './data/' + output_name +'.csv'

    data_worked = pd.DataFrame()
    x_initial = []
    v_initial = []
    x_step10 = []
    v_step10 = []
    x_step20 = []
    v_step20 = []
    x_step30 = []
    v_step30 = []
    x_step40 = []
    v_step40 = []
    x_step50 = []
    v_step50 = []

    for index in range(50,len(df)-1):
        x_initial.append(df.position[index - 50])
        v_initial.append(df.velocity[index - 50])
        x_step10.append(df.position[index - 40])
        v_step10.append(df.velocity[index - 40])
        x_step20.append(df.position[index - 30])
        v_step20.append(df.velocity[index - 30])
        x_step30.append(df.position[index - 20])
        v_step30.append(df.velocity[index - 20])
        x_step40.append(df.position[index - 10])
        v_step40.append(df.velocity[index - 10])
        x_step50.append(df.position[index])
        v_step50.append(df.velocity[index])

    data_worked['x_initial'] = x_initial
    data_worked['v_initial'] = v_initial
    data_worked['x_step10'] = x_step10
    data_worked['v_step10'] = v_step10
    data_worked['x_step20'] = x_step20
    data_worked['v_step20'] = v_step20
    data_worked['x_step30'] = x_step30
    data_worked['v_step30'] = v_step30
    data_worked['x_step40'] = x_step40
    data_worked['v_step40'] = v_step40
    data_worked['x_step50'] = x_step50
    data_worked['v_step50'] = v_step50


    # separate when training
    #Ndata = len(data_worked)
    #datatraining = data_worked[:202]
    #datavalidation = data_worked[202:250]
    #datatesting = data_worked[250:-1]

    data_worked.to_csv(output_path)


def main(args):
    # generate the data
    input_path = args.input

    print(f"input path: {input_path}")
    print(f"output path: {args.output}")

    # all the names that fit this name
    csv_files = glob.glob(input_path+"*.csv")

    print(f"to read: {csv_files}")

    # empty list for the data frames
    dfs = []
    for filename in csv_files:
        print(f"reading {filename} with {len(filename)} data points")
        df = pd.read_csv(filename, index_col=0)
        dfs.append(df)

    # combine all data frames into 1
    combined_df = pd.concat(dfs, ignore_index=True)


    if args.vmax:
        print(f"len of dataframe before vmax: {len(combined_df)}")

        bool_mask = combined_df['velocity'] <= args.vmax
        combined_df = combined_df[ bool_mask ].reset_index(drop=True)
        # its important to reset index or will get errors when trying to acces non existen indexs

        print(f"len of dataframe after vmax: {len(combined_df)}")
        

    if args.xmax:
        print(f"len of dataframe before xmax: {len(combined_df)}")
        
        bool_mask = combined_df['position'] <= args.xmax
        combined_df = combined_df[ bool_mask ].reset_index(drop=True)

        print(f"len of dataframe after xmax: {len(combined_df)}")


    # rework the data & save it
    rework_data(args, combined_df)

    # imprimir el DataFrame combinado
    print(f"ammount of data so far: {len(combined_df)}")
    #combined_df.tail()






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data cleaner for 2nd order differential equation made with integrate_data.py; works by creating strips of inital values and final values pairing each 10 time steps, until 50, creating far more data points for training')
    parser.add_argument('-input', type=str, help='Path of the input CSV file, use path to gen_data for generated data')
    parser.add_argument('-output', type=str, help='Name of the output CSV file to save on data/')
    parser.add_argument('-xmax', type=float, required=False, help='if provided, the data will be cut till the max value of position specified')
    parser.add_argument('-vmax', type=float, required=False, help='if provided, the data will be cut till the max value of velocity specified')
    args = parser.parse_args()

    if args.input:
        main(args)