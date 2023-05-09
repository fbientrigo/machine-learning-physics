from init_workspace import create_workspace 

from random_data_generator import random_data_generation
from integrate_data import compile_data_to_csv, integrate_diffeq
from clean_data import separate_data
import postprocessor
import dataset
import diffnet


def help():
    print("""
    DeepQL library has a workflow:
    1.- create your folder structure with `deepql.create_workspace()`
    2.- generate data from a differential equation of second order,
    this is done by passing the function $y'' = f(y,v)$ where it can have dependence
    on the position `y` or/and the speed `v = y'`
    choose: 
        a) generate random initial data `deepql.random_data_generation`
        b) generate one big simulation and cut it into pieces: `deepql.integrate_diffeq()`
            for cutting: `deepql.separate_data()`

    all data is stored in csv files, the library depends on pandas for this

    3.- in the case you're working with position, you can post process your data
    ...
    """)