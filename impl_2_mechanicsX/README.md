# Version of force dependence
- uses approach of generating random points and give a range of random pointss and applying the post processing

At the moment of generating data theres a decision on $x_f$, which is a determinated position, then when we use a combination of $(x_i, v_i)$ some will end up close to the vicinity of $x_f$ and some not, there is where we assign post processing to each one of this.

This approach in a way classifies, but the $x_f$ decission is not trivial and will have an impact.

In the paper is chooseen to be $x_f = 0$, which simplifys the function of post processing. An explanation could be the use or idea of roots.


# construction
- [ ] automatic data generation
- generate M number of points; control initial conditions with better automation process
    - what is better automation?



- every position equal to $x_f=0 \pm \epsilon$ will be taken as a final step and take the previous data points as part of its data












____

# Adapting a NN to a Differential Equation

Based on

‌Hashimoto, K., Sugishita, S., Tanaka, A., & Tomiya, A. (2018). Deep learning and holographic QCD. Physical Review D, 98(10). https://doi.org/10.1103/physrevd.98.106014


## folder structure:
- data: clean data for usage, use it for generating new data an cleaning in place as well
- 

# Usage
The steps are divided in 
- Data Generation: where goes `integrate_data.py`, uses RK5 integration scheme with `dt=1e-3`, the data is saved on `gen_data/` (generated data)
- Data Cleaning: `clean_data.py` produces time steps until 50 time steps (hardcoded), uses data from `gen_data` the data is then saved on `data/`
- 



# Data Generation
For use of the script example:
```
python .\integrate_data.py -name song_v150 -xi 0 -vi 150 -N 5000
```

All data generated will end up in `gen_data`; its recommended after a data set has stop being used, to create a folder for it inside and store the data there

The way is worked on the paper, is the unknown differential equation as a unknown function that's fixed, it separates different cases, where this function only depends on one of the variables

- only dependent on position (q for general coordinate)
$$
\frac{dp}{dt} = f(q)
$$

- only dependent on speed (here for generality, I write momentum)
$$
\frac{dp}{dt} = f(p)
$$

Its important to remember:
- $\Delta t$ is fixed with the depth of the Neural Network, so on the generation of data, only the initial conditions are controlled
- The unknown function is represented by an array as big as the range of inputs


# Data Cleaning
For use of the script example:
```
python .\clean_data.py -input .\gen_data\song_v0_dt1e-3.csv -output song_v0 -vmax 10
```

This data acces data in the 


## Loss functions and smoothness
- adding physics constrain to the function
- adding continuity condition to make the function more real like





