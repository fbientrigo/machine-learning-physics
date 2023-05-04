
# Deep Learning for Modeling Differential Equations
*introduction*


# Folder Structure
- `documentation`: explain different functions inside each project
- `impl_1_original`: contains the original repository from which the paper "Deep Learning and Holographic QCD" from Hashimoto, K. et al
- `impl_2_mechanics`: based on the concepts and methods of "AdS/Deep Learning made easy: simple examples" from Song, M. et al. Provides the same approach but to simpler differential equations presented in classical mechanics.
- `pytorch_introduction`: Provides jupyter notebooks with more and more elaborated implementations of networks, the uses of algorithms like K_Folding for improving training and cross validation and rough implementations of the final project, providing a guide for others to understand.

# Papers Cited and in Implementation:

‌Hashimoto, K., Sugishita, S., Tanaka, A., & Tomiya, A. (2018). Deep learning and holographic QCD. Physical Review D, 98(10). https://doi.org/10.1103/physrevd.98.106014


Song, M., Oh, M. S. H., Ahn, Y., & Kima, K.-Y. (2021). AdS/Deep-Learning made easy: simple examples *. Chinese Physics C, 45(7), 073111. https://doi.org/10.1088/1674-1137/abfc36


# Installation and usage
All is found inside the environment files, which contains the exact versions of every library used, activate it using:

```
$ conda env create -f environment.yml
```

then activvate the new environment and verify the list of dependencies:

```
$ conda activate deepl
$ conda env list
```

This environment contains the installation of:
- ipykernel
- matplotlib
- pandas
- numpy
- pytorch
- sklearn-scikit

If you wish to update the environment file after installing a new library:

```
$ conda activate deepl
$ conda env export > environment.yml
```

for the library to be able to work, while is in construction to be in a future installable by `pip` globally, you have to go to `/DeepQl/` and run:

```
pip istall -e .
```

That will install the module in editable mode, so all changes inside the library are updated.