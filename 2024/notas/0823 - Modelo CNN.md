Se trabaja con un feed forward CNN que pasa por un autoencoder
la idea luego es expandir este autoencoder en una Fully connected NN, y entonces predecir los eigenvalues $E_\lambda$

$$
V(x) \Rightarrow \text{CNN} \Rightarrow f(\vec\psi)=E_\lambda
$$
de manera que esta red pueda ser invertible, y comenzar a utilizar los eigenvalues para así generar distintos potenciales.



