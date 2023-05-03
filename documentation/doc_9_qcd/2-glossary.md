## Meaning of variables
- `H` $Mq$ quark mass
- `M` $\braket{\bar q q}$ (chiral condensate)
- `Hs` array of `float H` 
- `Ms` array of `float M` 

- `f` is direct polynomial fit (deg=3) of the data $(H,M) = (H, f(H))$

- `Phi` `Pi` is part of the discretized functions

Metric function:
$ h(\eta) = \partial_\eta \log \sqrt{f(\eta) \; g(\eta)^{d-1}} 


The relation is $\pi = \partial_n \phi \; {}_{((6))}$

The classical equation of motion used:
$$
\partial_\eta \pi = - h(\eta)  + m^2 \phi + \frac{\delta V[\phi]}{\delta \phi} \; {}_{((7))}
$$

$\phi$ is the solution the system is searching for



With the derivative of we can construct the discretization: 
$$
{}_{(6)} \rightarrow \; \phi(\eta + \Delta \eta) = \phi(\eta) + \Delta \eta \pi(\eta)
$$

$$
{}_{(7)} \rightarrow \pi(\eta + \Delta \eta) = \pi(\eta) - \Delta \eta ( h(\eta)  - m^2 \phi - \frac{\delta V[\phi]}{\delta \phi})
$$


## Meaning of conditions

Condition of enough closseness, 
$| f(H) - M | \leq \epsilon \; \; \forall \epsilon ~ N(\mu, \sigma^2)$
