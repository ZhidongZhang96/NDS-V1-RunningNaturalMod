# Speed Tuning

Tuning analysis of runing-speed of V1 ΔF/F

## 1. Methods

As in [`Plan.md`](Plan.md) and [`TASKS.md`](TASKS.md#person-a--analysis-1-speedtuning-medium--shared-reducers), this analysis
1. Bined the resposne ΔF/F by running speeds, compute the average and MSE across neurons to get the tuning curve.
2. Ran **Levene's t test** to obtain the significantly *running-tuned* neurons with $p<0.05$.
3. Within the *tuned* neurons, computed **Spearman's $\rho$** to recognize the *monotonicity* of their tuning, dividing them into 3 groups based on it.
    - Neurons don't show clear monotonicity, i.e. $p>0.05$: `non-monotonic`
    - Neurons have $p>0.05$ and $\rho>0$: `positive`
    - Neurons have $p>0.05$ and $\rho<0$: `negative`


## 2. Results
### 2.1 Number of tuned neurons

Across the three stimuli, the fewest neurons is tuned to running under `drifting_gratings`, while under `spontaneous` the number is the lowest. 

Therefore, the speed tuning of the 47 V1 neurons is stimuli-dependent.

![](figures/tuned_neurons_numbers.png)


### 2.2 Tuning curves grouped by tuning monotonicity

Here we show the tuning curves of neurons grouped by monotonicity across 3 stimuli, along with the tuning under `spontaneous` as baseline. The real ine linking dots denotes the average responses, the shadow intervals denotes the MSE. Average Spearman's $\rho$ of the selected neurons is also shown.


![](figures/tuning_by_monotonicity.png)


### 2.3 Each neuron's tuning
Here we show each neuron's tuning monotonicity across 4 different stimuli conditions, along with the `responsive` and `speed-tuned` mask based on the metadata from Allen's : [`neurons_metadata.csv`](../data/neurons_metadata.csv). 

> - `responsive` means that the neuron is actively responsive under certain stimuli (`p_dg`, `p_sg`, `p_ns`).
> - `speed-tuned` means that the neuron is signiticantly tuned by running speed under certain stimuli (`p_run_mod_dg`, `p_run_mod_sg`, `p_run_mod_ns`).

![](figures/tuning_of_neurons.png)