
The research question:  It has been shown that Layer 2/3 and 4 neurons in mice V1 are positively modulated by local motion during the visual responsiveness task, specifically in drifting grating. Through the Allen dataset, we want to verify whether those modulations are present in the same grating task and how different they are under naturalistic stimuli.

> ⚠️ **Cohort caveat (area).** This plan (and the bundled 47-cell `visual_coding_data.npz`) says "V1", but that cohort is Allen container `511510753` = **VISpm** (a higher visual area), not V1/VISp — layer/line are as intended (Cux2-CreERT2, L2/3–4). Genuine V1 results use a pooled 3-container VISp cohort (n=363). See [`TEAM_NOTE.md`](TEAM_NOTE.md). Also note the implementation diverges from this plan in places (e.g. the tuning significance test is a one-way ANOVA across speed bins, not the Levene/shuffle test written below).

Specifically, we want to focus on the modualation of running speed on the response amplitude, and compare the results of `drifting_gratings`, `static_gratings` and `natural_scenes`. It also contains the comparison to trials without stimuli, i.e., `spontaneous`

1. Pre-processing
	1. {a} Extract the trials for different stimuli (a trial = duration for a stimulus : `dg`=60=2s, `sg`=`ns`=7 $\approx$ 0.23s ) ==$30 Hz$==
2. Analysis of different stimuli (activity $R$ & running speed $V$)
	- Binned speed tuning, with and without stimuli (spontaneous) [@christensen2022ReducedNeuralActivity]
		1. {a} bin the running speed (e.g., into 20 bins) and compute the *average* response $\mathbb E_{\text{bin}} [R]$ with std -> tuning curve[^2]
		2. Levene’s t-test of variance to shuffled curve -> significantly tuned neurons
		3. Correlation
			- Spearman's rho -> monotonic relation
			- ~~Pearson's rho -> linear relation~~
	-  Binary conditions [@christensen2022ReducedNeuralActivity]
		1. {a} Divide the trials into `running` and `still` 
			- For different stimuli, select a good **offset** to get the actual 'response': `dg`=10, `sg`=`ns`=5 [^1]
			- `running` (avg > 3 , no one < 0.5), `still` (avg < 0.5, no one > 3), ignore others
		2. Compute each neuron's *average* response $\mathbb E_{\text{conditions}} [R]$ for each condition.
		3. Compute Modulation Index $$MI = \frac{R_{\text {run}} - R_{\text {still}}}{R_{\text {run}} + R_{\text {still}}}\,\in (-1,1)$$
		4. Plot scatter in a `running`-`still` manner, and fit a simple gain model:$$R_{\text {run}} = a\cdot R_{\text {still}} + b$$ , then $a$ denotes multiplication effect, $b$ denotes addition effect.

	- Fit predictive model [@liska2024RunningModulatesPrimate]
		1. {a} Fit different models $$\begin{aligned} R&= f(S) + \beta_0 &\text{(Null)} \\  &= f(S) + \beta_0  + \beta_{\text{add}} V &\text{(Add-only)} \\  &= f(S) + \beta_0  + \beta_{\text{mult}}(V\times S) &\text{(Mult-only)} \\   &= f(S) + \beta_0  + \beta_{\text{add}} V + \beta_{\text{mult}}(V\times S) &\text{(Full-model)} \end{aligned}$$
			- $f(S)$: the average response to the specific stimulus $S$, i.e., tuning
			- $\beta_0$: baseline, its drifting can be modeled by $\beta_0(t) = \sum_j b_j \phi_j(t)$, with 'tent' basis function $\phi_j(t)$
			- $\beta_{\text{add}} V$: additive term
			- $\beta_{\text{mult}}(V\times S)$: multiplicative term, can be modeled by $\mathrm{ReLU}[1 + \beta_{\text{mult}} V]$
			  
			  Overall, the full-model (for a single neuron $i$) is  $$r_i(t) = f_i(s) + \underbrace{\sum_j b_j \phi_j(t)}_{\text{drifting baseline}} + \underbrace{[\beta_{\text{add}} V(t)]_i}_{\text{additive term}} + \underbrace{\mathrm{ReLU}[1 + \beta_{\text{mult}} V(t)]_i}_{\text{multiplicative term}}$$
		2. compute the Coefficient of Determination $R^2$, and then $\Delta R^2_{\text{add}}$, $\Delta R^2_{\text{mult}}$ and $\Delta R^2_{\text{full}}$
		-  for natural scenes, $S$ can be defined as the average response, then the goal is to predict the residual of the take-out.