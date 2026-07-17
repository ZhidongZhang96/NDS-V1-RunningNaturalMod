Now the SpeedTuning does not consider neurons' various response to different conditions. One neuron might be pretty silent to most conditions except for its preferred one. Unless the locomotion affect the response mainly by additive shift, in this way the average response over all conditions may not stand out for 'significant difference'.

Plan:
- Binary Modulation: for different stimuli and spont,
    1. compute the *modulated neurons* 
    2. compute MI, e.g., draw histogram, store for further printing
    3. train gain model
- SpeedTuning: stricted the neurons of interest on the 'modulated' neurons
    1. compute the mean response of these neurons over conditions, regardless of running speed. Then compute the response under `blank_sweep` as baseline, to find the neurons' active condition of each stimuli.
    > 基于`blank`的mean_-std来筛选conditions，但是有些neuron对某个stim下的所有condition都没通过筛选。
    2. compute tuning on the preferred conditions, test the variance and monotonicity.
    3. plot the tuning curve based on their tuning and monotonicity (non-tuned, pos, neg, non-mon) 3x4, with response during spont as baseline
    4. plot the number of modulated & tuned neurons
    5. plot a huge grid map (cell # by stimuli), containing (i) modulated mask (ii) p-value of tuning variance (ii) colored rho

Be careful to interpret the values of MI and gain obstained.



> 'modulated' = different response in `run` and `still`
> 'tuned' = different variance in tuning curve