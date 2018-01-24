package com.cherkovskiy.neuronNetworks.api;

public interface BackPropagationLearnEngineBuilder extends LearnEngineBuilderBase {

    BackPropagationLearnEngineBuilder learningRate(double rate);

    /**
     * Punishment of large weights.
     * <br>
     * See: 5.6.4 Weight decay.
     *
     * @param decayFactor
     * @return
     */
    BackPropagationLearnEngineBuilder weightDecay(double decayFactor);


    BackPropagationLearnEngine build();
}
