package com.cherkovskiy.neuronNetworks.api;

public interface BackPropagationLearnEngine {

    BackPropagationLearnResult learn(NeuronNetwork neuronNetwork, NeuronNetworkDataSet neuronNetworkTrainSet);

    /**
     * Estimate NN is it appropriate to current dataset.
     *
     * @param neuronNetworkFromFile
     * @param neuronNetworkTrainSet
     * @return
     */
    BackPropagationLearnResult estimate(NeuronNetwork neuronNetworkFromFile, NeuronNetworkDataSet neuronNetworkTrainSet);
}
