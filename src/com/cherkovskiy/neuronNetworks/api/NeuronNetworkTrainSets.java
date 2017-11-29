package com.cherkovskiy.neuronNetworks.api;

public interface NeuronNetworkTrainSets extends Iterable<NeuronNetworkTrainSets.TrainSet> {

    interface TrainSet {
        NeuronNetworkInput getInput();

        NeuronNetworkOutput getOutput();
    }
}
