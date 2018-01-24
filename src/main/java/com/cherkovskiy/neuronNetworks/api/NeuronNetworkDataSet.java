package com.cherkovskiy.neuronNetworks.api;

import java.util.Iterator;

public interface NeuronNetworkDataSet {

    interface TrainSet {
        NeuronNetworkInput getInput();

        NeuronNetworkOutput getOutput();
    }

    Iterator<TrainSet> iteratorOverTrainSet();

    Iterator<TrainSet> iteratorOverCheckSet();
}
