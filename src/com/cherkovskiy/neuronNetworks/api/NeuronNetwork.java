package com.cherkovskiy.neuronNetworks.api;

import java.io.OutputStream;

public interface NeuronNetwork {

    NeuronNetworkOutput process(NeuronNetworkInput input);

    void writeAsXml(OutputStream to);

    void learn(NeuronNetworkTrainSets neuronNetworkTrainSet);

    boolean turnDebugOn(boolean debugMode);
}
