package com.cherkovskiy.neuronNetworks.api;

import java.io.OutputStream;

public interface NeuronNetwork {

    NeuronNetworkOutput process(NeuronNetworkInput input);

    void writeAsXml(OutputStream to);

    void learnBackProp(NeuronNetworkTrainSets neuronNetworkTrainSet);

    void setDebugMode(DebugLevels debugLevel);

    void learnResilientProp(NeuronNetworkTrainSets neuronNetworkTrainSet);

    void logErrorFunction(int everyCycles);
}
