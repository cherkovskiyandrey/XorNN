package com.cherkovskiy.neuronNetworks.api;

import java.io.InputStream;

public interface NeuronNetworkTrainSetBuilder {
    NeuronNetworkTrainSetBuilder setInputAndOutput(NeuronNetworkInput input, NeuronNetworkOutput output);

    NeuronNetworkTrainSets build();

    NeuronNetworkTrainSets build(InputStream fromXml);
}
