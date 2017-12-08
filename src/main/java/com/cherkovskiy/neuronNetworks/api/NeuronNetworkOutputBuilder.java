package com.cherkovskiy.neuronNetworks.api;

import java.util.List;

public interface NeuronNetworkOutputBuilder {

    NeuronNetworkOutputBuilder setOutputValues(List<Double> doubles);

    NeuronNetworkOutput build();
}
