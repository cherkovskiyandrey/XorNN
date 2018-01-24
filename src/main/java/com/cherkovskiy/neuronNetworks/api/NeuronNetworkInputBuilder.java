package com.cherkovskiy.neuronNetworks.api;

import java.util.Collection;

public interface NeuronNetworkInputBuilder {

    NeuronNetworkInputBuilder setInputValues(Collection<Double> inputs);

    NeuronNetworkInput build();
}
