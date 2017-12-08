package com.cherkovskiy.neuronNetworks.api;

import java.util.Collection;

public interface NeuronNetworkInputBuilder {

    NeuronNetworkInputBuilder setInputValues(Collection<? extends Double> inputs);

    NeuronNetworkInput build();
}
