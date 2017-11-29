package com.cherkovskiy.neuronNetworks.api;

import java.util.Collection;

public interface NeuronNetworkInput {

    int size();

    Collection<? extends Double> getInput();
}
