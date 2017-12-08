package com.cherkovskiy.neuronNetworks.api;


import com.cherkovskiy.neuronNetworks.impl.NeuronNetworkServiceImpl;

import javax.annotation.Nonnull;

public interface NeuronNetworkService {

    static NeuronNetworkService defaultInstance() {
        return new NeuronNetworkServiceImpl();
    }

    @Nonnull
    NeuronNetworkBuilder createBuilder();

    @Nonnull
    NeuronNetworkInputBuilder createInputBuilder();

    @Nonnull
    NeuronNetworkTrainSetBuilder createTrainSetBuilder();

    @Nonnull
    NeuronNetworkOutputBuilder createOutputBuilder();
}
