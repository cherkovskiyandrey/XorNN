package com.cherkovskiy.neuronNetworks.api;


import com.cherkovskiy.neuronNetworks.mlp.NeuronNetworkServiceImpl;

import javax.annotation.Nonnull;
import java.io.InputStream;

public interface NeuronNetworkService {

    static NeuronNetworkService defaultInstance() {
        return new NeuronNetworkServiceImpl();
    }

    @Nonnull
    NeuronNetworkBuilder createFeedforwardBuilder();

    @Nonnull
    BackPropagationLearnEngineBuilder createBackPropagationLearnEngineBuilder();

    @Nonnull
    ResilientBackPropagationLearnEngineBuilder createResilientBackPropagationLearnEngineBuilder();

    @Nonnull
    NeuronNetworkInputBuilder createInputBuilder();

    @Nonnull
    NeuronNetworkDataSetBuilder createTrainSetBuilder();

    @Nonnull
    NeuronNetworkOutputBuilder createOutputBuilder();

    @Nonnull
    ActivationFunction deserializeActFuncFrom(InputStream from);
}
