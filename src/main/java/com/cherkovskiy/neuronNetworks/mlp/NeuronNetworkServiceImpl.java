package com.cherkovskiy.neuronNetworks.mlp;

import com.cherkovskiy.neuronNetworks.api.*;

import javax.annotation.Nonnull;

public class NeuronNetworkServiceImpl implements NeuronNetworkService {

    @Override
    @Nonnull
    public NeuronNetworkBuilder createBuilder() {
        return new NeuronNetworkBuilderImpl();
    }

    @Override
    @Nonnull
    public NeuronNetworkInputBuilder createInputBuilder() {
        return new NeuronNetworkInputBuilderImpl();
    }

    @Nonnull
    @Override
    public NeuronNetworkTrainSetBuilder createTrainSetBuilder() {
        return new NeuronNetworkTrainSetBuilderImpl();
    }

    @Nonnull
    @Override
    public NeuronNetworkOutputBuilder createOutputBuilder() {
        return new NeuronNetworkOutputBuilderImpl();
    }
}
