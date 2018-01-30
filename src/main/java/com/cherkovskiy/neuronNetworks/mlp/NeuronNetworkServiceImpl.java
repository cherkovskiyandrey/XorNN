package com.cherkovskiy.neuronNetworks.mlp;

import com.cherkovskiy.neuronNetworks.api.*;

import javax.annotation.Nonnull;
import java.io.InputStream;

public class NeuronNetworkServiceImpl implements NeuronNetworkService {

    @Nonnull
    @Override
    public NeuronNetworkBuilder createFeedforwardBuilder() {
        return new FeedforwardNeuronNetworkBuilderImpl(this);
    }

    @Nonnull
    @Override
    public BackPropagationLearnEngineBuilder createBackPropagationLearnEngineBuilder() {
        //todo
    }

    @Nonnull
    @Override
    public ResilientBackPropagationLearnEngineBuilder createResilientBackPropagationLearnEngineBuilder() {
        //todo
    }

    @Override
    @Nonnull
    public NeuronNetworkInputBuilder createInputBuilder() {
        return new NeuronNetworkInputBuilderImpl();
    }

    @Nonnull
    @Override
    public NeuronNetworkDataSetBuilder createTrainSetBuilder() {
        return new NeuronNetworkTrainSetBuilderImpl();
    }

    @Nonnull
    @Override
    public NeuronNetworkOutputBuilder createOutputBuilder() {
        return new NeuronNetworkOutputBuilderImpl();
    }

    @Nonnull
    @Override
    public ActivationFunction deserializeActFuncFrom(InputStream from) {
        //todo
    }
}
