package com.cherkovskiy.neuronNetworks.impl;

import com.cherkovskiy.neuronNetworks.api.NeuronNetworkOutput;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkOutputBuilder;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class NeuronNetworkOutputBuilderImpl implements NeuronNetworkOutputBuilder {
    private List<Double> outputVector = new LinkedList<>();

    @Override
    public NeuronNetworkOutputBuilder setOutputValues(List<Double> doubles) {
        outputVector.clear();
        outputVector.addAll(doubles);
        return this;
    }

    @Override
    public NeuronNetworkOutput build() {
        return new NeuronNetworkOutputImpl(Collections.unmodifiableList(outputVector), outputVector.size());
    }
}
