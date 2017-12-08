package com.cherkovskiy.neuronNetworks.impl;

import com.cherkovskiy.neuronNetworks.api.NeuronNetworkInput;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkOutput;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkTrainSets;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkTrainSetBuilder;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class NeuronNetworkTrainSetBuilderImpl implements NeuronNetworkTrainSetBuilder {
    private List<Pair<NeuronNetworkInput, NeuronNetworkOutput>> trainInput = new ArrayList<>();

    @Override
    public NeuronNetworkTrainSetBuilder setInputAndOutput(NeuronNetworkInput input, NeuronNetworkOutput output) {
        trainInput.add(Pair.of(input, output));
        return this;
    }

    @Override
    public NeuronNetworkTrainSets build() {
        return new NeuronNetworkTrainSetInMemoryImpl(Collections.unmodifiableList(trainInput));
    }

    @Override
    public NeuronNetworkTrainSets build(InputStream fromXml) {
        throw new UnsupportedOperationException("Method has not been supported yet.");
    }
}
