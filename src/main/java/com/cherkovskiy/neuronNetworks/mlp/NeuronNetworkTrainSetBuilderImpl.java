package com.cherkovskiy.neuronNetworks.mlp;

import com.cherkovskiy.neuronNetworks.api.NeuronNetworkInput;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkOutput;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkDataSet;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkDataSetBuilder;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class NeuronNetworkTrainSetBuilderImpl implements NeuronNetworkDataSetBuilder {
    private List<Pair<NeuronNetworkInput, NeuronNetworkOutput>> trainInput = new ArrayList<>();
    private List<Pair<NeuronNetworkInput, NeuronNetworkOutput>> verifyingInput = new ArrayList<>();

    @Override
    public NeuronNetworkDataSetBuilder setTrainInputAndOutput(NeuronNetworkInput input, NeuronNetworkOutput output) {
        trainInput.add(Pair.of(input, output));
        return this;
    }

    @Override
    public NeuronNetworkDataSetBuilder setVerifyingInputAndOutput(NeuronNetworkInput input, NeuronNetworkOutput output) {
        verifyingInput.add(Pair.of(input, output));
        return this;
    }

    @Override
    public NeuronNetworkDataSet build() {
        return new NeuronNetworkTrainSetInMemoryImpl(Collections.unmodifiableList(trainInput));
    }

    @Override
    public NeuronNetworkDataSet build(InputStream inputStream) {
        throw new UnsupportedOperationException("Method has not been supported yet.");
    }
}
