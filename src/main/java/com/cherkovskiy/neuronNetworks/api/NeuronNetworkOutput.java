package com.cherkovskiy.neuronNetworks.api;

import java.util.List;

public interface NeuronNetworkOutput {
    List<Double> getOutput();

    List<Double> getOutputAllNeurons();

    List<Double> getInputs();

    List<Double> getInputsAllNeurons();
}
