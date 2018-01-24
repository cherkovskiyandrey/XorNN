package com.cherkovskiy.neuronNetworks.api;

import java.io.OutputStream;

public interface NeuronNetwork {

    NeuronNetworkOutput process(NeuronNetworkInput input);

    void writeToXls(OutputStream to);
}
