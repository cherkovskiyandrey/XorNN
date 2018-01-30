package com.cherkovskiy.neuronNetworks.api;

import java.io.IOException;
import java.io.OutputStream;

public interface NeuronNetwork {

    NeuronNetworkOutput process(NeuronNetworkInput input);

    void writeTo(OutputStream to) throws IOException;
}
