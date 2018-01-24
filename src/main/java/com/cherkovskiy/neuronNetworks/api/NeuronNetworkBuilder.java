package com.cherkovskiy.neuronNetworks.api;

import java.io.InputStream;

public interface NeuronNetworkBuilder {

    NeuronNetworkBuilder inputsNeurons(int amount);

    NeuronNetworkBuilder addHiddenLevel(int amount);

    NeuronNetworkBuilder outputNeurons(int amount);

    NeuronNetworkBuilder useBias(boolean b);

    NeuronNetworkBuilder setActivationFunction(ActivationFunction activationFunction);

    /**
     * Build {@link NeuronNetwork} from options.
     *
     * @return
     */
    NeuronNetwork build();

    /**
     * Build {@link NeuronNetwork} from xml file.
     * Options are ignored.
     *
     * @param fromXls
     * @return
     */
    NeuronNetwork build(InputStream fromXls);
}
