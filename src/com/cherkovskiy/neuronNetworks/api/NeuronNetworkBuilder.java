package com.cherkovskiy.neuronNetworks.api;

import java.io.InputStream;

public interface NeuronNetworkBuilder {

    NeuronNetworkBuilder setType(NeuronNetworkType type);

    NeuronNetworkBuilder inputsNeurons(int amount);

    NeuronNetworkBuilder addHiddenLevel(int amount);

    NeuronNetworkBuilder outputNeurons(int amount);

    NeuronNetworkBuilder useBias(boolean b);

    NeuronNetworkBuilder setActivationFunction(ActivationFunction activationFunction);

    NeuronNetworkBuilder setStopRelativeError(float percentInEachOut);

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
     * @param fromXml
     * @return
     */
    NeuronNetwork build(InputStream fromXml);
}
