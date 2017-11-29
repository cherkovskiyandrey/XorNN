package com.cherkovskiy.neuronNetworks.impl;

import com.cherkovskiy.neuronNetworks.api.NeuronNetworkInput;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkOutput;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkTrainSets;

import javax.annotation.Nonnull;
import java.util.Iterator;
import java.util.List;

class NeuronNetworkTrainSetInMemoryImpl implements NeuronNetworkTrainSets {
    private final List<Pair<NeuronNetworkInput, NeuronNetworkOutput>> trainInput; //TODO: get rid of Pair and use TrainSet implementation

    NeuronNetworkTrainSetInMemoryImpl(List<Pair<NeuronNetworkInput, NeuronNetworkOutput>> trainInput) {
        this.trainInput = trainInput;
    }

    @Nonnull
    @Override
    public Iterator<TrainSet> iterator() {
        final Iterator<Pair<NeuronNetworkInput, NeuronNetworkOutput>> originalItr = trainInput.iterator();

        return new Iterator<TrainSet>() {
            @Override
            public boolean hasNext() {
                return originalItr.hasNext();
            }

            @Override
            public TrainSet next() {
                final Pair<NeuronNetworkInput, NeuronNetworkOutput> origVal = originalItr.next();

                return new TrainSet() {
                    @Override
                    public NeuronNetworkInput getInput() {
                        return origVal.getT();
                    }

                    @Override
                    public NeuronNetworkOutput getOutput() {
                        return origVal.getU();
                    }
                };
            }
        };
    }
}
