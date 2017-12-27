package com.cherkovskiy.neuronNetworks.impl;

import com.cherkovskiy.neuronNetworks.api.ActivationFunction;

class NeuronNetworkDomain implements Cloneable {
    private final ActivationFunction activationFunction;
    private final int inputAmount;
    private double[][] topology;
    private final int outputAmount;

    NeuronNetworkDomain(int inputAmount, double[][] topology, int outputAmount, ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        this.inputAmount = inputAmount;
        this.topology = topology;
        this.outputAmount = outputAmount;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public int getInputAmount() {
        return inputAmount;
    }

    public double[][] getTopology() {
        return topology;
    }

    public int getOutputAmount() {
        return outputAmount;
    }

    public NeuronNetworkDomain clone() throws CloneNotSupportedException {
        final NeuronNetworkDomain result = (NeuronNetworkDomain) super.clone();
        result.topology = NeuronNetworkCoreHelper.nanArray(topology.length, topology[0].length);
        for (int i = 0; i < topology.length; i++) {
            System.arraycopy(topology[i], 0, result.topology[i], 0, topology[i].length);
        }
        return result;
    }

    public NeuronNetworkDomain newClone() {
        try {
            return clone();
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        final StringBuilder topologyAsString = new StringBuilder();
        for (int i = 0; i < topology.length; ++i) {
            for (int j = 0; j < topology[i].length; ++j) {
                final Double curVal = topology[i][j];
                if (!curVal.isNaN()) {
                    topologyAsString.append(String.format("|%7.3f|", curVal));
                } else {
                    topologyAsString.append("|-------|");
                }
            }
            topologyAsString.append(System.lineSeparator());
        }

        return "NeuronNetworkDomain{" +
                "inputAmount=" + inputAmount +
                ", outputAmount=" + outputAmount + System.lineSeparator() +
                ", topology=" + System.lineSeparator() +
                topologyAsString.toString() + System.lineSeparator() +
                '}';
    }
}
