package com.cherkovskiy.neuronNetworks.mlp;

import com.cherkovskiy.neuronNetworks.api.ActivationFunction;
import com.cherkovskiy.neuronNetworks.api.NeuronNetwork;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkInput;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkOutput;

import java.io.*;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.LinkedList;
import java.util.List;

class FeedforwardNeuronNetworkImpl implements NeuronNetwork, Cloneable {
    public final static String TYPE = "FEED_FORWARD";
    public static String UUID = "27b4fe1c-9390-47e3-8eb2-155c8fc138f1";

    private final ActivationFunction activationFunction;
    private final int inputAmount;
    private double[][] topology;
    private final int outputAmount;

    FeedforwardNeuronNetworkImpl(int inputAmount, double[][] topology, int outputAmount, ActivationFunction activationFunction) {
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

    public FeedforwardNeuronNetworkImpl clone() throws CloneNotSupportedException {
        final FeedforwardNeuronNetworkImpl result = (FeedforwardNeuronNetworkImpl) super.clone();
        result.topology = NeuronNetworkCoreHelper.nanArray(topology.length, topology[0].length);
        for (int i = 0; i < topology.length; i++) {
            System.arraycopy(topology[i], 0, result.topology[i], 0, topology[i].length);
        }
        return result;
    }

    public FeedforwardNeuronNetworkImpl newClone() {
        try {
            return clone();
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public NeuronNetworkOutput process(NeuronNetworkInput input) {
        if (input.size() != inputAmount) {
            throw new IllegalArgumentException(String.format("Incompatible input: size of input array is %d != %d BackPropagationLearnEngineImpl amount of input neurons",
                    input.size(), inputAmount));
        }
        final List<Double> inputVector = new LinkedList<>();
        inputVector.addAll(input.getInput());

        final List<Double> outputVector = new LinkedList<>();
        outputVector.addAll(input.getInput());

        for (int i = input.getInput().size(); i < topology.length; ++i) {
            double sum = 0;
            for (int j = 0; j < topology[i].length; ++j) {
                final double rate = topology[i][j];
                if (!Double.isNaN(rate)) {
                    sum += outputVector.get(j) * rate;
                }
            }
            inputVector.add(sum);
            outputVector.add(activationFunction.activate(sum));
        }

        return new NeuronNetworkOutputImpl(inputVector, inputAmount, outputVector, outputAmount);
    }

    @Override
    public void writeTo(OutputStream to) throws IOException {
        final BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(to, StandardCharsets.UTF_8));
        try {
            writer.write(TYPE);
            writer.write("\n");
            writer.write(UUID);
            writer.write("\n");

            try (ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream()) {
                try (OutputStream outputStream = Base64.getEncoder().wrap(byteArrayOutputStream)) {
                    activationFunction.serializeTo(outputStream);
                }
                writer.write(byteArrayOutputStream.toString(StandardCharsets.UTF_8.name()));
                writer.write("\n");
            }

            writer.write(Integer.toString(inputAmount));
            writer.write("\n");
            writer.write(Integer.toString(outputAmount));
            writer.write("\n");
            writer.write(Integer.toString(topology.length));
            writer.write("\n");
            for (double[] level : topology) {
                for (int j = 0; j < level.length; j++) {
                    writer.write(j + ":" + level[j] + ";");
                }
                writer.write("\n");
            }
        } finally {
            writer.flush();
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

        return "FeedforwardNeuronNetworkImpl{" +
                "inputAmount=" + inputAmount +
                ", outputAmount=" + outputAmount + System.lineSeparator() +
                ", topology=" + System.lineSeparator() +
                topologyAsString.toString() + System.lineSeparator() +
                '}';
    }
}
