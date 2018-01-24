package com.cherkovskiy.neuronNetworks.mlp;

import com.cherkovskiy.neuronNetworks.api.ActivationFunction;
import com.cherkovskiy.neuronNetworks.api.NeuronNetwork;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkBuilder;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkType;

import java.io.InputStream;
import java.util.LinkedList;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class NeuronNetworkBuilderImpl implements NeuronNetworkBuilder {
    private NeuronNetworkType type;
    private ActivationFunction activationFunction;
    private int input;
    private LinkedList<Integer> hiddenLevels = new LinkedList<>();
    private int output;
    private boolean useStatModule;
    private double range;
    private double step;
    private double weightDecay = Double.NaN;
    private double learningRate =  0.01;  //according to 92 page: 0.01 ≤ η ≤ 0.9;

    public NeuronNetworkBuilder setType(NeuronNetworkType type) {
        this.type = type;
        return this;
    }

    public NeuronNetworkBuilder inputsNeurons(int amount) {
        if (amount < 1) {
            throw new IllegalArgumentException(String.format("Amount of inputs neurons must be at least one: %d", amount));
        }
        this.input = amount;
        return this;
    }

    public NeuronNetworkBuilder addHiddenLevel(int amount) {
        if (amount < 1) {
            throw new IllegalArgumentException("Level can`t be empty");
        }
        this.hiddenLevels.add(amount);
        return this;
    }

    public NeuronNetworkBuilder outputNeurons(int amount) {
        if (amount < 1) {
            throw new IllegalArgumentException(String.format("Amount of outputs neurons must be at least one: %d", amount));
        }
        this.output = amount;
        return this;
    }

    public NeuronNetworkBuilder useBias(boolean b) {
        return this;
    }

    @Override
    public NeuronNetworkBuilder setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        return this;
    }

    @Override
    public NeuronNetworkBuilder useStatModule(boolean b, double range, double step) {
        this.useStatModule = b;
        this.range = range;
        this.step = step;
        return this;
    }

    @Override
    public NeuronNetworkBuilder learningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    @Override
    public NeuronNetworkBuilder weightDecay(double b) {
        this.weightDecay = b;
        return this;
    }

    public NeuronNetwork build() {
        int allNeurons = input + hiddenLevels.stream().mapToInt(Integer::intValue).sum() + output;
        final double[][] topology = NeuronNetworkCoreHelper.nanArray(allNeurons, allNeurons);
        int levelBegin = 0;
        int levelEnd = input;

        for (int levelAmount : Stream.concat(hiddenLevels.stream(), Stream.of(output)).collect(Collectors.toList())) {
            for (int levelNeuron = levelEnd; levelNeuron < levelEnd + levelAmount; levelNeuron++) {
                for (int prevLevelNeuron = levelBegin; prevLevelNeuron < levelEnd; prevLevelNeuron++) {
                    topology[levelNeuron][prevLevelNeuron] = getNextRandom();
                }
            }
            levelBegin = levelEnd;
            levelEnd = levelEnd + levelAmount;
        }

        return new NeuronNetworkImpl(
                new NeuronNetworkDomain(input, topology, output, activationFunction),
                learningRate,
                weightDecay,
                useStatModule ? new StatisticsProvider(range, step) : null
        );
    }

    @Override
    public NeuronNetwork build(InputStream fromXml) {
        throw new UnsupportedOperationException("Method has not been supported yet.");
    }


    private static Double getNextRandom() {
        while (true) {
            double val = ThreadLocalRandom.current().nextDouble(-0.5, 0.5);
            if (Math.abs(val) > 1.E-5) {
                return val;
            }
        }
    }
}
