package com.cherkovskiy.neuronNetworks.impl;

import com.cherkovskiy.neuronNetworks.api.*;

import java.io.OutputStream;
import java.util.*;


public class NeuronNetworkImpl implements NeuronNetwork {
    private final ActivationFunction activationFunction;
    private final int inputAmount;
    private final Double[][] topology; //TODO: simplify - get rid of boxing/unboxing
    private final int outputAmount;
    private final double learnRate = 0.5; //TODO: redesign to dynamic approach
    private boolean debugMode = false;

    NeuronNetworkImpl(int inputAmount, Double[][] topology, int outputAmount, ActivationFunction activationFunction) {
        this.inputAmount = inputAmount;
        this.topology = topology;
        this.outputAmount = outputAmount;
        this.activationFunction = activationFunction;
    }

    @Override
    public NeuronNetworkOutput process(NeuronNetworkInput input) {
        if (input.size() != inputAmount) {
            throw new IllegalArgumentException(String.format("Incompatible input: size of input array is %d != NeuronNetworkImpl amount of input neurons %d", input.size(), inputAmount));
        }
        final List<Double> inputVector = new LinkedList<>();
        inputVector.addAll(input.getInput());

        final List<Double> outputVector = new LinkedList<>();
        outputVector.addAll(input.getInput());

        for (int i = input.getInput().size(); i < topology.length; ++i) {
            double sum = 0;
            for (int j = 0; j < topology[i].length; ++j) {
                final Double rate = topology[i][j];
                if (rate != null) {
                    sum += outputVector.get(j) * rate;
                }
            }
            inputVector.add(sum);
            outputVector.add(activationFunction.activate(sum));
        }

        return new NeuronNetworkOutputImpl(inputVector, inputAmount, outputVector, outputAmount);
    }

    @Override
    public void writeAsXml(OutputStream to) {
        throw new UnsupportedOperationException("Method has not been supported yet.");
    }

    @Override
    public void learn(NeuronNetworkTrainSets neuronNetworkTrainSet) {
        //TODO: check size before process!

        //TODO: when we stop? - обновляем минимальное значение функции ошибки (-> 0) и topology ей соответсвующую + выводить смену тренда функции ошибки: уменьшается или увеличивается
        //TODO: calculate summary error
        double currentMinError = -10d;
        while (true) {
            double fullError = 0d;

            for (NeuronNetworkTrainSets.TrainSet trainSet : neuronNetworkTrainSet) {
                fullError += doLearnEachPattern(trainSet);
            }
            fullError /= 2;

            if (currentMinError < -1.d) {
                currentMinError = fullError;
            }
            if (fullError < currentMinError) {
                currentMinError = fullError;
            }

            if (debugMode) {
                System.out.println("----------------------");
                System.out.println("| Current errors: " + fullError + "; min errors: " + currentMinError + "; TREND: " +
                        (fullError > currentMinError ? "!!!ascent!!!" : "decent"));
                System.out.println("----------------------");
                System.out.println("==============================================");
            }
        }
    }

    @Override
    public boolean turnDebugOn(boolean debugMode) {
        boolean oldDebugMode = this.debugMode;
        this.debugMode = debugMode;

        return oldDebugMode;
    }

    //without recursion
    private double doLearnEachPattern(NeuronNetworkTrainSets.TrainSet trainSet) {
        final NeuronNetworkOutput neuronNetworkOutput = process(trainSet.getInput());
        final List<Double> currentAllInputs = neuronNetworkOutput.getInputsAllNeurons();
        final List<Double> currentAllOutputs = neuronNetworkOutput.getOutputAllNeurons();

        final List<Double> currentOutput = neuronNetworkOutput.getOutput();
        final List<Double> teachOutput = trainSet.getOutput().getOutput();

        final double[] dArray = new double[topology.length];
        final ArrayDeque<Double> deltaRates = new ArrayDeque<>();

        if (debugMode) {
            logCurrentOutput(trainSet.getInput(), neuronNetworkOutput);
        }

        for (int currentNeuronIndex = topology.length - 1, outputCounter = outputAmount;
             currentNeuronIndex >= inputAmount;
             currentNeuronIndex--, outputCounter--) {

            final boolean isOutputNeuron = outputCounter > 0;
            final Double[] currentNeuronLinks = topology[currentNeuronIndex];

            if (isOutputNeuron) {
                double d = activationFunction.derivative(currentAllInputs.get(currentNeuronIndex)) * (teachOutput.get(outputCounter - 1) - currentOutput.get(outputCounter - 1));
                dArray[currentNeuronIndex] = d;
            } else {
                final List<Integer> linkedUpStream = getUpStream(currentNeuronIndex);
                double sumUpStream = 0.;
                for (int curUpStream : linkedUpStream) {
                    double upStreamRate = topology[curUpStream][currentNeuronIndex];
                    sumUpStream += upStreamRate * dArray[curUpStream];
                }

                double d = activationFunction.derivative(currentAllInputs.get(currentNeuronIndex)) * sumUpStream;
                dArray[currentNeuronIndex] = d;
            }

            for (int linkedNeuronIndex = currentNeuronLinks.length - 1; linkedNeuronIndex != 0; linkedNeuronIndex--) {
                final Double rate = currentNeuronLinks[linkedNeuronIndex];
                if (rate != null) {
                    double deltaRate = learnRate * currentAllOutputs.get(linkedNeuronIndex) * dArray[currentNeuronIndex];
                    deltaRates.add(deltaRate);
                }
            }
        }

        applyDeltaRates(deltaRates);

        return calcError(teachOutput, currentOutput);
    }

    private double calcError(List<Double> teachOutput, List<Double> currentOutput) {
        double result = 0d;
        for (int i = 0; i < teachOutput.size(); ++ i) {
            result += Math.pow((teachOutput.get(i) - currentOutput.get(i)), 2);
        }
        return result;
    }

    private void logCurrentOutput(NeuronNetworkInput input, NeuronNetworkOutput output) {
        final StringBuilder inputStr = new StringBuilder("Input: ");
        for (double inputVal : input.getInput()) {
            inputStr.append(String.format("|%14.10f|", inputVal));
        }
        inputStr.append(" => ");
        for (double outputVal : output.getOutput()) {
            inputStr.append(String.format("|%14.10f|", outputVal));
        }

        System.out.println(inputStr.toString());
    }

    private List<Integer> getUpStream(int currentNeuronIndex) {
        final List<Integer> result = new ArrayList<>();
        for (int i = currentNeuronIndex + 1; i < topology.length; ++i) {
            Double rate = topology[i][currentNeuronIndex];
            if (rate != null) {
                result.add(i);
            }
        }
        return result;
    }

    private void applyDeltaRates(Queue<Double> deltaRates) {
        for (int currentNeuronIndex = topology.length - 1, outputCounter = outputAmount;
             currentNeuronIndex >= inputAmount;
             currentNeuronIndex--, outputCounter--) {

            final Double[] currentNeuronLinks = topology[currentNeuronIndex];

            for (int linkedNeuronIndex = currentNeuronLinks.length - 1; linkedNeuronIndex != 0; linkedNeuronIndex--) {
                final Double rate = currentNeuronLinks[linkedNeuronIndex];
                if (rate != null) {
                    currentNeuronLinks[linkedNeuronIndex] += deltaRates.poll();
                }
            }
        }
    }

    @Override
    public String toString() {
        final StringBuilder topologyAsString = new StringBuilder();
        for (int i = 0; i < topology.length; ++i) {
            for (int j = 0; j < topology[i].length; ++j) {
                final Double curVal = topology[i][j];
                if (curVal != null) {
                    topologyAsString.append(String.format("|%7.3f|", curVal));
                } else {
                    topologyAsString.append("|-------|");
                }
            }
            topologyAsString.append(System.lineSeparator());
        }

        return "NeuronNetworkImpl{" +
                "inputAmount=" + inputAmount +
                ", outputAmount=" + outputAmount + System.lineSeparator() +
                ", topology=" + System.lineSeparator() +
                topologyAsString.toString() + System.lineSeparator() +
                '}';
    }
}
