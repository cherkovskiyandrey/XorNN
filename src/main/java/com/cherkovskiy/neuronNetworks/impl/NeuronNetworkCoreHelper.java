package com.cherkovskiy.neuronNetworks.impl;

import com.cherkovskiy.neuronNetworks.api.NeuronNetworkInput;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkOutput;
import com.cherkovskiy.neuronNetworks.api.NeuronNetworkTrainSets;

import java.util.*;

class NeuronNetworkCoreHelper {

    static NeuronNetworkOutput process(NeuronNetworkInput input, NeuronNetworkDomain nn) {
        if (input.size() != nn.getInputAmount()) {
            throw new IllegalArgumentException(String.format("Incompatible input: size of input array is %d != %d NeuronNetworkImpl amount of input neurons", input.size(), nn.getInputAmount()));
        }
        final List<Double> inputVector = new LinkedList<>();
        inputVector.addAll(input.getInput());

        final List<Double> outputVector = new LinkedList<>();
        outputVector.addAll(input.getInput());

        for (int i = input.getInput().size(); i < nn.getTopology().length; ++i) {
            double sum = 0;
            for (int j = 0; j < nn.getTopology()[i].length; ++j) {
                final Double rate = nn.getTopology()[i][j];
                if (!rate.isNaN()) {
                    sum += outputVector.get(j) * rate;
                }
            }
            inputVector.add(sum);
            outputVector.add(nn.getActivationFunction().activate(sum));
        }

        return new NeuronNetworkOutputImpl(inputVector, nn.getInputAmount(), outputVector, nn.getOutputAmount());
    }

    public static void checkCompatible(NeuronNetworkTrainSets neuronNetworkTrainSet, NeuronNetworkDomain nn) {
        neuronNetworkTrainSet.forEach(trainSet -> {
            if (trainSet.getInput().size() != nn.getInputAmount()) {
                throw new IllegalArgumentException(String.format(
                        "Incompatible input: size of input array is %d != %d NeuronNetworkImpl amount of input neurons",
                        trainSet.getInput().size(), nn.getInputAmount()));
            }

            if (trainSet.getOutput().getOutput().size() != nn.getOutputAmount()) {
                throw new IllegalArgumentException(String.format(
                        "Incompatible output: size of output array is %d != %d NeuronNetworkImpl amount of output neurons",
                        trainSet.getOutput().getOutput().size(), nn.getOutputAmount()));
            }
        });
    }

    public static void checkCompatible(NeuronNetworkDomain nn1, NeuronNetworkDomain nn2) {
        //todo
    }


    /**
     * @param teachOutput
     * @param neuronNetworkOutput
     * @param learnRate
     * @param nn
     * @return delta rates of weight for certain one pattern form end of topology
     */
    public static ArrayDeque<Double> quickBpLearnPattern(List<Double> teachOutput,
                                                         NeuronNetworkOutput neuronNetworkOutput,
                                                         double learnRate,
                                                         NeuronNetworkDomain nn) {

        final List<Double> currentAllInputs = neuronNetworkOutput.getInputsAllNeurons();
        final List<Double> currentAllOutputs = neuronNetworkOutput.getOutputAllNeurons();
        final List<Double> currentOutput = neuronNetworkOutput.getOutput();

        final double[] dArray = new double[nn.getTopology().length];
        final ArrayDeque<Double> deltaRates = new ArrayDeque<>();

        for (int currentNeuronIndex = nn.getTopology().length - 1, outputCounter = nn.getOutputAmount();
             currentNeuronIndex >= nn.getInputAmount();
             currentNeuronIndex--, outputCounter--) {

            final boolean isOutputNeuron = outputCounter > 0;
            final double[] currentNeuronLinks = nn.getTopology()[currentNeuronIndex];

            if (isOutputNeuron) {
                double d = nn.getActivationFunction().derivative(currentAllInputs.get(currentNeuronIndex)) * (teachOutput.get(outputCounter - 1) - currentOutput.get(outputCounter - 1));
                dArray[currentNeuronIndex] = d;
            } else {
                final List<Integer> linkedUpStream = getUpStream(currentNeuronIndex, nn);
                double sumUpStream = 0.;
                for (int curUpStream : linkedUpStream) {
                    double upStreamRate = nn.getTopology()[curUpStream][currentNeuronIndex];
                    sumUpStream += upStreamRate * dArray[curUpStream];
                }

                double d = nn.getActivationFunction().derivative(currentAllInputs.get(currentNeuronIndex)) * sumUpStream;
                dArray[currentNeuronIndex] = d;
            }

            for (int linkedNeuronIndex = currentNeuronLinks.length - 1; linkedNeuronIndex != 0; linkedNeuronIndex--) {
                final Double rate = currentNeuronLinks[linkedNeuronIndex];
                if (!rate.isNaN()) {
                    double deltaRate = learnRate * currentAllOutputs.get(linkedNeuronIndex) * dArray[currentNeuronIndex];
                    deltaRates.add(deltaRate);
                }
            }
        }

        return deltaRates;
    }


    /**
     * Change nn accord. to deltaRates
     *
     * @param deltaRates
     * @param nn
     */
    public static void applyDeltaRates(Queue<Double> deltaRates, NeuronNetworkDomain nn) {
        for (int currentNeuronIndex = nn.getTopology().length - 1, outputCounter = nn.getOutputAmount();
             currentNeuronIndex >= nn.getInputAmount();
             currentNeuronIndex--, outputCounter--) {

            final double[] currentNeuronLinks = nn.getTopology()[currentNeuronIndex];

            for (int linkedNeuronIndex = currentNeuronLinks.length - 1; linkedNeuronIndex != 0; linkedNeuronIndex--) {
                final Double rate = currentNeuronLinks[linkedNeuronIndex];
                if (!rate.isNaN()) {
                    currentNeuronLinks[linkedNeuronIndex] += deltaRates.poll();
                }
            }
        }
    }

    /**
     * Calc error for one case.
     *
     * @param teachOutput
     * @param currentOutput
     * @return
     */
    public static double calcEuclideError(List<Double> teachOutput, List<Double> currentOutput) {
        double result = 0d;
        for (int i = 0; i < teachOutput.size(); ++i) {
            result += Math.pow((teachOutput.get(i) - currentOutput.get(i)), 2);
        }
        return result;
    }

    /**
     * Calc relative error for one case.
     *
     * @param teachOutput
     * @param currentOutput
     * @return
     */
    public static Double calcRelativeError(List<Double> teachOutput, List<Double> currentOutput, NeuronNetworkDomain nn) {
        double result = 0d;
        for (int i = 0; i < teachOutput.size(); ++i) {
            result += Math.abs(teachOutput.get(i) - currentOutput.get(i)) / nn.getActivationFunction().getRange();
        }
        return result / teachOutput.size();
    }

    private static List<Integer> getUpStream(int currentNeuronIndex, NeuronNetworkDomain nn) {
        final List<Integer> result = new ArrayList<>();
        for (int i = currentNeuronIndex + 1; i < nn.getTopology().length; ++i) {
            Double rate = nn.getTopology()[i][currentNeuronIndex];
            if (!rate.isNaN()) {
                result.add(i);
            }
        }
        return result;
    }

    public static double[][] nanArray(int a, int b) {
        double[][] result = new double[a][b];
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                result[i][j] = Double.NaN;
            }
        }
        return result;
    }
}
