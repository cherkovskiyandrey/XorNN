package com.cherkovskiy.neuronNetworks.impl;

import com.cherkovskiy.neuronNetworks.api.*;

import java.io.OutputStream;
import java.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NeuronNetworkImpl implements NeuronNetwork {
    private static final Logger COMMON_LOGGER = LoggerFactory.getLogger("CommonLogger");

    private final ActivationFunction activationFunction;
    private final int inputAmount;
    private final Double[][] topology; //TODO: simplify - get rid of boxing/unboxing
    private final int outputAmount;
    private final double learnRate = 0.01;
    private DebugLevels debugLevel = DebugLevels.ERROR;
    private float stopRelativeError = 0.1f; // not more 10% in each output

    NeuronNetworkImpl(int inputAmount, Double[][] topology, int outputAmount, ActivationFunction activationFunction, float stopRelativeError) {
        this.inputAmount = inputAmount;
        this.topology = topology;
        this.outputAmount = outputAmount;
        this.activationFunction = activationFunction;
        this.stopRelativeError = stopRelativeError;

        COMMON_LOGGER.debug(this.toString());
    }

    @Override
    public NeuronNetworkOutput process(NeuronNetworkInput input) {
        if (input.size() != inputAmount) {
            throw new IllegalArgumentException(String.format("Incompatible input: size of input array is %d != %d NeuronNetworkImpl amount of input neurons", input.size(), inputAmount));
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
    public void learnBackProp(NeuronNetworkTrainSets neuronNetworkTrainSet) {
        neuronNetworkTrainSet.forEach(trainSet -> {
            if (trainSet.getInput().size() != inputAmount) {
                throw new IllegalArgumentException(String.format(
                        "Incompatible input: size of input array is %d != %d NeuronNetworkImpl amount of input neurons",
                        trainSet.getInput().size(), inputAmount));
            }

            if (trainSet.getOutput().getOutput().size() != outputAmount) {
                throw new IllegalArgumentException(String.format(
                        "Incompatible output: size of output array is %d != %d NeuronNetworkImpl amount of output neurons",
                        trainSet.getOutput().getOutput().size(), outputAmount));
            }
        });

        double currentMinEuclidError = Double.MAX_VALUE;
        double fullRelativeError = 100d;
        long epochNumber = 0;

        //while (fullRelativeError > stopRelativeError) { //TODO: см questions.txt "Изменить подход остановки обучения"
        double fullEuclidError = Double.MAX_VALUE;
        //while (fullEuclidError > 0.01d) {
        while(true) {
            fullEuclidError = 0d;
            for (NeuronNetworkTrainSets.TrainSet trainSet : neuronNetworkTrainSet) {
                final Pair<Double, Double> errors = doLearnEachPattern(trainSet, epochNumber); 
                fullEuclidError += errors.getFirst();
                fullRelativeError += errors.getSecond();
            }
            fullEuclidError /= 2;
            fullRelativeError /= neuronNetworkTrainSet.getSize();

            if (fullEuclidError < currentMinEuclidError) {
                currentMinEuclidError = fullEuclidError;
            }
            epochNumber++;

            if (debugLevel.doOut(epochNumber)) {
                COMMON_LOGGER.debug(this.toString());
            }

            if (debugLevel.isLessThanOrEqual(DebugLevels.INFO)) {
                String message = "| Epoch: " + epochNumber +
                        "; full relative error: " + fullRelativeError +
                        "; current error: " + fullEuclidError +
                        "; min error: " + currentMinEuclidError +
                        "; TREND: " + (fullEuclidError > currentMinEuclidError ? "!!!ascent!!!" : "decent");
                COMMON_LOGGER.debug(message);
            }
        }

//        if (debugLevel.isLessThanOrEqual(DebugLevels.OFF)) {
//            for (NeuronNetworkTrainSets.TrainSet trainSet : neuronNetworkTrainSet) {
//                final NeuronNetworkOutput neuronNetworkOutput = process(trainSet.getInput());
//                logAllNets(neuronNetworkOutput);
//            }
//            COMMON_LOGGER.debug(this.toString());
//        }
    }

    @Override
    public void setDebugMode(DebugLevels debugLevel) {
        this.debugLevel = debugLevel;
    }

    @Override
    public void learnResilientProp(NeuronNetworkTrainSets neuronNetworkTrainSet) {
        throw new UnsupportedOperationException("Is not implemented yet.");
    }

    //without recursion
    private Pair<Double, Double> doLearnEachPattern(NeuronNetworkTrainSets.TrainSet trainSet, long epochNumber) {
        final NeuronNetworkOutput neuronNetworkOutput = process(trainSet.getInput());
        final List<Double> currentAllInputs = neuronNetworkOutput.getInputsAllNeurons();
        final List<Double> currentAllOutputs = neuronNetworkOutput.getOutputAllNeurons();

        final List<Double> currentOutput = neuronNetworkOutput.getOutput();
        final List<Double> teachOutput = trainSet.getOutput().getOutput();

        final double[] dArray = new double[topology.length];
        final ArrayDeque<Double> deltaRates = new ArrayDeque<>();

        if (debugLevel.isLessThanOrEqual(DebugLevels.INFO)) {
            logCurrentOutput(trainSet.getInput(), neuronNetworkOutput);
            if(debugLevel.isLessThanOrEqual(DebugLevels.DEBUG) || debugLevel.doOut(epochNumber)) {
                logAllNets(neuronNetworkOutput);
            }
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

        if (debugLevel.isLessThanOrEqual(DebugLevels.DEBUG)) {
            COMMON_LOGGER.debug(this.toString());
        }

        return Pair.of(calcEuclideError(teachOutput, currentOutput), calcRelativeError(teachOutput, currentOutput));
    }

    private double calcEuclideError(List<Double> teachOutput, List<Double> currentOutput) {
        double result = 0d;
        for (int i = 0; i < teachOutput.size(); ++ i) {
            result += Math.pow((teachOutput.get(i) - currentOutput.get(i)), 2);
        }
        return result;
    }

    private Double calcRelativeError(List<Double> teachOutput, List<Double> currentOutput) {
        double result = 0d;
        for (int i = 0; i < teachOutput.size(); ++ i) {
            result += Math.abs(teachOutput.get(i) - currentOutput.get(i))/activationFunction.getRange();
        }
        return result/teachOutput.size();
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
        COMMON_LOGGER.debug(inputStr.toString());
    }

    private void logAllNets(NeuronNetworkOutput neuronNetworkOutput) {
        final StringBuilder outStr = new StringBuilder("Nets: ");
        for (double outputVal : neuronNetworkOutput.getInputsAllNeurons()) {
            outStr.append(String.format("|%10.5f|", outputVal));
        }

        COMMON_LOGGER.debug(outStr.toString());
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
