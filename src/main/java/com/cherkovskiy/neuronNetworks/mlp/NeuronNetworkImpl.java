package com.cherkovskiy.neuronNetworks.mlp;

import com.cherkovskiy.neuronNetworks.api.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.OutputStream;
import java.util.ArrayDeque;
import java.util.List;
import java.util.Optional;

public class NeuronNetworkImpl implements NeuronNetwork {
    private static final Logger COMMON_LOGGER = LoggerFactory.getLogger("CommonLogger");

    private final NeuronNetworkDomain nn;
    private final double learnRate = 0.01; //according to 92 page: 0.01 ≤ η ≤ 0.9
    private DebugLevels debugLevel = DebugLevels.ERROR;
    private float stopRelativeError = 0.1f; // not more 10% in each output
    private final Optional<StatisticsProvider> statisticsProvider;

    NeuronNetworkImpl(NeuronNetworkDomain neuronNetworkDomain, float stopRelativeError, StatisticsProvider statisticsProvider) {
        this.nn = neuronNetworkDomain;
        this.stopRelativeError = stopRelativeError;
        this.statisticsProvider = Optional.ofNullable(statisticsProvider);

        COMMON_LOGGER.debug(this.toString());
    }

    @Override
    public NeuronNetworkOutput process(NeuronNetworkInput input) {
        return NeuronNetworkCoreHelper.process(input, nn);
    }

    @Override
    public void writeAsXml(OutputStream to) {
        throw new UnsupportedOperationException("Method has not been supported yet.");
    }

    @Override
    public void learnBackProp(NeuronNetworkTrainSets neuronNetworkTrainSet) {
        NeuronNetworkCoreHelper.checkCompatible(neuronNetworkTrainSet, nn);

        double currentMinEuclidError = Double.MAX_VALUE;
        double fullRelativeError = 100d;
        long epochNumber = 0;

        //while (fullRelativeError > stopRelativeError) { //TODO: см questions.txt "Изменить подход остановки обучения"
        double fullEuclidError = Double.MAX_VALUE;
        //while (fullEuclidError > 0.01d) {
        while (true) {
            fullEuclidError = 0d;

            statisticsProvider.ifPresent(p -> p.topologySnapshort(nn));
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

            if (debugLevel.doOut(epochNumber)) {
                COMMON_LOGGER.debug(this.toString());
            }

            final long finalEpochNumber = epochNumber;
            statisticsProvider.ifPresent(p -> p.buildRatesDependencies(nn, neuronNetworkTrainSet, finalEpochNumber));


            if (debugLevel.isLessThanOrEqual(DebugLevels.INFO)) {
                String message = "| Epoch: " + epochNumber +
                        "; full relative error: " + fullRelativeError +
                        "; current error: " + fullEuclidError +
                        "; min error: " + currentMinEuclidError +
                        "; TREND: " + (fullEuclidError > currentMinEuclidError ? "!!!ascent!!!" : "decent");
                COMMON_LOGGER.debug(message);
            }

            epochNumber++;
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

        final List<Double> currentOutput = neuronNetworkOutput.getOutput();
        final List<Double> teachOutput = trainSet.getOutput().getOutput();

        if (debugLevel.isLessThanOrEqual(DebugLevels.INFO)) {
            logCurrentOutput(trainSet.getInput(), neuronNetworkOutput);
            if (debugLevel.isLessThanOrEqual(DebugLevels.DEBUG) || debugLevel.doOut(epochNumber)) {
                logAllNets(neuronNetworkOutput);
            }
        }

        final ArrayDeque<Double> deltaRates = NeuronNetworkCoreHelper.quickBpLearnPattern(teachOutput, neuronNetworkOutput, learnRate, nn);
        NeuronNetworkCoreHelper.applyDeltaRates(deltaRates, nn);

        if (debugLevel.isLessThanOrEqual(DebugLevels.TRACE)) {
            COMMON_LOGGER.debug(this.toString());
        }

        return Pair.of(NeuronNetworkCoreHelper.calcEuclideError(teachOutput, currentOutput), NeuronNetworkCoreHelper.calcRelativeError(teachOutput, currentOutput, nn));
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

    @Override
    public String toString() {
        //TODO
        return "NeuronNetworkImpl{" +
                "NeuronNetworkDomain=" + nn.toString() +
                '}';
    }
}
