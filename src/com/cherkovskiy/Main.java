package com.cherkovskiy;

import com.cherkovskiy.neuronNetworks.api.*;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.stream.Collectors;

public class Main {

    public static void main(String[] args) throws IOException {
        final NeuronNetworkService neuronNetworkService = NeuronNetworkService.defaultInstance();

        final NeuronNetwork xor = neuronNetworkService.createBuilder()
                .setType(NeuronNetworkType.FEEDFORFARD)
                .setActivationFunction(StandardActivationFunctions.SIGMOID)
                .inputsNeurons(2)
                .addHiddenLevel(3)
                .addHiddenLevel(3)
                .outputNeurons(1)
                .useBias(false)
                .build();

        System.out.println(xor);

        final NeuronNetworkTrainSetBuilder neuronNetworkTrainSetBuilder = neuronNetworkService.createTrainSetBuilder();
        final NeuronNetworkInputBuilder inputBuilder = neuronNetworkService.createInputBuilder();
        final NeuronNetworkOutputBuilder outputBuilder = neuronNetworkService.createOutputBuilder();

        final NeuronNetworkTrainSets neuronNetworkTrainSet = neuronNetworkTrainSetBuilder
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 0d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(1d, 0d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 1d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(1d, 1d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d)).build())

                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 0d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d)).build())
                //.setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 1d)).build(), outputBuilder.setOutputValues(Arrays.asList(0.1d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 2d)).build(), outputBuilder.setOutputValues(Arrays.asList(0.2d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 3d)).build(), outputBuilder.setOutputValues(Arrays.asList(0.3d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 4d)).build(), outputBuilder.setOutputValues(Arrays.asList(0.4d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 5d)).build(), outputBuilder.setOutputValues(Arrays.asList(0.5d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 6d)).build(), outputBuilder.setOutputValues(Arrays.asList(0.6d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 7d)).build(), outputBuilder.setOutputValues(Arrays.asList(0.7d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 10d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d)).build())

                .build();

        //todo
        //final NeuronNetworkTrainSets neuronNetworkTrainSet = neuronNetworkTrainSetBuilder.build(new FileInputStream("some_trained_file"));

        xor.turnDebugOn(true);

        //TODO: реализация алгоритма оптимального по скорости обучения: resilient backpropagation (short Rprop)
        xor.learn(neuronNetworkTrainSet);

        for (Double[] i : Arrays.asList(
                new Double[]{0d, 0d}
                , new Double[]{0d, 1d}
                , new Double[]{1d, 0d}
                , new Double[]{1d, 1d}
        )) {
            final NeuronNetworkOutput output = xor.process(neuronNetworkService.createInputBuilder().setInputValues(Arrays.stream(i).collect(Collectors.toList())).build());

            System.out.println("Output: " + output.getOutput());
            System.out.println("Full output: " + output);
        }

        try (final OutputStream outputStream = new BufferedOutputStream(new FileOutputStream("xor-nn.xml"))) {
            xor.writeAsXml(outputStream);
        }
    }

}

