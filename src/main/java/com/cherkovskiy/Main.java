package com.cherkovskiy;

import com.cherkovskiy.neuronNetworks.api.*;
import com.cherkovskiy.neuronNetworks.core.activationFunctions.StandardActivationFunctions;

import java.io.IOException;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) throws IOException {
        final NeuronNetworkService neuronNetworkService = NeuronNetworkService.defaultInstance();

        final NeuronNetwork xor = neuronNetworkService.createBuilder()
                .setType(NeuronNetworkType.FEEDFORFARD)
                //.setActivationFunction(StandardActivationFunctions.SIGMOID) //регулярно застреваем - понять почему
                .setActivationFunction(StandardActivationFunctions.HYPERBOLIC_TANG_ANGUITA) // работает действительно очень круто и быстро приходит к цели
                .inputsNeurons(2)
                .addHiddenLevel(3)
                .addHiddenLevel(3)
                .outputNeurons(1)
                .useBias(false)
                .setStopRelativeError(0.05f) // error not more 1% for each out
                .build();

        System.out.println(xor);

        final NeuronNetworkTrainSetBuilder neuronNetworkTrainSetBuilder = neuronNetworkService.createTrainSetBuilder();
        final NeuronNetworkInputBuilder inputBuilder = neuronNetworkService.createInputBuilder();
        final NeuronNetworkOutputBuilder outputBuilder = neuronNetworkService.createOutputBuilder();

        final NeuronNetworkTrainSets neuronNetworkTrainSet = neuronNetworkTrainSetBuilder

                //Отлично обучается
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 0d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(1d, 0d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 1d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(1d, 1d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d)).build())

                //Так себе результаты - скатываемся иногда в локальный миниум и не можем из него вылезти
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 0d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 2d)).build(), outputBuilder.setOutputValues(Arrays.asList(0.2d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 3d)).build(), outputBuilder.setOutputValues(Arrays.asList(0.3d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 4d)).build(), outputBuilder.setOutputValues(Arrays.asList(0.4d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 5d)).build(), outputBuilder.setOutputValues(Arrays.asList(0.5d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 6d)).build(), outputBuilder.setOutputValues(Arrays.asList(0.6d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 7d)).build(), outputBuilder.setOutputValues(Arrays.asList(0.7d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 10d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d)).build())

                .build();

        //todo
        //final NeuronNetworkTrainSets neuronNetworkTrainSet = neuronNetworkTrainSetBuilder.build(new FileInputStream("some_trained_file"));

        xor.setDebugMode(DebugLevels.DEBUG_EVERY_100_000);

        xor.learnBackProp(neuronNetworkTrainSet);

        //TODO: implement first of all!
        //xor.learnResilientProp(neuronNetworkTrainSet);

        //TODO: check
//        for (Double[] i : Arrays.asList(
//                new Double[]{0d, 0d}
//                , new Double[]{0d, 1d}
//                , new Double[]{1d, 0d}
//                , new Double[]{1d, 1d}
//        )) {
//            final NeuronNetworkOutput output = xor.process(neuronNetworkService.createInputBuilder().setInputValues(Arrays.stream(i).collect(Collectors.toList())).build());
//
//            System.out.println("Output: " + output.getOutput());
//            System.out.println("Full output: " + output);
//        }
//
//        try (final OutputStream outputStream = new BufferedOutputStream(new FileOutputStream("xor-nn.xml"))) {
//            xor.writeAsXml(outputStream);
//        }
    }

}

