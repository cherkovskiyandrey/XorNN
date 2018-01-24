package com.cherkovskiy;

import com.cherkovskiy.neuronNetworks.api.*;
import com.cherkovskiy.neuronNetworks.core.activationFunctions.StandardActivationFunctions;

import java.io.IOException;
import java.util.Arrays;
import java.util.Optional;

public class Main {

    public static void main(String[] args) throws IOException {
        final NeuronNetworkService neuronNetworkService = NeuronNetworkService.defaultInstance();

        //TODO: разделить создание сети от утановки парамтров обучения, логирования и других вещей
        final NeuronNetwork xor = neuronNetworkService.createBuilder()
                .setType(NeuronNetworkType.FEEDFORFARD)
                //.setActivationFunction(StandardActivationFunctions.SIGMOID) //регулярно застреваем - понять почему
                .setActivationFunction(StandardActivationFunctions.HYPERBOLIC_TANG_ANGUITA) // работает действительно очень круто и быстро приходит к цели
                .inputsNeurons(3)
                .addHiddenLevel(5)
                .outputNeurons(1)
                .useBias(false)

                .learningRate(0.01)  //according to 92 page: 0.01 ≤ η ≤ 0.9
                .weightDecay(0.02) //TODO

                .useStatModule(false, 1., 0.1)

                //TODO:
                //.setStopCondition(1000, 10E-5) // типа если за 1000 циклов минимумальное значение не будет обновлено больше чем на 10E-5 - заканчиваем
                .build();

        final NeuronNetworkTrainSetBuilder neuronNetworkTrainSetBuilder = neuronNetworkService.createTrainSetBuilder();
        final NeuronNetworkInputBuilder inputBuilder = neuronNetworkService.createInputBuilder();
        final NeuronNetworkOutputBuilder outputBuilder = neuronNetworkService.createOutputBuilder();

        final NeuronNetworkTrainSets neuronNetworkTrainSet = neuronNetworkTrainSetBuilder

                //Отлично обучается
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 0d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(1d, 0d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 1d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(1d, 1d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d)).build())

                //Отлично обучается на на большой сети
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 0d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d, 0d, 0d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 1d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d, 0d, 1d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 2d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d, 1d, 0d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 3d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d, 1d, 1d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 4d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d, 0d, 0d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 5d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d, 0d, 1d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 6d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d, 1d, 0d)).build())
//                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 7d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d, 1d, 1d)).build())


                // На такой архитектуре - регулярно сталкиваемся с локальным миниумом - решенеи - https://www.researchgate.net/publication/220237893_Avoiding_the_Local_Minima_Problem_in_Backpropagation_Algorithm_with_Modified_Error_Function
                //                .inputsNeurons(3)
                //                .addHiddenLevel(4)
                //                .outputNeurons(1)
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 0d, 0d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 0d, 1d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 1d, 0d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(0d, 1d, 1d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(1d, 0d, 0d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(1d, 0d, 1d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(1d, 1d, 0d)).build(), outputBuilder.setOutputValues(Arrays.asList(0d)).build())
                .setInputAndOutput(inputBuilder.setInputValues(Arrays.asList(1d, 1d, 1d)).build(), outputBuilder.setOutputValues(Arrays.asList(1d)).build())


                .build();

        //todo
        //final NeuronNetworkTrainSets neuronNetworkTrainSet = neuronNetworkTrainSetBuilder.build(new FileInputStream("some_trained_file"));

        xor.setDebugMode(DebugLevels.DEBUG_EVERY_100_000);
        xor.logErrorFunction(500);

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

