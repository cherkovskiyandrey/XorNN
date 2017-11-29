package com.cherkovskiy.neuronNetworks.api;

public interface ActivationFunction {

    Double activate(Double netInput);

    Double derivative(Double netInput);
}
