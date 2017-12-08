package com.cherkovskiy.neuronNetworks.api;

public interface ActivationFunction {

    double activate(double netInput);

    double derivative(double netInput);

    double getRange();
}
