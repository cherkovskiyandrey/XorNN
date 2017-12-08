package com.cherkovskiy.neuronNetworks.core.activationFunctions;

import com.cherkovskiy.neuronNetworks.api.ActivationFunction;

public class Sigmoid implements ActivationFunction {

    @Override
    public double activate(double netInput) {
        return 1. / (1. + Math.exp(-netInput));
    }

    @Override
    public double derivative(double netInput) {
        final double activated = activate(netInput);
        return activated * (1 - activated);
    }

    @Override
    public double getRange() {
        return 1.;
    }
}
