package com.cherkovskiy.neuronNetworks.core.activationFunctions;

import com.cherkovskiy.neuronNetworks.api.ActivationFunction;

public enum StandardActivationFunctions implements ActivationFunction {

    SIGMOID(new Sigmoid()),
    HYPERBOLIC_TANG_ANGUITA(new HyperbolicTangAnguita())
    ;

    private final ActivationFunction activationFunction;

    StandardActivationFunctions(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    @Override
    public double activate(double netInput) {
        return activationFunction.activate(netInput);
    }

    @Override
    public double derivative(double netInput) {
        return activationFunction.derivative(netInput);
    }

    @Override
    public double getRange() {
        return activationFunction.getRange();
    }
}
