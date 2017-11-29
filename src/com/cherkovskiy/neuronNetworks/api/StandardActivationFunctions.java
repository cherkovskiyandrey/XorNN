package com.cherkovskiy.neuronNetworks.api;

public enum StandardActivationFunctions implements ActivationFunction {

    SIGMOID {
        @Override
        public Double activate(Double netInput) {
            return 1. / (1. + Math.exp(-netInput));
        }

        @Override
        public Double derivative(Double netInput) {
            final double activated = activate(netInput);
            return activated * (1 - activated);
        }
    };

    @Override
    public abstract Double activate(Double netInput);

    @Override
    public abstract Double derivative(Double netInput);
}
