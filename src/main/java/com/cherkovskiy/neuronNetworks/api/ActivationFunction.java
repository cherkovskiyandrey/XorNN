package com.cherkovskiy.neuronNetworks.api;

import java.io.OutputStream;

public interface ActivationFunction {

    double activate(double netInput);

    double derivative(double netInput);

    double getRange();

    void serializeTo(OutputStream to);
}
