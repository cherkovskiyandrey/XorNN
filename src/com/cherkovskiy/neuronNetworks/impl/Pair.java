package com.cherkovskiy.neuronNetworks.impl;

class Pair<T, U> {
    private final T t;
    private final U u;

    Pair(T t, U u) {
        this.t = t;
        this.u = u;
    }

    static <T1, U1> Pair<T1, U1> of(T1 input, U1 output) {
        return new Pair<>(input, output);
    }

    public T getT() {
        return t;
    }

    public U getU() {
        return u;
    }
}
