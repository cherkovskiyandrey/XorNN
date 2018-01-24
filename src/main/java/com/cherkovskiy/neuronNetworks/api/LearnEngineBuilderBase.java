package com.cherkovskiy.neuronNetworks.api;

public interface LearnEngineBuilderBase {

    /**
     * If error function does not change greater than maxErrorFluctuation in last epochAmount epoch, learn will stop.
     * It is a successful end of cycle.
     *
     * @param epochAmount
     * @param maxErrorFluctuation
     * @return
     */
    BackPropagationLearnEngineBuilder setStopCondition(int epochAmount, double maxErrorFluctuation);

    /**
     * Usually it is protection from long cycles.
     *
     * @param epochAmount
     * @return
     */
    BackPropagationLearnEngineBuilder setMaxEpochPerCycle(int epochAmount);

    /**
     * Criteria to successful stop learning current cycle.
     *
     * @param errorVal
     * @return
     */
    BackPropagationLearnEngineBuilder setSuccessErrorValue(double errorVal);

    /**
     * Use in extension topology.
     *
     * @param neurons
     * @return
     */
    BackPropagationLearnEngineBuilder setMinNeuronsPerLevel(int neurons);

    /**
     * Use in extension topology.
     *
     * @param neurons
     * @return
     */
    BackPropagationLearnEngineBuilder setMaxNeuronsPerLevel(int neurons);

    /**
     * Use in extension topology.
     *
     * @param levels
     * @return
     */
    BackPropagationLearnEngineBuilder setMinLevels(int levels);

    /**
     * Use in extension topology.
     *
     * @param levels
     * @return
     */
    BackPropagationLearnEngineBuilder setMaxLevels(int levels);

    BackPropagationLearnEngineBuilder setDebugMode(DebugLevels debugLevel);

    BackPropagationLearnEngineBuilder logErrorFunction(int everyCycles);

    /**
     * Just to investigation goals.
     * To process-intensive! Don`t use in release.
     *
     * @param on
     * @param range
     * @param step
     * @return
     */
    BackPropagationLearnEngineBuilder useStatModule(boolean on, double range, double step);
}
