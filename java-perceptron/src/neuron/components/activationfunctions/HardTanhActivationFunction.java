package neuron.components.activationfunctions;

import neuron.components.ActivationFunction;

/**
 * Created by u624 on 4/10/17.
 */
public class HardTanhActivationFunction implements ActivationFunction {
    @Override
    public double activate(double input, double previousActivationState) {
        return Math.max(-1, Math.min(1, input));
    }
}
