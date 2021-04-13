package neuron.components.activationfunctions;

import neuron.components.ActivationFunction;

/**
 * Created by u624 on 4/10/17.
 */
public class LeCunTanhActivationFunction implements ActivationFunction {
    @Override
    public double activate(double input, double previousActivationState) {
        return 1.7159 * Math.tanh(2.0 / 3.0 * input);
    }
}
