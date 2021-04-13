package neuron.components.activationfunctions;

import neuron.components.ActivationFunction;

/**
 * Created by u624 on 4/10/17.
 */
public class MNISTActivationFunction implements ActivationFunction {
    @Override
    public double activate(double input, double previousActivationState) {
        return Math.max(0, input) + Math.cos(input);
    }
}
