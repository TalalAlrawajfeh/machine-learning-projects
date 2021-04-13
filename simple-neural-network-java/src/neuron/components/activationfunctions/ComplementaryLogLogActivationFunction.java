package neuron.components.activationfunctions;

import neuron.components.ActivationFunction;

/**
 * Created by u624 on 4/10/17.
 */
public class ComplementaryLogLogActivationFunction implements ActivationFunction {
    @Override
    public double activate(double input, double previousActivationState) {
        return 1 - Math.exp(-Math.exp(input));
    }
}
