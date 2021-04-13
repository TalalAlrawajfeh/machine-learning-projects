package neuron.components.activationfunctions;

import neuron.components.ActivationFunction;

/**
 * Created by u624 on 4/10/17.
 */
public class BipolarActivationFunction implements ActivationFunction {
    @Override
    public double activate(double input, double previousActivationState) {
        if (input <= 0) {
            return -1;
        }
        return 1;
    }
}
