package neuron.components.activationfunctions;

import neuron.components.ActivationFunction;

/**
 * Created by u624 on 4/10/17.
 */
public class BipolarSigmoidActivationFunction implements ActivationFunction {
    @Override
    public double activate(double input, double previousActivationState) {
        double expValue = Math.exp(-input);
        return (1 - expValue) / (1 + expValue);
    }
}
