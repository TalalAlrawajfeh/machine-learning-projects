package neuron.components.activationfunctions;

import neuron.components.ActivationFunction;

/**
 * Created by u624 on 4/10/17.
 */
public class SigmoidActivationFunction implements ActivationFunction {
    private double coefficient;

    public SigmoidActivationFunction(double coefficient) {
        this.coefficient = coefficient;
    }

    @Override
    public double activate(double input, double previousActivationState) {
        return 1 / (1 + Math.exp(-coefficient * input));
    }
}
