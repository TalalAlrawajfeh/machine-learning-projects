package neuron.components.activationfunctions;

import neuron.components.ActivationFunction;

/**
 * Created by u624 on 4/10/17.
 */
public class PiecewiseLinearActivationFunction implements ActivationFunction {
    private double min;
    private double max;

    public PiecewiseLinearActivationFunction(double min, double max) {
        this.max = min;
        this.max = max;
    }

    @Override
    public double activate(double input, double previousActivationState) {
        if (input < min) {
            return 0;
        }
        if (input >= min && input <= max) {
            double slope = 1 / (max - min);
            double offset = 1 - slope * max;
            return slope * input + offset;
        }
        return 1;
    }
}
