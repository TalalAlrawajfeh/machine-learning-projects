package neuron.components.propagationfunctions;

import neuron.components.PropagationFunction;

/**
 * Created by u624 on 4/10/17.
 */
public class WeightedSumPropagationFunction implements PropagationFunction {
    @Override
    public double propagate(double[] inputs, double[] weights) {
        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return sum;
    }
}
