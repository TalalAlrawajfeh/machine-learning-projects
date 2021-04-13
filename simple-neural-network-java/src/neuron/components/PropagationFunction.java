package neuron.components;

/**
 * Created by u624 on 4/8/17.
 */
@FunctionalInterface
public interface PropagationFunction {
    double propagate(double[] inputs, double[] weights);
}
