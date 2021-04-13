package neuron.components;

/**
 * Created by u624 on 4/8/17.
 */
public interface ActivationFunction {
    double activate(double input, double previousActivationState);

    default ActivationFunction compose(ActivationFunction activationFunction) {
        return (input, previousActivationState) -> activate(activationFunction.activate(input, previousActivationState),
                previousActivationState);
    }
}
