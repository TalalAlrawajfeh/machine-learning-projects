package neuron.components.outputfunctions;

import neuron.components.OutputFunction;

/**
 * Created by u624 on 4/10/17.
 */
public class IdentityOutputFunction implements OutputFunction {
    @Override
    public double output(double activation) {
        return activation;
    }
}
