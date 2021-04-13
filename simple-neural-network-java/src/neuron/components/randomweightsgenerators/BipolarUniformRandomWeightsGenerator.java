package neuron.components.randomweightsgenerators;

import neuron.components.RandomWeightsGenerator;

import java.util.Random;

/**
 * Created by u624 on 4/10/17.
 */
public class BipolarUniformRandomWeightsGenerator implements RandomWeightsGenerator {
    private Random random = new Random(System.currentTimeMillis());

    @Override
    public double nextRandomWeight() {
        return 1 - (double) random.nextInt(1001) / 1000 * 2;
    }
}
