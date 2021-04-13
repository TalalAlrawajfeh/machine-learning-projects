package example;

import neuron.components.Neuron;
import neuron.components.Neuron.NeuronBuilder;
import neuron.components.TrainingUnit;
import neuron.components.activationfunctions.TanhActivationFunction;
import neuron.components.outputfunctions.IdentityOutputFunction;
import neuron.components.propagationfunctions.WeightedSumPropagationFunction;
import neuron.components.randomweightsgenerators.BipolarUniformRandomWeightsGenerator;
import readers.CSVReader;

import java.nio.file.Path;
import java.util.stream.Collectors;

/**
 * Created by u624 on 4/9/17.
 */
public class PerceptronClassifierExample {
    private Neuron neuron = new NeuronBuilder()
            .setPropagationFunction(new WeightedSumPropagationFunction())
            .setActivationFunction(new TanhActivationFunction())
            .setOutputFunction(new IdentityOutputFunction())
            .setRandomWeightsGenerator(new BipolarUniformRandomWeightsGenerator())
            .setNumberOfInputs(2)
            .build();

    public void train(Path trainingDataFile) {
        neuron.train(getTrainingUnits(trainingDataFile), 0.03);
    }

    private TrainingUnit[] getTrainingUnits(Path trainingDataFile) {
        return CSVReader
                .readFile(trainingDataFile)
                .stream()
                .map(row -> new TrainingUnit(new double[]{Double.parseDouble(row.get(0)),
                        Double.parseDouble(row.get(1))}, Double.parseDouble(row.get(2))))
                .collect(Collectors.toList())
                .toArray(new TrainingUnit[0]);
    }

    public int classify(double x, double y) {
        return neuron.feedForward(new double[]{x, y}) <= 0 ? -1 : 1;
    }
}
