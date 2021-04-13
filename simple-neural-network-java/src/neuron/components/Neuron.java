package neuron.components;

import java.util.*;
import java.util.function.Predicate;

/**
 * Created by u624 on 4/8/17.
 */
public class Neuron {
    private static final String RANDOM_WEIGHTS_GENERATOR = "randomWeightsGenerator";
    private static final String PROPAGATION_FUNCTION = "propagationFunction";
    private static final String ACTIVATION_FUNCTION = "activationFunction";
    private static final String NUMBER_OF_INPUTS = "numberOfInputs";
    private static final String OUTPUT_FUNCTION = "outputFunction";

    private Map<String, Predicate<Object>> validations = new HashMap<>();
    private Map<String, String> validationsExceptions = new HashMap<>();
    private Map<String, Object> parameters = new HashMap<>();

    private PropagationFunction propagationFunction;
    private ActivationFunction activationFunction;
    private OutputFunction outputFunction;
    private RandomWeightsGenerator randomWeightsGenerator;
    private int numberOfInputs;

    private double[] weights;
    private double bias;
    private double previousActivationState = 0;

    private Neuron(PropagationFunction propagationFunction, ActivationFunction activationFunction,
                   OutputFunction outputFunction, RandomWeightsGenerator randomWeightsGenerator,
                   int numberOfInputs) {
        parameters.put(PROPAGATION_FUNCTION, propagationFunction);
        parameters.put(ACTIVATION_FUNCTION, activationFunction);
        parameters.put(OUTPUT_FUNCTION, outputFunction);
        parameters.put(RANDOM_WEIGHTS_GENERATOR, randomWeightsGenerator);
        parameters.put(NUMBER_OF_INPUTS, numberOfInputs);
        validateParameters(parameters);
        this.propagationFunction = propagationFunction;
        this.activationFunction = activationFunction;
        this.outputFunction = outputFunction;
        this.randomWeightsGenerator = randomWeightsGenerator;
        this.numberOfInputs = numberOfInputs;
        initializeRandomWeights();
    }

    private void validateParameters(Map<String, Object> parameters) {
        prepareValidations();
        prepareValidationsExceptions();
        Optional<Map.Entry<String, Predicate<Object>>> entry
                = validations.entrySet().stream().filter(e -> !e.getValue().test(parameters.get(e.getKey()))).findAny();
        if (entry.isPresent()) {
            throw new NeuronException(validationsExceptions.get(entry.get().getKey()));
        }
    }

    private void prepareValidations() {
        validations.put(PROPAGATION_FUNCTION, Objects::nonNull);
        validations.put(ACTIVATION_FUNCTION, Objects::nonNull);
        validations.put(OUTPUT_FUNCTION, Objects::nonNull);
        validations.put(RANDOM_WEIGHTS_GENERATOR, Objects::nonNull);
        validations.put(NUMBER_OF_INPUTS, n -> (int) n > 0);
    }

    private void prepareValidationsExceptions() {
        validationsExceptions.put(PROPAGATION_FUNCTION, "Propagation function cannot be null");
        validationsExceptions.put(ACTIVATION_FUNCTION, "Activation function cannot be null");
        validationsExceptions.put(OUTPUT_FUNCTION, "Output function cannot be null");
        validationsExceptions.put(RANDOM_WEIGHTS_GENERATOR, "Random weights generator cannot be null");
        validationsExceptions.put(NUMBER_OF_INPUTS, "Number of inputs must be greater than 0");
    }

    private void initializeRandomWeights() {
        weights = new double[numberOfInputs];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = randomWeightsGenerator.nextRandomWeight();
        }
        bias = randomWeightsGenerator.nextRandomWeight();
    }

    public double feedForward(double[] inputs) {
        double[] biasedInputs = getBiasedInputs(inputs);
        double[] biasedWeights = getBiasedWeights();
        double activation = activationFunction.activate(propagationFunction.propagate(biasedInputs, biasedWeights),
                previousActivationState);
        previousActivationState = activation;
        return outputFunction.output(activation);
    }

    public void train(TrainingUnit[] trainingSet, double learningRate) {
        for (TrainingUnit trainingUnit : trainingSet) {
            double[] inputs = trainingUnit.getInputs();
            double error = trainingUnit.getExpectedOutput() - feedForward(inputs);
            bias += learningRate * error;
            for (int i = 0; i < weights.length; i++) {
                weights[i] += learningRate * error * inputs[i];
            }
        }
    }

    private double[] getBiasedInputs(double[] inputs) {
        double[] biasedInputs = Arrays.copyOf(inputs, inputs.length + 1);
        biasedInputs[inputs.length] = 1;
        return biasedInputs;
    }

    private double[] getBiasedWeights() {
        double[] biasedWeights = Arrays.copyOf(weights, weights.length + 1);
        biasedWeights[weights.length] = bias;
        return biasedWeights;
    }

    public static class NeuronBuilder {
        private PropagationFunction propagationFunction;
        private ActivationFunction activationFunction;
        private OutputFunction outputFunction;
        private RandomWeightsGenerator randomWeightsGenerator;
        private Integer numberOfInputs;

        public NeuronBuilder setPropagationFunction(PropagationFunction propagationFunction) {
            this.propagationFunction = propagationFunction;
            return this;
        }

        public NeuronBuilder setActivationFunction(ActivationFunction activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public NeuronBuilder setOutputFunction(OutputFunction outputFunction) {
            this.outputFunction = outputFunction;
            return this;
        }

        public NeuronBuilder setNumberOfInputs(Integer numberOfInputs) {
            this.numberOfInputs = numberOfInputs;
            return this;
        }

        public NeuronBuilder setRandomWeightsGenerator(RandomWeightsGenerator randomWeightsGenerator) {
            this.randomWeightsGenerator = randomWeightsGenerator;
            return this;
        }

        public Neuron build() {
            return new Neuron(propagationFunction, activationFunction, outputFunction, randomWeightsGenerator,
                    numberOfInputs);
        }
    }
}
