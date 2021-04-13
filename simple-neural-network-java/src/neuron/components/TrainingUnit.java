package neuron.components;

/**
 * Created by u624 on 4/10/17.
 */
public class TrainingUnit {
    private double[] inputs;
    private double expectedOutput;

    public TrainingUnit() {
        /* default constructor */
    }

    public TrainingUnit(double[] inputs, double expectedOutput) {
        this.inputs = inputs;
        this.expectedOutput = expectedOutput;
    }

    public double[] getInputs() {
        return inputs;
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
    }

    public double getExpectedOutput() {
        return expectedOutput;
    }

    public void setExpectedOutput(double expectedOutput) {
        this.expectedOutput = expectedOutput;
    }
}