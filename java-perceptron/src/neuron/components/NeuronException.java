package neuron.components;

/**
 * Created by u624 on 4/8/17.
 */
public class NeuronException extends RuntimeException {
    public NeuronException() {
        /* default constructor */
    }

    public NeuronException(String s) {
        super(s);
    }

    public NeuronException(String s, Throwable throwable) {
        super(s, throwable);
    }

    public NeuronException(Throwable throwable) {
        super(throwable);
    }

    public NeuronException(String s, Throwable throwable, boolean b, boolean b1) {
        super(s, throwable, b, b1);
    }
}
