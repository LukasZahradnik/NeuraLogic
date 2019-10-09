package networks.computation.evaluation.values;

import com.sun.istack.internal.NotNull;
import networks.computation.evaluation.values.distributions.ValueInitializer;
import settings.Settings;

import java.util.Iterator;
import java.util.function.Function;
import java.util.logging.Logger;

/**
 * Created by gusta on 8.3.17.
 */
public class ScalarValue extends Value {
    private static final Logger LOG = Logger.getLogger(ScalarValue.class.getName());

    /**
     * The actual value
     */
    public double value;

    public ScalarValue() {
        value = 0;
    }

    public ScalarValue(double val) {
        value = val;
    }

    public ScalarValue(ValueInitializer valueInitializer) {
        initialize(valueInitializer);
    }

    @NotNull
    @Override
    public Iterator<Double> iterator() {
        return new ValueIterator();
    }

    protected class ValueIterator implements Iterator<Double> {
        int i = 0;

        @Override
        public boolean hasNext() {
            return i == 0;
        }

        @Override
        public Double next() {
            i++;
            return value;
        }
    }

    @Override
    public void initialize(ValueInitializer valueInitializer) {
        valueInitializer.initScalar(this);
    }

    @Override
    public ScalarValue zero() {
        value = 0;
        return this;
    }

    @Override
    public Value clone() {
        return new ScalarValue(value);
    }

    @Override
    public Value getForm() {
        return new ScalarValue(0);
    }

    @Override
    public int[] size() {
        return new int[0];
    }

    @Override
    public Value apply(Function<Double, Double> function) {
        return new ScalarValue(function.apply(value));
    }

    @Override
    public String toString() {
        return Settings.nf.format(value);
    }

    /**
     * Default Double Dispatch
     *
     * @param value
     * @return
     */
    @Override
    public Value times(Value value) {
        return value.times(this);
    }

    @Override
    protected ScalarValue times(ScalarValue value) {
        return new ScalarValue(this.value * value.value);
    }

    @Override
    protected VectorValue times(VectorValue vector) {
        VectorValue clone = vector.clone();
        for (int i = 0; i < vector.values.length; i++) {
            clone.values[i] *= this.value;
        }
        return clone;
    }

    @Override
    protected MatrixValue times(MatrixValue value) {
        MatrixValue clone = value.clone();
        for (int i = 0; i < clone.rows; i++) {
            for (int j = 0; j < clone.cols; j++) {
                clone.values[i][j] *= this.value;
            }
        }
        return clone;
    }

    /**
     * Default Double Dispatch
     *
     * @param value
     * @return
     */
    @Override
    public Value plus(Value value) {
        return value.plus(this);
    }

    @Override
    protected ScalarValue plus(ScalarValue value) {
        return new ScalarValue(this.value + value.value);
    }

    @Override
    protected VectorValue plus(VectorValue value) {
        VectorValue clone = value.clone();
        for (int i = 0; i < clone.values.length; i++) {
            clone.values[i] += this.value;
        }
        return clone;
    }

    @Override
    protected MatrixValue plus(MatrixValue value) {
        MatrixValue clone = value.clone();
        for (int i = 0; i < clone.rows; i++) {
            for (int j = 0; j < clone.cols; j++) {
                clone.values[i][j] += this.value;
            }
        }
        return clone;
    }

    /**
     * Default Double Dispatch
     *
     * @param value
     * @return
     */
    @Override
    public Value minus(Value value) {
        return value.minus(this);
    }

    /**
     * DD - switch of sides??!! todo check
     *
     * @param value
     * @return
     */
    @Override
    protected ScalarValue minus(ScalarValue value) {
        return new ScalarValue(value.value - this.value);
    }

    /**
     * DD - switch of sides??!! todo check
     *
     * @param value
     * @return
     */
    @Override
    protected VectorValue minus(VectorValue value) {
        VectorValue clone = value.clone();
        for (int i = 0; i < clone.values.length; i++) {
            clone.values[i] -= this.value;
        }
        return clone;
    }

    /**
     * DD - switch of sides??!! todo check
     *
     * @param value
     * @return
     */
    @Override
    protected MatrixValue minus(MatrixValue value) {
        MatrixValue clone = value.clone();
        for (int i = 0; i < clone.rows; i++) {
            for (int j = 0; j < clone.cols; j++) {
                clone.values[i][j] -= this.value;
            }
        }
        return clone;
    }

    /**
     * Default Double Dispatch
     *
     * @param value
     */
    @Override
    public void incrementBy(Value value) {
        value.incrementBy(this);
    }

    /**
     * DD - switch of sides!!
     *
     * @param value
     */
    @Override
    protected void incrementBy(ScalarValue value) {
        value.value += this.value;
    }

    /**
     * DD - switch of sides!!
     *
     * @param value
     */
    @Override
    protected void incrementBy(VectorValue value) {
        for (int i = 0; i < value.values.length; i++) {
            value.values[i] += this.value;
        }
    }

    /**
     * DD - switch of sides!!
     *
     * @param value
     */
    @Override
    protected void incrementBy(MatrixValue value) {
        for (int i = 0; i < value.rows; i++) {
            for (int j = 0; j < value.cols; j++) {
                value.values[i][j] += this.value;
            }
        }
    }

    @Override
    public boolean greaterThan(Value maxValue) {
        return maxValue.greaterThan(this);
    }

    @Override
    protected boolean greaterThan(ScalarValue maxValue) {
        return maxValue.value > this.value;
    }

    /**
     * Greater iff greater than majority
     *
     * @param maxValue
     * @return
     */
    @Override
    protected boolean greaterThan(VectorValue maxValue) {
        int greater = 0;
        for (int i = 0; i < maxValue.values.length; i++) {
            if (maxValue.values[i] > this.value) {
                greater++;
            }
        }
        return greater > maxValue.values.length / 2;
    }

    /**
     * Greater iff greater than majority
     *
     * @param maxValue
     * @return
     */
    @Override
    protected boolean greaterThan(MatrixValue maxValue) {
        int greater = 0;
        for (int i = 0; i < maxValue.rows; i++) {
            for (int j = 0; j < maxValue.cols; j++) {
                if (maxValue.values[i][j] > this.value) {
                    greater++;
                }
            }
        }
        return greater > maxValue.cols * maxValue.rows / 2;
    }

    @Override
    public int hashCode() {
        return (int) value * 1000000;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof ScalarValue) {
            ScalarValue obj1 = (ScalarValue) obj;
            if (obj1.value == this.value) {
                return true;
            }
        }
        return false;
    }
}