package cz.cvut.fel.ida.algebra.values;

import cz.cvut.fel.ida.algebra.values.inits.ValueInitializer;
import org.jetbrains.annotations.NotNull;

import java.text.NumberFormat;
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
    public void transpose() {
        //void
    }

    @Override
    public Value transposedView() {
        return this;
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
    public double get(int i) {
        if (i != 0) {
            LOG.severe("Scalar value: asking for i-th element!");
        }
        return value;
    }

    @Override
    public void set(int i, double value) {
        if (i != 0) {
            LOG.severe("Scalar value: asking for i-th element!");
        }
        this.value = value;
    }

    @Override
    public void increment(int i, double value) {
        if (i != 0) {
            LOG.severe("Scalar value: asking for i-th element!");
        }
        this.value += value;
    }

    @Override
    public String toString(NumberFormat numberFormat) {
        return numberFormat.format(value);
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
        VectorValue clone = new VectorValue(vector.mat.mul(this.value), vector.rowOrientation);
        return clone;
    }

    @Override
    protected MatrixValue times(MatrixValue value) {
        return new MatrixValue(value.mat.mul(this.value), value.rows, value.cols);
    }

    @Override
    protected Value times(TensorValue value) {
        TensorValue clone = value.clone();
        for (int i = 0; i < clone.tensor.values.length; i++) {
            clone.tensor.values[i] *= this.value;
        }
        return clone;
    }

    @Override
    public Value elementTimes(Value value) {
        return value.elementTimes(this);
    }

    @Override
    protected Value elementTimes(ScalarValue value) {
        return new ScalarValue(this.value * value.value);
    }

    @Override
    protected Value elementTimes(VectorValue vector) {
        VectorValue clone = new VectorValue(vector.mat.mul(this.value), vector.rowOrientation);
        return clone;
    }

    @Override
    protected Value elementTimes(MatrixValue value) {
        return new MatrixValue(value.mat.mul(this.value), value.rows, value.cols);
    }

    @Override
    protected Value elementTimes(TensorValue value) {
        TensorValue clone = value.clone();
        for (int i = 0; i < clone.tensor.values.length; i++) {
            clone.tensor.values[i] *= this.value;
        }
        return clone;
    }

    @Override
    public Value kroneckerTimes(Value value) {
        return value.elementTimes(this);
    }

    @Override
    protected Value kroneckerTimes(ScalarValue value) {
        return new ScalarValue(this.value * value.value);
    }

    @Override
    protected Value kroneckerTimes(VectorValue value) {
        VectorValue clone = new VectorValue(value.mat.mul(this.value), value.rowOrientation);
        return clone;
    }

    @Override
    protected Value kroneckerTimes(MatrixValue value) {
        return new MatrixValue(value.mat.mul(this.value), value.rows, value.cols);
    }

    @Override
    protected Value kroneckerTimes(TensorValue value) {
        TensorValue clone = value.clone();
        for (int i = 0; i < clone.tensor.values.length; i++) {
            clone.tensor.values[i] *= this.value;
        }
        return clone;
    }

    @Override
    public Value elementDivideBy(Value value) {
        return value.elementDivideBy(this);
    }

    @Override
    protected ScalarValue elementDivideBy(ScalarValue value) {
        return new ScalarValue(value.value / this.value);
    }

    @Override
    protected Value elementDivideBy(VectorValue vector) {
        VectorValue clone = new VectorValue(vector.mat.rdiv(this.value), vector.rowOrientation);
        return clone;
    }

    @Override
    protected Value elementDivideBy(MatrixValue value) {
        return new MatrixValue(value.mat.div(this.value), value.rows, value.cols);
    }

    @Override
    protected Value elementDivideBy(TensorValue value) {
        TensorValue clone = value.clone();
        for (int i = 0; i < clone.tensor.values.length; i++) {
            clone.tensor.values[i] /= this.value;
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
        VectorValue clone = new VectorValue(value.mat.add(this.value), value.rowOrientation);
        return clone;
    }

    @Override
    protected MatrixValue plus(MatrixValue value) {
        return new MatrixValue(value.mat.add(this.value), value.rows, value.cols);
    }

    @Override
    protected Value plus(TensorValue value) {
        TensorValue clone = value.clone();
        for (int i = 0; i < clone.tensor.values.length; i++) {
            clone.tensor.values[i] += this.value;
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
        VectorValue clone = new VectorValue(value.mat.rsub(this.value), value.rowOrientation);
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
        return new MatrixValue(value.mat.sub(this.value), value.rows, value.cols);
    }

    @Override
    protected Value minus(TensorValue value) {
        TensorValue clone = value.clone();
        for (int i = 0; i < clone.tensor.values.length; i++) {
            clone.tensor.values[i] -= this.value;
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
        value.mat.addi(this.value);
    }

    /**
     * DD - switch of sides!!
     *
     * @param value
     */
    @Override
    protected void incrementBy(MatrixValue value) {
        value.mat.addi(this.value);
    }

    @Override
    protected void incrementBy(TensorValue value) {
        for (int i = 0; i < value.tensor.values.length; i++) {
            value.tensor.values[i] += this.value;
        }
    }

    @Override
    public void elementMultiplyBy(Value value) {
        value.elementMultiplyBy(this);
    }

    @Override
    protected void elementMultiplyBy(ScalarValue value) {
        value.value *= this.value;
    }

    @Override
    protected void elementMultiplyBy(VectorValue value) {
        value.mat.muli(this.value);
    }

    @Override
    protected void elementMultiplyBy(MatrixValue value) {
        value.mat.muli(this.value);
    }

    @Override
    protected void elementMultiplyBy(TensorValue value) {
        for (int i = 0; i < value.tensor.values.length; i++) {
            value.tensor.values[i] *= this.value;
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
        for (int i = 0; i < maxValue.mat.length; i++) {
            if (maxValue.mat.data[i] > this.value) {
                greater++;
            }
        }
        return greater > maxValue.mat.length / 2;
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
                if (maxValue.mat.get(i, j) > this.value) {
                    greater++;
                }
            }
        }
        return greater > maxValue.cols * maxValue.rows / 2;
    }

    @Override
    protected boolean greaterThan(TensorValue maxValue) {
        int greater = 0;
        for (int i = 0; i < maxValue.tensor.values.length; i++) {
            if (maxValue.tensor.values[i] > this.value) {
                greater++;
            }
        }
        return greater > maxValue.tensor.values.length / 2;
    }

    @Override
    public boolean equals(Value obj) {
        if (obj instanceof ScalarValue) {
            if (value == ((ScalarValue) obj).value) {
                return true;
            }
        }
        return false;
    }

    @Override
    public int hashCode() {
        return Double.valueOf(value).hashCode();
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