package cz.cvut.fel.ida.algebra.values;

import cz.cvut.fel.ida.algebra.values.inits.ValueInitializer;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.jetbrains.annotations.NotNull;

import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;
import java.util.logging.Logger;

/**
 * We consider vectors as both column and row vectors, default is column orientation.
 * Created by gusta on 8.3.17.
 */
public class VectorValue extends Value {
    private static final Logger LOG = Logger.getLogger(VectorValue.class.getName());
    /**
     * The actual vector of values
     */
    public DoubleMatrix mat;

    /**
     * Information about orientation/transposition
     */
    public boolean rowOrientation = false;

    public VectorValue(int size) {
        mat = new DoubleMatrix(size);
        mat.reshape(rows(), cols());
    }

    public VectorValue(DoubleMatrix matrix) {
        mat = matrix;
        mat.reshape(rows(), cols());
    }

    public VectorValue(DoubleMatrix matrix, boolean rowOrientation) {
        this.rowOrientation = rowOrientation;
        mat = matrix;
        mat.reshape(rows(), cols());
    }

    public VectorValue(List<Double> vector) {
        mat = new DoubleMatrix(vector.stream().mapToDouble(d -> d).toArray());
        mat.reshape(rows(), cols());
    }

    public VectorValue(int size, ValueInitializer valueInitializer) {
        initialize(valueInitializer);

        mat = new DoubleMatrix(size);
        mat.reshape(rows(), cols());
    }

    public VectorValue(double[] values) {
        mat = new DoubleMatrix(values);
        mat.reshape(rows(), cols());
    }

    public VectorValue(double[] values, boolean rowOrientation) {
        this.rowOrientation = rowOrientation;

        mat = new DoubleMatrix(values);
        mat.reshape(rows(), cols());
    }

    public VectorValue(int size, boolean rowOrientation) {
        this.rowOrientation = rowOrientation;

        mat = new DoubleMatrix(size);
        mat.reshape(rows(), cols());
    }


    protected int rows() {
        if (rowOrientation) {
            return 1;
        } else {
            return mat.length;
        }
    }

    protected int cols() {
        if (!rowOrientation) {
            return 1;
        } else {
            return mat.length;
        }
    }

    public void setRowOrientation(boolean rowOrientation) {
        this.rowOrientation = rowOrientation;
        this.mat.reshape(rows(), cols());
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
            return i < mat.length;
        }

        @Override
        public Double next() {
            return mat.get(i++);
        }
    }

    @Override
    public void initialize(ValueInitializer valueInitializer) {
        valueInitializer.initVector(this);
    }

    @Override
    public VectorValue zero() {
        for (int i = 0; i < mat.length; i++) {
            mat.data[i] = 0;
        }
        return this;
    }

    @Override
    public VectorValue clone() {
        VectorValue clone = new VectorValue(mat.dup(), rowOrientation);
        return clone;
    }

    @Override
    public VectorValue getForm() {
        VectorValue form = new VectorValue(mat.length);
        form.rowOrientation = rowOrientation;
        form.mat.reshape(form.cols(), form.rows());

        return form;
    }

    @Override
    public void transpose() {
        rowOrientation = !rowOrientation;
        mat.reshape(cols(), rows());
    }

    @Override
    public Value transposedView() {
        return new VectorValue(mat.dup(), !rowOrientation);
    }

    @Override
    public int[] size() {
        if (rowOrientation) {
            return new int[]{1, mat.length};
        } else {
            return new int[]{mat.length, 1};
        }
    }

    @Override
    public Value apply(Function<Double, Double> function) {
        VectorValue result = new VectorValue(mat.length);

        for (int i = 0; i < mat.length; i++) {
            result.mat.put(i, function.apply(mat.get(i)));
        }
        return result;
    }

    @Override
    public double get(int i) {
        return mat.data[i];
    }

    @Override
    public void set(int i, double value) {
        mat.data[i] = value;
    }

    @Override
    public void increment(int i, double value) {
        mat.data[i] += value;
    }

    @Override
    public String toString(NumberFormat numberFormat) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < mat.length; i++) {
            sb.append(",").append(numberFormat.format(mat.data[i]));
        }
        sb.replace(0, 1, "[");
        sb.append("]");
        return sb.toString();
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
    protected VectorValue times(ScalarValue value) {
        VectorValue result = new VectorValue(mat.mul(value.value), this.rowOrientation);
        return result;
    }

    /**
     * Dot product vs matrix multiplication depending on orientation of the vectors
     *
     * @param value
     * @return
     */
    @Override
    protected Value times(VectorValue value) {
        if (value.rowOrientation && !this.rowOrientation && value.mat.length == mat.length) {
            return new ScalarValue(mat.dot(value.mat));
        } else if (!value.rowOrientation && this.rowOrientation) {
            LOG.finest(() -> "Performing vector x vector matrix multiplication.");
            double[][] resultValues = new double[value.mat.length][mat.length];
            for (int i = 0; i < value.mat.length; i++) {
                for (int j = 0; j < mat.length; j++) {
                    resultValues[i][j] = value.mat.data[i] * mat.data[j];
                }
            }

            return new MatrixValue(resultValues);
        } else {
            String err = "Incompatible dimensions for vector multiplication: " + Arrays.toString(value.size()) + " vs " + Arrays.toString(size()) + " (try transposition)";
            LOG.severe(err);
            throw new ArithmeticException(err); // todo measure if any cost of this
//            return null;
        }
    }

    /**
     * Vectors are by default taken as columns, so multiplication is against the rows of the matrix.
     * <p>
     * The result is naturally a vector of size rows(Matrix)
     *
     * @param value
     * @return
     */
    @Override
    protected VectorValue times(MatrixValue value) {
        if (value.cols != mat.length) {
            String err = "Matrix row length mismatch with vector length for multiplication: " + value.cols + " vs." + mat.length;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        if (value.cols > 1 && rowOrientation) {
            LOG.severe("Multiplying matrix with a row-oriented vector!");
            throw new ArithmeticException("Multiplying matrix with a row-oriented vector!");
        }

        this.mat.reshape(rows(), cols());

        VectorValue result = new VectorValue(value.mat.mmul(this.mat));
        System.out.println("---");
        System.out.println(result.mat);
        DoubleMatrix res = new DoubleMatrix(rows());

        System.out.println(SimpleBlas.gemm(1.0, value.mat, mat, 0.0, res));


        return result;
    }

    @Override
    protected Value times(TensorValue value) {
        throw new ArithmeticException("Algebraic operation between Tensor and Vector are not implemented yet");
    }

    @Override
    public Value elementTimes(Value value) {
        return value.elementTimes(this);
    }

    @Override
    protected Value elementTimes(ScalarValue value) {
        return new VectorValue(mat.mul(value.value), rowOrientation);
    }

    @Override
    protected Value elementTimes(VectorValue value) {
        if (value.mat.length != mat.length) {
            String err = "Vector elementTimes dimension mismatch: " + value.mat.length + " vs." + mat.length;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }

        return new VectorValue(mat.mul(value.mat), rowOrientation);
    }

    @Override
    protected Value elementTimes(MatrixValue value) {
        LOG.warning("Calculation matrix element-wise product with vector...");
        if (value.cols != mat.length) {
            String err = "Matrix elementTimes vector broadcast dimension mismatch: " + value.cols + " vs." + mat.length;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        double[][] resultValues = new double[value.rows][value.cols];

        for (int i = 0; i < value.rows; i++) {
            for (int j = 0; j < value.cols; j++) {
                resultValues[i][j] = value.mat.get(i, j) * this.mat.data[j];
            }
        }
        return new MatrixValue(resultValues);
    }

    @Override
    protected Value elementTimes(TensorValue value) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Vector are not implemented yet");
    }

    @Override
    public Value kroneckerTimes(Value value) {
        return value.kroneckerTimes(this);
    }

    @Override
    protected Value kroneckerTimes(ScalarValue value) {
        return new VectorValue(mat.mul(value.value), rowOrientation);
    }

    @Override
    protected Value kroneckerTimes(VectorValue value) {
        int rows = rows() * value.rows();
        int cols = cols() * value.cols();

        if (rows == 1 || cols == 1) {
            double[] resultValues = new double[rows * cols];

            for (int i = 0; i < value.mat.length; i++) {
                for (int j = 0; j < mat.length; j++) {
                    resultValues[i * mat.length + j] = value.mat.data[i] * mat.data[j];
                }
            }
            return new VectorValue(resultValues, rows == 1);
        } else {
            double[][] resultValues = new double[rows][cols];

            if (rowOrientation) {
                for (int i = 0; i < value.mat.length; i++) {
                    for (int j = 0; j < mat.length; j++) {
                        resultValues[i][j] = value.mat.data[i] * mat.data[j];
                    }
                }
            } else {
                for (int i = 0; i < value.mat.length; i++) {
                    for (int j = 0; j < mat.length; j++) {
                        resultValues[j][i] = value.mat.data[i] * mat.data[j];
                    }
                }
            }
            return new MatrixValue(resultValues);
        }
    }

    @Override
    protected Value kroneckerTimes(MatrixValue matrix) {
        int rows = rows() * matrix.rows;
        int cols = cols() * matrix.cols;

        double[][] resultValues = new double[rows][cols];
        if (rowOrientation) {
            for (int r1 = 0; r1 < matrix.rows; r1++) {
                for (int c1 = 0; c1 < matrix.cols; c1++) {
                    for (int k = 0; k < mat.length; k++) {
                        resultValues[r1][c1 * mat.length + k] = matrix.mat.get(r1, c1) * mat.data[k];
                    }
                }
            }
        } else {
            for (int r1 = 0; r1 < matrix.rows; r1++) {
                for (int c1 = 0; c1 < matrix.cols; c1++) {
                    for (int k = 0; k < mat.length; k++) {
                        resultValues[r1 * mat.length + k][c1] = matrix.mat.get(r1, c1) * mat.data[k];
                    }
                }
            }
        }
        return new MatrixValue(resultValues);
    }

    @Override
    protected Value kroneckerTimes(TensorValue value) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Vector are not implemented yet");
    }

    @Override
    public Value elementDivideBy(Value value) {
        return value.elementDivideBy(this);
    }

    @Override
    protected Value elementDivideBy(ScalarValue value) {
        return new VectorValue(mat.rdiv(value.value), rowOrientation);
    }

    @Override
    protected Value elementDivideBy(VectorValue value) {
        if (value.mat.length != mat.length) {
            String err = "Vector elementTimes dimension mismatch: " + value.mat.length + " vs." + mat.length;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        return new VectorValue(mat.rdiv(value.mat), rowOrientation);
    }

    @Override
    protected Value elementDivideBy(MatrixValue value) {
        LOG.warning("Calculation matrix element-wise product with vector...");
        if (value.cols != mat.length) {
            String err = "Matrix elementTimes vector broadcast dimension mismatch: " + value.cols + " vs." + mat.length;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        return new MatrixValue(mat.rdiv(value.mat), value.rows, value.cols);
    }

    @Override
    protected Value elementDivideBy(TensorValue value) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Vector are not implemented yet");
    }

    /**
     * Default double dispatch
     *
     * @param value
     * @return
     */
    @Override
    public Value plus(Value value) {
        return value.plus(this);
    }

    @Override
    protected VectorValue plus(ScalarValue value) {
        return new VectorValue(mat.add(value.value), rowOrientation);
    }

    @Override
    protected VectorValue plus(VectorValue value) {
        if (value.mat.length != mat.length) {
            String err = "Vector element plus dimension mismatch: " + value.mat.length + " vs." + mat.length;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        return new VectorValue(value.mat.add(mat), rowOrientation);
    }

    /**
     * This is just not allowed unless the matrix is degenerated to vector, but it is just safer to disallow.
     *
     * @param value
     * @return
     */
    @Override
    protected Value plus(MatrixValue value) {
        throw new ArithmeticException("Incompatible summation of matrix plus vector ");
    }

    @Override
    protected Value plus(TensorValue value) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Vector are not implemented yet");
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

    @Override
    protected Value minus(ScalarValue value) {
        return new VectorValue(mat.rsub(value.value), rowOrientation);
    }

    @Override
    protected Value minus(VectorValue value) {
        if (value.mat.length != mat.length) {
            String err = "Vector minus dimension mismatch: " + value.mat.length + " vs." + mat.length;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        return new VectorValue(mat.rsub(value.mat), rowOrientation);
    }

    @Override
    protected Value minus(MatrixValue value) {
        throw new ArithmeticException("Incompatible dimensions of algebraic operation - matrix minus vector");
    }

    @Override
    protected Value minus(TensorValue value) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Vector are not implemented yet");
    }

    /**
     * Default double dispatch
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
        throw new ArithmeticException("Incompatible dimensions of algebraic operation - scalar increment by vector");
    }

    /**
     * DD - switch of sides!!
     *
     * @param value
     */
    @Override
    protected void incrementBy(VectorValue value) {
        if (value.mat.length != mat.length) {
            String err = "Vector incrementBy dimension mismatch: " + value.mat.length + " vs." + mat.length;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        value.mat.addi(mat);
    }

    /**
     * DD - switch of sides!!
     *
     * @param value
     */
    @Override
    protected void incrementBy(MatrixValue value) {
        String err = "Incompatible dimensions of algebraic operation - matrix increment by vector";
        LOG.severe(err);
        throw new ArithmeticException(err);

    }

    @Override
    protected void incrementBy(TensorValue value) {
        throw new ArithmeticException("Algebraic operation between Tensor and Vector are not implemented yet");
    }

    @Override
    public void elementMultiplyBy(Value value) {
        value.elementMultiplyBy(this);
    }

    @Override
    protected void elementMultiplyBy(ScalarValue value) {
        String err = "Incompatible dimensions of algebraic operation - scalar multiplyBy by vector";
        LOG.severe(err);
        throw new ArithmeticException(err);
    }

    @Override
    protected void elementMultiplyBy(VectorValue value) {
        if (value.mat.length != mat.length) {
            String err = "Vector multiplyBy dimension mismatch: " + value.mat.length + " vs." + mat.length;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        value.mat.muli(mat);
    }

    @Override
    protected void elementMultiplyBy(MatrixValue value) {
        String err = "Incompatible dimensions of algebraic operation - matrix multiplyBy by vector";
        LOG.severe(err);
        throw new ArithmeticException(err);
    }

    @Override
    protected void elementMultiplyBy(TensorValue value) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Vector are not implemented yet");
    }

    /**
     * Default double dispatch
     *
     * @param maxValue
     * @return
     */
    @Override
    public boolean greaterThan(Value maxValue) {
        return maxValue.greaterThan(this);
    }

    @Override
    protected boolean greaterThan(ScalarValue maxValue) {
        int greater = 0;
        for (int i = 0; i < mat.length; i++) {
            if (maxValue.value > mat.data[i]) {
                greater++;
            }
        }
        return greater > mat.length / 2;
    }

    @Override
    protected boolean greaterThan(VectorValue maxValue) {
        if (maxValue.mat.length != mat.length) {
            String err = "Vector greaterThan dimension mismatch: " + maxValue.mat.length + " vs." + mat.length;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        int greater = 0;
        for (int i = 0; i < mat.length; i++) {
            if (maxValue.mat.data[i] > mat.data[i]) {
                greater++;
            }
        }
        return greater > mat.length / 2;
    }

    @Override
    protected boolean greaterThan(MatrixValue maxValue) {
        LOG.severe("Incompatible dimensions of algebraic operation - matrix greaterThan vector");
        return false;
    }

    @Override
    protected boolean greaterThan(TensorValue maxValue) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Vector are not implemented yet");
    }

    @Override
    public boolean equals(Value obj) {
        if (obj instanceof VectorValue) {
            if (Arrays.equals(mat.data, ((VectorValue) obj).mat.data)) {
                return true;
            }
        }
        return false;
    }


    @Override
    public int hashCode() {
        long hashCode = 1;
        for (int i = 0; i < mat.length; i++)
            hashCode = 31 * hashCode + Double.valueOf(mat.data[i]).hashCode();
        return Long.hashCode(hashCode);
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof VectorValue)) {
            return false;
        }
        VectorValue vectorValue = (VectorValue) obj;
        if (vectorValue.mat.length != mat.length)
            return false;

        for (int i = 0; i < mat.length; i++) {
            if (mat.data[i] != vectorValue.mat.data[i])
                return false;
        }
        return true;
    }
}