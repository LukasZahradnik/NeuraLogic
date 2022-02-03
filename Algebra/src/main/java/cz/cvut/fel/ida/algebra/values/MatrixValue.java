package cz.cvut.fel.ida.algebra.values;

import cz.cvut.fel.ida.algebra.values.inits.ValueInitializer;
import org.jblas.DoubleMatrix;
import org.jblas.NativeBlas;
import org.jblas.SimpleBlas;
import org.jetbrains.annotations.NotNull;

import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;
import java.util.logging.Logger;

/**
 * By default we consider matrices stored row-wise, i.e. M[rows][cols].
 *
 * @see VectorValue
 * <p>
 * Created by gusta on 8.3.17.
 */
public class MatrixValue extends Value {
    private static final Logger LOG = Logger.getLogger(MatrixValue.class.getName());

    public int rows;
    public int cols;

    /**
     * The actual values
     */

    public DoubleMatrix mat;

    public MatrixValue(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;

        mat = new DoubleMatrix(rows, cols);

        mat.reshape(this.rows, this.cols);
    }

    public MatrixValue(DoubleMatrix matrix, int rows, int cols) {
        this.rows = rows;
        this.cols = cols;

        mat = matrix;
        mat.reshape(this.rows, this.cols);
    }

    public MatrixValue(List<List<Double>> vectors) {
        this.rows = vectors.size();
        this.cols = vectors.get(0).size();
        double[][] values = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                values[i][j] = vectors.get(i).get(j);
            }
        }
        mat = new DoubleMatrix(values);
        mat.reshape(this.rows, this.cols);
    }

    public MatrixValue(double[][] values) {
        this.rows = values.length;
        this.cols = values[0].length;

        mat = new DoubleMatrix(values);
        mat.reshape(this.rows, this.cols);
    }


    @NotNull
    @Override
    public Iterator<Double> iterator() {
        return new MatrixValue.ValueIterator();
    }

    /**
     * The default iteration is row-wise, i.e. all the elements from first row go before all the elements from second rows etc., just like the default storage of a matrix.
     */
    protected class ValueIterator implements Iterator<Double> {
        int row;
        int col;

        final int maxCol = cols - 1;
        final int maxRow = rows - 1;

        @Override
        public boolean hasNext() {
            return row < maxRow || col < maxCol;
        }

        @Override
        public Double next() {
            double next = mat.get(row, col);
            if (col < cols - 1)
                col++;
            else {
                row++;
                col = 0;
            }
            return next;
        }
    }

    @Override
    public void initialize(ValueInitializer valueInitializer) {
        valueInitializer.initMatrix(this);
    }

    @Override
    public MatrixValue zero() {
        for (int i = 0; i < mat.length; i++) {
            mat.data[i] = 0;
        }
        return this;
    }

    @Override
    public MatrixValue clone() {
        MatrixValue clone = new MatrixValue(this.mat.dup(), rows, cols);
        return clone;
    }

    @Override
    public MatrixValue getForm() {
        return new MatrixValue(rows, cols);
    }

    @Override
    public void transpose() {
        mat = mat.transpose();

        int tmp = rows;
        rows = cols;
        cols = tmp;
    }

    @Override
    public Value transposedView() {
//        LOG.severe("Transposed view of a matrix (without actual transposition) not implemented, returning a transposed copy instead!");
        MatrixValue value = new MatrixValue(mat.transpose(), cols, rows);
        return value;
    }

    @Override
    public int[] size() {
        return new int[]{rows, cols};
    }

    @Override
    public Value apply(Function<Double, Double> function) {
        MatrixValue result = new MatrixValue(rows, cols);

        for (int i = 0; i < result.mat.length; ++i) {
            result.mat.data[i] = function.apply(mat.data[i]);
        }

        return result;
    }

    @Override
    public double get(int i) {
        return mat.get(i / rows, i % cols);
    }

    @Override
    public void set(int i, double value) {
        mat.put(i / rows, i % cols, value);
    }

    @Override
    public void increment(int i, double value) {
        mat.put(i / rows, i % cols, mat.get(i / rows, i % cols) + value);
    }


    @Override
    public String toString(NumberFormat numberFormat) {
        StringBuilder sb = new StringBuilder();
        sb.append("[\n");
        for (int j = 0; j < mat.rows; j++) {
            sb.append("[");
            for (int i = 0; i < mat.columns; i++) {
                sb.append(numberFormat.format(mat.get(j, i))).append(",");
            }
            sb.replace(sb.length()-1, sb.length(), "],\n");
        }
        sb.replace(sb.length()-2, sb.length(), "\n]");
        return sb.toString();
    }

    /**
     * DDD
     *
     * @param value
     * @return
     */
    @Override
    public Value times(Value value) {
        return value.times(this);
    }

    @Override
    protected MatrixValue times(ScalarValue value) {
        MatrixValue clone = this.clone();

        clone.mat.muli(value.value);

        return clone;
    }

    @Override
    protected VectorValue times(VectorValue value) {
        if (rows != value.mat.length) {
            String err = "Matrix row length mismatch with vector length for multiplication: " + rows + " vs." + value.mat.length;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        if (!value.rowOrientation) {
            throw new ArithmeticException("Column vector times matrix, try transposition. Vector size = " + value.mat.length);
        }

        return new VectorValue(value.mat.mmul(mat));
    }

    /**
     * Remember that the double-dispatch is changing rhs and lhs sides
     * <p>
     * MatrixValue lhs = value;
     * MatrixValue rhs = this;
     * <p>
     * <p>
     * todo use the Strassen algorithm for bigger matrices or just outsource to eigen!
     *
     * @param value
     * @return
     */
    @Override
    protected MatrixValue times(MatrixValue value) {
        if (value.cols != rows) {
            String err = "Matrix to matrix dimension mismatch for multiplication" + value.cols + " != " + rows;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }

        return new MatrixValue(value.mat.mmul(mat), value.rows, this.cols);
    }

    @Override
    protected Value times(TensorValue value) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Matrix are not implemented yet");
    }

    @Override
    public Value elementTimes(Value value) {
        return value.elementTimes(this);
    }

    @Override
    protected Value elementTimes(ScalarValue value) {
        return new MatrixValue(mat.mul(value.value), rows, cols);
    }

    @Override
    protected Value elementTimes(VectorValue value) {
        LOG.warning("Calculation vector element-wise product with matrix...");
        if (rows != value.mat.length) {
            String err = "Matrix row length mismatch with vector length for multiplication" + rows + " vs." + value.mat.length;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        return new MatrixValue(mat.mul(value.mat), rows, cols);
    }

    @Override
    protected Value elementTimes(MatrixValue value) {
        if (value.cols != cols || value.rows != rows) {
            String err = "Matrix elementTimes dimension mismatch: " + Arrays.toString(this.size()) + " vs." + Arrays.toString(value.size());
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        return new MatrixValue(value.mat.mul(mat), rows, cols);
    }

    @Override
    protected Value elementTimes(TensorValue value) {
        throw new ArithmeticException("Algebraic operation between Tensor and Matrix are not implemented yet");
    }

    @Override
    public Value kroneckerTimes(Value value) {
        return value.kroneckerTimes(this);
    }

    @Override
    protected Value kroneckerTimes(ScalarValue value) {
        return new MatrixValue(mat.mul(value.value), rows, cols);
    }

    @Override
    protected Value kroneckerTimes(VectorValue vectorValue) {
        int rows = vectorValue.rows() * this.rows;
        int cols = vectorValue.cols() * this.cols;

        double[][] resultValues = new double[rows][cols];
        if (vectorValue.rowOrientation) {
            for (int c1 = 0; c1 < vectorValue.mat.length; c1++) {
                for (int r2 = 0; r2 < this.rows; r2++) {
                    for (int c2 = 0; c2 < this.cols; c2++) {
                        resultValues[r2][c1 * this.cols + c2] = vectorValue.mat.data[c1] * mat.get(r2, c2);
                    }
                }
            }
        } else {
            for (int r1 = 0; r1 < vectorValue.mat.length; r1++) {
                for (int r2 = 0; r2 < this.rows; r2++) {
                    for (int c2 = 0; c2 < this.cols; c2++) {
                        resultValues[r1 * this.rows + r2][c2] = vectorValue.mat.data[r1] * mat.get(r2, c2);
                    }
                }
            }
        }
        return new MatrixValue(resultValues);
    }

    @Override
    protected Value kroneckerTimes(MatrixValue otherValue) {
        int rows = this.rows * otherValue.rows;
        int cols = this.cols * otherValue.cols;

        double[][] resultValues = new double[rows][cols];

        for (int r1 = 0; r1 < otherValue.rows; r1++) {
            for (int c1 = 0; c1 < otherValue.cols; c1++) {
                for (int r2 = 0; r2 < this.rows; r2++) {
                    for (int c2 = 0; c2 < this.cols; c2++) {
                        resultValues[r1 * this.rows + r2][c1 * this.cols + c2] = otherValue.mat.get(r1, c1) * mat.get(r2, c2);
                    }
                }
            }
        }
        return new MatrixValue(resultValues);
    }

    @Override
    protected Value kroneckerTimes(TensorValue value) {
        throw new ArithmeticException("Algebraic operation between Tensor and Matrix are not implemented yet");
    }

    @Override
    public Value elementDivideBy(Value value) {
        return value.elementDivideBy(this);
    }

    @Override
    protected Value elementDivideBy(ScalarValue value) {
        return new MatrixValue(mat.div(value.value), rows, cols);
    }

    @Override
    protected Value elementDivideBy(VectorValue value) {
        LOG.warning("Calculation vector element-wise division with matrix...");
        if (rows != value.mat.length) {
            String err = "Matrix row length mismatch with vector length for multiplication" + rows + " vs." + value.mat.length;
            LOG.severe(err);
            throw new ArithmeticException(err);
        }

        return new MatrixValue(mat.rdiv(value.mat), rows, cols);

    }

    @Override
    protected Value elementDivideBy(MatrixValue value) {
        if (value.cols != cols || value.rows != rows) {
            String err = "Matrix elementTimes dimension mismatch: " + Arrays.toString(this.size()) + " vs." + Arrays.toString(value.size());
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        return new MatrixValue(mat.rdiv(value.mat), rows, cols);
    }

    @Override
    protected Value elementDivideBy(TensorValue value) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Matrix are not implemented yet");
    }

    /**
     * DDD
     *
     * @param value
     * @return
     */
    @Override
    public Value plus(Value value) {
        return value.plus(this);
    }

    @Override
    protected MatrixValue plus(ScalarValue value) {
        return new MatrixValue(mat.add(value.value), rows, cols);
    }

    @Override
    protected Value plus(VectorValue value) {
        throw new ArithmeticException("Incompatible summation of matrix plus vector ");
    }

    @Override
    protected Value plus(MatrixValue value) {
        if (rows != value.rows || cols != value.cols) {
            String err = "Matrix plus dimension mismatch: " + Arrays.toString(this.size()) + " vs." + Arrays.toString(value.size());
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        return new MatrixValue(value.mat.add(mat), rows, cols);
    }

    @Override
    protected Value plus(TensorValue value) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Matrix are not implemented yet");
    }

    /**
     * DDD
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
        return new MatrixValue(mat.rsub(value.value), rows, cols);
    }

    @Override
    protected Value minus(VectorValue value) {
        LOG.severe("Incompatible dimensions of algebraic operation - vector minus matrix");
        return null;
    }

    @Override
    protected Value minus(MatrixValue value) {
        if (rows != value.rows || cols != value.cols) {
            String err = "Matrix minus dimension mismatch: " + Arrays.toString(this.size()) + " vs." + Arrays.toString(value.size());
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        return new MatrixValue(mat.rsub(value.mat), rows, cols);
    }

    @Override
    protected Value minus(TensorValue value) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Matrix are not implemented yet");
    }

    /**
     * DDD
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
        String err = "Incompatible dimensions of algebraic operation - scalar incrementBy by matrix";
        LOG.severe(err);
        throw new ArithmeticException(err);
    }

    /**
     * DD - switch of sides!!
     *
     * @param value
     */
    @Override
    protected void incrementBy(VectorValue value) {
        String err = "Incompatible dimensions of algebraic operation - vector incrementBy by matrix";
        LOG.severe(err);
        throw new ArithmeticException(err);
    }

    /**
     * DD - switch of sides!!
     *
     * @param value
     */
    @Override
    protected void incrementBy(MatrixValue value) {
        if (rows != value.rows || cols != value.cols) {
            String err = "Matrix incrementBy dimension mismatch: " + Arrays.toString(this.size()) + " vs." + Arrays.toString(value.size());
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        value.mat.addi(mat);
    }

    @Override
    protected void incrementBy(TensorValue value) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Matrix are not implemented yet");
    }

    @Override
    public void elementMultiplyBy(Value value) {
        value.elementMultiplyBy(this);
    }

    @Override
    protected void elementMultiplyBy(ScalarValue value) {
        String err = "Incompatible dimensions of algebraic operation - scalar multiplyBy by matrix";
        LOG.severe(err);
        throw new ArithmeticException(err);
    }

    @Override
    protected void elementMultiplyBy(VectorValue value) {
        String err = "Incompatible dimensions of algebraic operation - vector multiplyBy by matrix";
        LOG.severe(err);
        throw new ArithmeticException(err);
    }

    @Override
    protected void elementMultiplyBy(MatrixValue value) {
        if (rows != value.rows || cols != value.cols) {
            String err = "Matrix multiplyBy dimension mismatch: " + Arrays.toString(this.size()) + " vs." + Arrays.toString(value.size());
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        value.mat.muli(mat);
    }

    @Override
    protected void elementMultiplyBy(TensorValue value) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Matrix are not implemented yet");
    }

    /**
     * DDD
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
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (mat.get(i, j) < maxValue.value) {
                    greater++;
                }
            }
        }
        return greater > cols * rows / 2;
    }

    @Override
    protected boolean greaterThan(VectorValue maxValue) {
        LOG.severe("Incompatible dimensions of algebraic operation - vector greaterThan matrix");
        return false;
    }

    @Override
    protected boolean greaterThan(MatrixValue maxValue) {
        if (rows != maxValue.rows || cols != maxValue.cols) {
            String err = "Matrix greaterThan dimension mismatch: " + Arrays.toString(this.size()) + " vs." + Arrays.toString(maxValue.size());
            LOG.severe(err);
            throw new ArithmeticException(err);
        }
        int greater = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (mat.get(i, j) < maxValue.mat.get(i, j)) {
                    greater++;
                }
            }
        }
        return greater > cols * rows / 2;
    }

    @Override
    protected boolean greaterThan(TensorValue maxValue) {
        throw new ArithmeticException("Algebbraic operation between Tensor and Matrix are not implemented yet");
    }

    @Override
    public boolean equals(Value obj) {
        if (obj instanceof MatrixValue) {
            if (Arrays.equals(mat.data, ((MatrixValue) obj).mat.data)) {
                return true;
            }
        }
        return false;
    }
}
