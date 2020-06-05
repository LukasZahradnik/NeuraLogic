package cz.cvut.fel.ida.algebra.values;

import cz.cvut.fel.ida.utils.generic.TestAnnotations;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class ValueTest {

    @TestAnnotations.Fast
    public void compareTo() {
        Value small = new ScalarValue(-3);
        ScalarValue big = new ScalarValue(10);

        int compareTo = small.compareTo(big);
        assertEquals(compareTo, -1);
    }

    @TestAnnotations.Fast
    public void compareToSame() {
        Value small = new ScalarValue(10);
        ScalarValue big = new ScalarValue(10);

        int compareTo = small.compareTo(big);
        assertEquals(compareTo, 0);
    }

    @TestAnnotations.Fast
    public void compareToAll() {
        Value small = new ScalarValue(10);
        VectorValue big = new VectorValue(Arrays.asList(1.0, 2.0, 3.0));

        int compareTo = small.compareTo(big);
        assertEquals(compareTo, 1);
    }

    @TestAnnotations.Fast
    public void compareToAllsmaller() {
        Value small = new ScalarValue(0.5);
        VectorValue big = new VectorValue(Arrays.asList(1.0, 2.0, 3.0));

        int compareTo = small.compareTo(big);
        assertEquals(compareTo, -1);
    }

    @TestAnnotations.Fast
    public void compareToAllsmallerOposite() {
        Value small = new ScalarValue(0.5);
        VectorValue big = new VectorValue(Arrays.asList(1.0, 2.0, 3.0));

        int compareTo = big.compareTo(small);
        assertEquals(compareTo, 1);
    }

    @TestAnnotations.Fast
    public void compareToAllMixed() {
        Value small = new ScalarValue(2.0);
        VectorValue big = new VectorValue(Arrays.asList(1.0, 2.0, 3.0));

        int compareTo = big.compareTo(small);
        assertEquals(0, compareTo);
    }

    @TestAnnotations.Fast
    public void equals() {
        Value small = new ScalarValue(10);
        ScalarValue big = new ScalarValue(10);

        boolean compareTo = small.equals(big);
        assertTrue(compareTo);
    }

    @TestAnnotations.Fast
    public void equalsNot() {
        Value small = new ScalarValue(10);
        ScalarValue big = new ScalarValue(2);

        boolean compareTo = small.equals(big);
        assertFalse(compareTo);
    }

    @TestAnnotations.Fast
    public void equalsWrong() {
        Value small = new ScalarValue(10);
        VectorValue big = new VectorValue(Arrays.asList(1.0, 2.0, 3.0));

        boolean compareTo = small.equals(big);
        assertFalse(compareTo);
    }

    @TestAnnotations.Fast
    public void kroneckerVectorVector1() {
        Value a = new VectorValue(Arrays.asList(1.0, 2.0));
        Value b = new VectorValue(Arrays.asList(1.0, 2.0, 3.0));

        Value kroneckerTimes = a.kroneckerTimes(b);
        assertEquals(new VectorValue(new double[]{1.0, 2.0, 3.0, 2, 4, 6}, false), kroneckerTimes);
    }

    @TestAnnotations.Fast
    public void kroneckerVectorVector2() {
        Value a = new VectorValue(new double[]{1.0, 2.0}, true);
        Value b = new VectorValue(new double[]{1.0, 2.0, 3.0}, true);

        Value kroneckerTimes = a.kroneckerTimes(b);
        assertEquals(new VectorValue(new double[]{1.0, 2.0, 3.0, 2, 4, 6}, true), kroneckerTimes);
    }

    @TestAnnotations.Fast
    public void kroneckerVectorVector3() {
        Value a = new VectorValue(new double[]{1.0, 2.0}, true);
        Value b = new VectorValue(new double[]{1.0, 2.0, 3.0}, false);

        Value kroneckerTimes = a.kroneckerTimes(b);
        boolean equals = new MatrixValue(new double[][]{{1.0, 2.0}, {2.0, 4}, {3, 6}}).equals(kroneckerTimes);
        assertTrue(equals);
    }

    @TestAnnotations.Fast
    public void kroneckerVectorVector4() {
        Value a = new VectorValue(new double[]{1.0, 2.0}, false);
        Value b = new VectorValue(new double[]{1.0, 2.0, 3.0}, true);

        Value kroneckerTimes = a.kroneckerTimes(b);
        boolean equals = new MatrixValue(new double[][]{{1.0, 2.0}, {2.0, 4}, {3, 6}}).transposedView().equals(kroneckerTimes);
        assertTrue(equals);
    }

    @TestAnnotations.Fast
    public void kroneckerVectorMatrix1() {
        Value a = new VectorValue(new double[]{1.0, 2.0}, false);
        Value b = new MatrixValue(new double[][]{{1.0, 2.0}, {3.0, 4.0}});

        Value kroneckerTimes = a.kroneckerTimes(b);
        boolean equals = new MatrixValue(new double[][]{{1.0, 2.0}, {3, 4}, {2, 4}, {6, 8}}).equals(kroneckerTimes);
        assertTrue(equals);
    }

    @TestAnnotations.Fast
    public void kroneckerVectorMatrix2() {
        Value a = new VectorValue(new double[]{1.0, 2.0, 3.0}, true);
        Value b = new MatrixValue(new double[][]{{1.0, 2.0}, {3.0, 4.0}});

        Value kroneckerTimes = a.kroneckerTimes(b);
        boolean equals = new MatrixValue(new double[][]{{1.0, 2.0, 2, 4, 3, 6}, {3, 4, 6, 8, 9, 12}}).equals(kroneckerTimes);
        assertTrue(equals);
    }

    @TestAnnotations.Fast
    public void kroneckerMatrixVector1() {
        Value a = new VectorValue(new double[]{1.0, 2.0, 3.0}, true);
        Value b = new MatrixValue(new double[][]{{1.0, 2.0}, {3.0, 4.0}});

        Value kroneckerTimes = b.kroneckerTimes(a);
        boolean equals = new MatrixValue(new double[][]{{1.0, 2.0, 3, 2, 4, 6}, {3, 6, 9, 4, 8, 12}}).equals(kroneckerTimes);
        assertTrue(equals);
    }

    @TestAnnotations.Fast
    public void kroneckerMatrixVector2() {
        Value a = new VectorValue(new double[]{1.0, 2.0, 3.0}, false);
        Value b = new MatrixValue(new double[][]{{1.0, 2.0}, {3.0, 4.0}});

        Value kroneckerTimes = b.kroneckerTimes(a);
        boolean equals = new MatrixValue(new double[][]{{1.0, 2.0}, {2, 4}, {3, 6}, {3, 4}, {6, 8}, {9, 12}}).equals(kroneckerTimes);
        assertTrue(equals);
    }

    @TestAnnotations.Fast
    public void kroneckerMatrixMatrix1() {
        Value a = new MatrixValue(new double[][]{{1.0, 2.0, 3.0}, {3.0, 4.0, 5.0}});
        Value b = new MatrixValue(new double[][]{{1.0, 2.0}, {3.0, 4.0}});

        Value kroneckerTimes = a.kroneckerTimes(b);
        boolean equals = new MatrixValue(new double[][]{{1.0, 2.0, 2.0, 4.0, 3.0, 6.0}, {3.0, 4.0, 6.0, 8.0, 9.0, 12.0}, {3.0, 6.0, 4.0, 8.0, 5.0, 10.0}, {9.0, 12.0, 12.0, 16.0, 15.0, 20.0}}).equals(kroneckerTimes);
        assertTrue(equals);
    }

}