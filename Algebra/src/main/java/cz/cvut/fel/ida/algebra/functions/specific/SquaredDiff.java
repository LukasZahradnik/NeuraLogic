package cz.cvut.fel.ida.algebra.functions.specific;

import cz.cvut.fel.ida.algebra.functions.ErrorFcn;
import cz.cvut.fel.ida.algebra.values.ScalarValue;
import cz.cvut.fel.ida.algebra.values.Value;

import java.util.logging.Logger;

public class SquaredDiff implements ErrorFcn {
    private static final Logger LOG = Logger.getLogger(SquaredDiff.class.getName());

    static Value oneHalf = new ScalarValue(0.5);

    public static SquaredDiff singleton = new SquaredDiff();

    @Override
    public Value evaluate(Value output, Value target) {
        if (output.getClass() != target.getClass()){
            LOG.severe("Prediction output and target label are of different algebraic types! (e.g. scalar vs vector)");
        }
        Value diff = output.minus(target);

        double accumulator = 0d;
        int elements = 0;

        for (double value : diff) {
            accumulator += value * value;
            elements += 1;
        }

        return new ScalarValue(accumulator / elements);
    }

    @Override
    public Value differentiate(Value output, Value target)   {
        return target.minus(output);
    }

    @Override
    public SquaredDiff getSingleton() {
        return singleton;
    }
}
