package cz.cvut.fel.ida.algebra.functions.specific;

import cz.cvut.fel.ida.algebra.functions.Activation;
import cz.cvut.fel.ida.utils.generic.Pair;

import java.util.function.Function;
import java.util.logging.Logger;

/**
 * Created by gusta on 8.3.17.
 */
public class Tanh extends Activation {
    private static final Logger LOG = Logger.getLogger(Tanh.class.getName());

    private static final Function<Double, Double> tanh = Math::tanh;

    private static final Function<Double, Double> diffTanh = in -> {
        if (in > 100 || in < -100)
            return 0.0;
        double tanh1 = Math.tanh(in);
        return 1 - (tanh1 * tanh1);
    };

    public Tanh() {
        super(tanh, diffTanh);
    }

    @Override
    public Tanh replaceWithSingleton() {
        return Singletons.tanh;
    }

    @Override
    public Pair<Double, Double> getSaturationRange() {
        return new Pair<>(-0.99, 0.99);
    }
}