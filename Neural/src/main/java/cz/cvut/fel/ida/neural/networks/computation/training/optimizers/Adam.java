package cz.cvut.fel.ida.neural.networks.computation.training.optimizers;

import cz.cvut.fel.ida.algebra.values.*;
import cz.cvut.fel.ida.algebra.weights.Weight;
import cz.cvut.fel.ida.setup.Settings;

import java.util.Collection;
import java.util.logging.Logger;

public class Adam implements Optimizer {
    private static final Logger LOG = Logger.getLogger(Adam.class.getName());

    public Value learningRate;
    public final double beta1;
    public final double beta2;
    public final double epsilon;

    public Adam(Value learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    public Adam(Value learningRate, double i_beta1, double i_beta2, double i_epsilon) {
        this.learningRate = learningRate;
        this.beta1 = i_beta1;
        this.beta2 = i_beta2;
        this.epsilon = i_epsilon;
    }

    public void performGradientStep(Collection<Weight> updatedWeights, Value[] gradients, int iteration) {
        //correction
        final double fix1 = 1 / (1 - Math.pow(beta1, iteration));
        final double fix2 = 1 / (1 - Math.pow(beta2, iteration));
        final double lr = ((ScalarValue) learningRate).value;

        for (Weight weight : updatedWeights) {
            if (weight.value instanceof ScalarValue) {
                double velocity = ((ScalarValue) weight.velocity).value;
                double momentum = ((ScalarValue) weight.momentum).value;
                double gradient = ((ScalarValue) gradients[weight.index]).value;

                momentum = (momentum * beta1) - (gradient * (1 - beta1));
                velocity = (velocity * beta2) + (gradient * gradient * (1 - beta2));

                ((ScalarValue) weight.momentum).value = momentum;
                ((ScalarValue) weight.velocity).value = velocity;
                ((ScalarValue) weight.value).value += momentum * fix1 * (-1 / (Math.sqrt(velocity * fix2) + epsilon)) * lr;

                continue;
            }

            double[] value, momentum, velocity, gradient;

            if (weight.value instanceof VectorValue) {
                value = ((VectorValue) weight.value).values;
                momentum = ((VectorValue) weight.momentum).values;
                velocity = ((VectorValue) weight.velocity).values;
                gradient = ((VectorValue) gradients[weight.index]).values;
            } else if (weight.value instanceof MatrixValue) {
                value = ((MatrixValue) weight.value).values;
                momentum = ((MatrixValue) weight.momentum).values;
                velocity = ((MatrixValue) weight.velocity).values;
                gradient = ((MatrixValue) gradients[weight.index]).values;
            } else {
                continue; // Maybe throw?
            }

            for (int i = 0; i < value.length; i++) {
                momentum[i] = (momentum[i] * beta1) - (gradient[i] * (1 - beta1));
                velocity[i] = (velocity[i] * beta2) + (gradient[i] * gradient[i] * (1 - beta2));
                value[i] += momentum[i] * fix1 * (-1 / (Math.sqrt(velocity[i] * fix2) + epsilon)) * lr;
            }
        }

    }

    @Override
    public void restart(Settings settings) {

    }
}
