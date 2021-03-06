package cz.cvut.fel.ida.neural.networks.computation.iteration.visitors.states.neurons;

import cz.cvut.fel.ida.algebra.values.Value;
import cz.cvut.fel.ida.neural.networks.computation.iteration.visitors.states.StateVisiting;
import cz.cvut.fel.ida.neural.networks.structure.components.neurons.states.State;
import cz.cvut.fel.ida.setup.Settings;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Created by gusta on 8.3.17.
 */
public class Backproper extends StateVisiting.Computation {
    private static final Logger LOG = Logger.getLogger(Backproper.class.getName());

    public Backproper(int stateIndex) {
        super(stateIndex);
    }

    /**
     * Get possibly different StateVisitors of Backproper's type to manipulate Neurons' States
     *
     * @param settings
     * @param i
     * @return
     */
    public static Backproper getFrom(Settings settings, int i) {
        return new Backproper(i);   //todo base on settings
    }

    /**
     * Get multiple evaluators with different state access views/indices
     *
     * @param settings
     * @param count
     * @return
     */
    @Deprecated
    public static List<Backproper> getParallelEvaluators(Settings settings, int count) {
        List<Backproper> backpropers = new ArrayList<>(count);
        for (int i = 0; i < backpropers.size(); i++) {
            backpropers.add(i, new Backproper(i));
        }
        return backpropers;
    }

    @Override
    public Value visit(State.Neural.Computation state) {
        Value acumGradient = state.getGradient(); //top-down accumulation //todo test add check if non-zero and cut otherwise?
        Value inputFcnDerivative = state.getAggregationState().gradient(); //bottom-up accumulation

        Value currentLevelDerivative;
        if (acumGradient.getClass().equals(inputFcnDerivative.getClass())) {
            currentLevelDerivative = acumGradient.elementTimes(inputFcnDerivative);  //elementTimes here - since the fcn to be differentiated was applied element-wise on a vector
        } else {
            currentLevelDerivative = acumGradient.transposedView().times(inputFcnDerivative);  //times here - since the fcn was a complex vector function (e.g. softmax) and has a matrix derivative (Jacobian)
        }

        //there is no setting (remembering) of the calculated gradient (as opposed to output, which is reused), it is just returned
        return currentLevelDerivative;
    }
}