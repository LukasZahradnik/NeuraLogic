package cz.cvut.fel.ida.neural.networks.computation.iteration;

import cz.cvut.fel.ida.neural.networks.computation.iteration.visitors.neurons.NeuronVisitor;
import cz.cvut.fel.ida.neural.networks.structure.components.NeuralNetwork;
import cz.cvut.fel.ida.neural.networks.structure.components.neurons.BaseNeuron;
import cz.cvut.fel.ida.neural.networks.structure.components.neurons.Neurons;
import cz.cvut.fel.ida.neural.networks.structure.components.neurons.states.State;

/**
 * Iteration strategy {@link IterationStrategy} based on the clean Iterator pattern. I.e. we are traversing the structure
 * of {@link NeuralNetwork} and returning the elements one-by-one, to be visited by {@link #neuronVisitor}.
 */
public abstract class NeuronIterating extends IterationStrategy implements NeuronIterator {

    /**
     * Takes care of aggregating/propagating values to neighbours
     */
    protected NeuronVisitor.Weighted neuronVisitor;

    public NeuronIterating(NeuralNetwork<State.Neural.Structure> network, Neurons outputNeuron, NeuronVisitor.Weighted pureNeuronVisitor) {
        super(network, outputNeuron);
        this.neuronVisitor = pureNeuronVisitor;
    }

    /**
     * A default implementation for a typical NeuronIterating. Can be called for both BUp&TDown directions,
     * since the directionality is taken care of by the next() method.
     */
    @Override
    public void iterate() {
        while (hasNext()){
            BaseNeuron<Neurons, State.Neural> nextNeuron = next();
            nextNeuron.visit(neuronVisitor);
        }
    }
}