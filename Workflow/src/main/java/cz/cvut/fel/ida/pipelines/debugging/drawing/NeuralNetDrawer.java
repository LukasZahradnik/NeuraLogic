package cz.cvut.fel.ida.pipelines.debugging.drawing;

import cz.cvut.fel.ida.algebra.functions.Combination;
import cz.cvut.fel.ida.algebra.functions.Transformation;
import cz.cvut.fel.ida.algebra.values.Value;
import cz.cvut.fel.ida.algebra.weights.Weight;
import cz.cvut.fel.ida.drawing.Drawer;
import cz.cvut.fel.ida.drawing.GraphViz;
import cz.cvut.fel.ida.neural.networks.computation.iteration.modes.Topologic;
import cz.cvut.fel.ida.neural.networks.computation.iteration.visitors.neurons.NeuronVisitor;
import cz.cvut.fel.ida.neural.networks.computation.iteration.visitors.states.neurons.Evaluator;
import cz.cvut.fel.ida.neural.networks.computation.training.NeuralSample;
import cz.cvut.fel.ida.neural.networks.structure.components.NeuralNetwork;
import cz.cvut.fel.ida.neural.networks.structure.components.neurons.BaseNeuron;
import cz.cvut.fel.ida.neural.networks.structure.components.neurons.Neurons;
import cz.cvut.fel.ida.neural.networks.structure.components.neurons.WeightedNeuron;
import cz.cvut.fel.ida.neural.networks.structure.components.neurons.states.State;
import cz.cvut.fel.ida.neural.networks.structure.components.neurons.types.*;
import cz.cvut.fel.ida.neural.networks.structure.components.types.TopologicNetwork;
import cz.cvut.fel.ida.setup.Settings;
import cz.cvut.fel.ida.utils.generic.Pair;

import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Created by gusta on 8.3.17.
 */
public class NeuralNetDrawer extends Drawer<NeuralSample> {
    private static final Logger LOG = Logger.getLogger(NeuralNetDrawer.class.getName());

    private NeuronDrawer neuronDrawer;
    private Topologic.BUpIterator bUpIterator;
    public int stateIndex = -1; //which state (=values if minibatches) to draw? (-1 = default)

    public NeuralNetDrawer(Settings settings) {
        super(settings);
    }

    @Override
    public void loadGraph(NeuralSample sample) {
        this.neuronDrawer = new NeuronDrawer(sample.query.evidence, stateIndex, graphviz);
        this.bUpIterator = new Topologic((TopologicNetwork<State.Neural.Structure>) sample.query.evidence).new BUpIterator(sample.query.neuron, neuronDrawer);

        startStyledGraph();
        iterateNetwork();
        if (sample.query.neuron != null) {
            Map<String, String> targetAttributes = new LinkedHashMap<>();
            targetAttributes.put("xlabel", "\n\n\ntarget = " + sample.target + "  ");
            graphviz.addln(dot.node(sample.query.neuron.getIndex(), "", "targetNeuron", targetAttributes));
        }
        graphviz.end_graph();
    }

    private void iterateNetwork() {
        while (this.bUpIterator.hasNext()) {
            BaseNeuron<Neurons, State.Neural> nextNeuron = this.bUpIterator.next();
            nextNeuron.visit((NeuronVisitor.Weighted.Detailed) neuronDrawer);
        }
    }

    public class NeuronDrawer extends NeuronVisitor.Weighted.Detailed {
        private final GraphViz gv;
        private final NeuralNetwork<State.Structure> neuralNetwork;

        public NeuronDrawer(NeuralNetwork<State.Structure> network, int stateIndex, GraphViz gv) {
            super(network, new Evaluator(stateIndex), null);
            this.gv = gv;
            this.neuralNetwork = network;
        }

        private String getNeuronLabel(BaseNeuron neuron) {
            boolean detailed = numberFormat != null && style.displayValues;
            String typeName = detailed ? neuron.getClass().getSimpleName() : "n";
            String name = typeName + ":" + neuron.index + ":" + neuron.name;
            State.Neural.Computation state = neuron.getComputationView(stateVisitor.stateIndex);

            Combination neuronCombination = null;
            Transformation neuronTransformation = null;
            if (!(neuron instanceof FactNeuron)) {
                neuronCombination = neuron.getCombination();
                neuronTransformation = neuron.getTransformation();
            }

            StringBuilder sb = new StringBuilder();
            sb.append(name);
            if (detailed && style.displayValues) {
                sb.append("\nval: ").append(state.getValue().toString(numberFormat));
            }
            if (detailed && style.displayGradients) {
                Value stateGradient = state.getGradient();
                if (stateGradient != null) {
                    sb.append("\ngrad: ").append(stateGradient.toString(numberFormat));
                }
            }
            if (detailed && style.displayDimensions) {
                sb.append("\ndim: ").append(Arrays.toString(state.getValue().size()));
            }
            if (detailed && style.displayFunctions) {
                String fcns = "";
                if (neuronCombination != null)
                    fcns += neuronCombination.getName();
                if (neuronTransformation != null)
                    fcns += "+" + neuronTransformation.getName();
                sb.append("\nfcn: ").append(fcns);
            }
            return sb.toString();
        }

        private String getEdgeLabel(Weight weight) {
            if (!style.displayValues || !style.displayWeights) {
                return "";
            }
            return weight.name + ":" + weight.value.toString(numberFormat);
        }

        private <T extends Neurons, S extends State.Neural> String getEdges(BaseNeuron<T, S> neuron) {
            Iterator<T> inputs = neuralNetwork.getInputs(neuron);
            StringBuilder sb = new StringBuilder();
            while (inputs.hasNext()) {
                T input = inputs.next();
                sb.append(dot.edge(input.getIndex(), neuron.index, "unweightedInput", "")).append("\n");
            }
            return sb.toString();
        }

        private <T extends Neurons, S extends State.Neural> String getEdges(WeightedNeuron<T, S> neuron) {
            Pair<Iterator<T>, Iterator<Weight>> inputs = neuralNetwork.getInputs(neuron);
            Iterator<T> inputNeurons = inputs.r;
            Iterator<Weight> inputWeights = inputs.s;
            StringBuilder sb = new StringBuilder();
            while (inputNeurons.hasNext()) {
                T input = inputNeurons.next();
                Weight weight = inputWeights.next();
                String styleKey = weight.isFixed ? "fixedWeightInput" : "weightedInput";
                sb.append(dot.edge(input.getIndex(), neuron.index, styleKey, getEdgeLabel(weight))).append("\n");
            }
            return sb.toString();
        }

        @Override
        public <T extends Neurons, S extends State.Neural> void visit(BaseNeuron<T, S> neuron) {
            LOG.severe("DDD while drawing");
        }

        @Override
        public <T extends Neurons, S extends State.Neural> void visit(WeightedNeuron<T, S> neuron) {
            LOG.severe("DDD while drawing");
        }

        @Override
        public void visit(AtomNeuron neuron) {
            gv.addln(dot.node(neuron.index, getNeuronLabel(neuron), "atomNeuron"));
            gv.addln(getEdges(neuron));
        }

        @Override
        public void visit(NegationNeuron neuron) {
            gv.addln(dot.node(neuron.index, getNeuronLabel(neuron), "negationNeuron"));
            gv.addln(getEdges(neuron));
        }

        @Override
        public void visit(WeightedAtomNeuron neuron) {
            gv.addln(dot.node(neuron.index, getNeuronLabel(neuron), "weightedAtomNeuron"));
            gv.addln(getEdges(neuron));
        }

        @Override
        public void visit(AggregationNeuron neuron) {
            gv.addln(dot.node(neuron.index, getNeuronLabel(neuron), "aggregationNeuron"));
            gv.addln(getEdges(neuron));
        }

        @Override
        public void visit(RuleNeuron neuron) {
            gv.addln(dot.node(neuron.index, getNeuronLabel(neuron), "ruleNeuron"));
            gv.addln(getEdges(neuron));
        }

        @Override
        public void visit(WeightedRuleNeuron neuron) {
            gv.addln(dot.node(neuron.index, getNeuronLabel(neuron), "weightedRuleNeuron"));
            gv.addln(getEdges(neuron));
        }


        @Override
        public void visit(FactNeuron neuron) {
            gv.addln(dot.node(neuron.index, getNeuronLabel(neuron), "factNeuron"));
            gv.addln();
        }
    }
}
