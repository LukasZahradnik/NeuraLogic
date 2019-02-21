package networks.structure.building.builders;

import constructs.example.ValuedFact;
import constructs.template.components.BodyAtom;
import constructs.template.components.WeightedRule;
import ida.ilp.logic.Literal;
import networks.structure.building.NeuronMaps;
import networks.structure.components.neurons.BaseNeuron;
import networks.structure.components.neurons.WeightedNeuron;
import networks.structure.components.neurons.types.*;
import networks.structure.components.types.DetailedNetwork;
import networks.structure.components.types.TopologicNetwork;
import networks.structure.components.weights.Weight;
import networks.structure.metadata.inputMappings.LinkedMapping;
import networks.structure.metadata.inputMappings.NeuronMapping;
import networks.structure.metadata.inputMappings.WeightedNeuronMapping;
import networks.structure.metadata.states.State;
import org.jetbrains.annotations.NotNull;
import settings.Settings;

import java.util.*;
import java.util.logging.Logger;

public class NeuralNetBuilder {
    private static final Logger LOG = Logger.getLogger(NeuralNetBuilder.class.getName());

    /**
     * The single point of reference for creating anything neural from logic parts
     */
    public NeuralBuilder neuralBuilder;
    private Settings settings;

    public NeuralNetBuilder(Settings settings, NeuralBuilder neuralBuilder) {
        this.neuralBuilder = neuralBuilder;
        this.settings = settings;
    }

    public NeuralNetBuilder(Settings settings) {
        this.neuralBuilder = new NeuralBuilder();
        this.settings = settings;
    }

    /**
     * Given a literal together with all corresponding rules where it appears as a head, create AtomNeuron -> AggNeurons -> RuleNeurons neurons and their connections.
     * Reuse existing neuronMaps, i.e., detect if a required neuron already exists in these,
     * and if so, then we need to add extra input mapping not to change the previously existing neurons, while still being able to reuse them.
     * <p>
     *
     * @param head  a head of all the subsequent rules
     * @param rules all the rules with all the groundings where it appears as a head
     */
    public void loadNeuronsFromRules(Literal head, LinkedHashMap<WeightedRule, LinkedHashSet<WeightedRule>> rules) {
        NeuronMaps neuronMaps = neuralBuilder.neuronFactory.neuronMaps;

        boolean newAtomNeuron = false;
        AtomNeuron headAtomNeuron;

        //1) head AtomNeuron creation
        if ((headAtomNeuron = neuronMaps.atomNeurons.get(head)) == null) {
            newAtomNeuron = true;
            headAtomNeuron = neuralBuilder.neuronFactory.createAtomNeuron(rules.entrySet().iterator().next().getValue().iterator().next().head); //it doesn't matter which rule's head (they are all the same)
        } else {
            headAtomNeuron.isShared = true;
            if (rules.entrySet().size() > 0) {  //if there are NEW rules for this headAtomNeuron to be processed, it means that we need to change its inputs in context of this new network! todo check for normal mode if really only the new rules are passed in
                WeightedNeuronMapping<AggregationNeuron> inputMapping;
                if ((inputMapping = (WeightedNeuronMapping<AggregationNeuron>) neuronMaps.extraInputMapping.get(headAtomNeuron)) != null) {    //if previously existing atom neuron already had input overmapping, create a new (incremental) one
                    neuronMaps.extraInputMapping.put(headAtomNeuron, new WeightedNeuronMapping<>(inputMapping));
                } else {
                    neuronMaps.extraInputMapping.put(headAtomNeuron, new WeightedNeuronMapping<>(headAtomNeuron.getInputs(), headAtomNeuron.getWeights()));
                }
            }
        }
        //2) AggregationNeurons creation
        for (Map.Entry<WeightedRule, LinkedHashSet<WeightedRule>> rules2groundings : rules.entrySet()) {
            boolean newAggNeuron = false;
            AggregationNeuron aggNeuron;

            if ((aggNeuron = neuronMaps.aggNeurons.get(rules2groundings.getKey())) == null) {
                newAggNeuron = true;
                aggNeuron = neuralBuilder.neuronFactory.createAggNeuron(rules2groundings.getKey());
            } else {
                aggNeuron.isShared = true;
                if (rules2groundings.getValue().size() > 0) {   //todo check
                    NeuronMapping<RuleNeuron> inputMapping;
                    if ((inputMapping = (NeuronMapping<RuleNeuron>) neuronMaps.extraInputMapping.get(aggNeuron)) != null) {    //if previously existing aggregation neuron already had input overmapping, create a new (incremental) one
                        neuronMaps.extraInputMapping.put(aggNeuron, new NeuronMapping<>(inputMapping));
                    } else {
                        neuronMaps.extraInputMapping.put(aggNeuron, new NeuronMapping<>(aggNeuron.getInputs()));
                    }
                }
            }
            if (newAtomNeuron) {
                headAtomNeuron.addInput(aggNeuron, rules2groundings.getKey().weight);
            } else {
                LOG.info("Warning-  modifying previous state - Creating input overmapping for this Atom neuron: " + headAtomNeuron);
                WeightedNeuronMapping<AggregationNeuron> inputMapping = (WeightedNeuronMapping<AggregationNeuron>) neuronMaps.extraInputMapping.get(headAtomNeuron);
                inputMapping.addLink(aggNeuron);
                inputMapping.addWeight(rules2groundings.getKey().weight);
            }
            //3) RuleNeurons creation
            for (WeightedRule grounding : rules2groundings.getValue()) {
                RuleNeurons ruleNeuron;

                if ((ruleNeuron = neuronMaps.ruleNeurons.get(grounding)) == null) {
                    if (grounding.hasWeightedBody()) {
                        ruleNeuron = neuralBuilder.neuronFactory.createWeightedRuleNeuron(grounding);
                    } else {
                        ruleNeuron = neuralBuilder.neuronFactory.createRuleNeuron(grounding);
                    }
                } else {
                    //ruleNeuron.isShared = true;
                    LOG.severe("Inconsistency - Specific rule neuron already contained in neuronmap!! This should never happen...");
                }
                if (newAggNeuron) {
                    aggNeuron.addInput(ruleNeuron);
                } else {
                    LOG.info("Warning-  modifying previous state - Creating input overmapping for this Agg neuron: " + aggNeuron);
                    NeuronMapping<RuleNeurons> inputMapping = (NeuronMapping<RuleNeurons>) neuronMaps.extraInputMapping.get(headAtomNeuron);
                    inputMapping.addLink(ruleNeuron);
                }
            }
        }
    }

    /**
     * Create FactNeurons mapped back to ground Literals
     * <p>
     * Remove fact neurons from neuronmaps that are never used? (probably unnecessary, they wont appear in the final network anyway)
     * No, keep them, as they are a valid part of the network (we can possibly query them).
     *
     * @param groundFacts
     * @return
     */
    public void loadNeuronsFromFacts(Map<Literal, ValuedFact> groundFacts) {
        for (Map.Entry<Literal, ValuedFact> factEntry : groundFacts.entrySet()) {
            neuralBuilder.neuronFactory.createFactNeuron(factEntry.getValue());
        }
    }

    /**
     * Given all existing neurons (either newly created or reused), connect RuleNeurons -> AtomNeurons (or FactNeurons).
     * Only newly created rule neurons, i.e. those that haven't been connected to their inputs yet, are processed.
     *
     * @return
     */
    @NotNull
    public void connectAllNeurons() {
        NeuronMaps neuronMaps = neuralBuilder.neuronFactory.neuronMaps;

        for (Map.Entry<WeightedRule, RuleNeurons> entry : neuronMaps.ruleNeurons.entrySet()) {
            RuleNeurons ruleNeuron = entry.getValue();
            if (ruleNeuron.inputCount() == entry.getKey().body.size()) {
                continue;   //this rule neuron is already connected (was created and taken from previous sample), connect only the newly created RuleNeurons
            }
            for (BodyAtom bodyAtom : entry.getKey().body) {
                Weight weight = bodyAtom.getConjunctWeight();

                AtomFact input = neuronMaps.atomNeurons.get(bodyAtom.getLiteral()); //input is an atom neuron?
                if (input == null) { //input is a fact neuron!
                    FactNeuron factNeuron = neuronMaps.factNeurons.get(bodyAtom.getLiteral());
                    if (factNeuron == null) {
                        LOG.severe("Error: no input found for this neuron!!: " + bodyAtom);
                    } else {
                        factNeuron.isShared = true;
                    }
                    input = factNeuron;
                    //weight = new Weight(new ScalarValue(settings.aggNeuronInputWeight));
                }
                if (bodyAtom.isNegated()) {
                    NegationNeuron negationNeuron = neuralBuilder.neuronFactory.createNegationNeuron(input, bodyAtom.getNegationActivation());
                    input = negationNeuron;
                }
                if (ruleNeuron instanceof WeightedNeuron) {
                    ((WeightedNeuron) ruleNeuron).addInput(input, weight);
                } else {
                    ((RuleNeuron) ruleNeuron).addInput(input);
                }
            }
        }
    }

    /**
     * This is only meant to go through the most necessary postprocessing steps to make for a valid neural network.
     * For the more advanced postprocessing optimization there is a whole configurable pipeline in {@link pipelines.building.NeuralNetsBuilder}
     *
     * @param id
     * @return
     */
    public DetailedNetwork finalizeStoredNetwork(String id) {
        DetailedNetwork neuralNetwork = neuralBuilder.networkFactory.createDetailedNetwork(neuralBuilder.neuronFactory.neuronMaps, id);

        StatesBuilder statesBuilder = neuralBuilder.statesBuilder;
        //fill all the states with correct dimension values
        statesBuilder.inferValues(neuralNetwork);

        if (settings.dropoutRate > 0) {
            statesBuilder.setupDropoutStates(neuralNetwork);  //setup individual dropout rates for each neuron
        }

        //if there are input overmappings, create appropriate states for them to be later stored in neural cache
        if (!neuralNetwork.extraInputMapping.isEmpty()) {
            statesBuilder.addLinkedInputsToNetworkStates(neuralNetwork);
        }

        //if there is the need, check parentCounts and store them by the network if needed
        if (settings.parentCounting) {
            neuralNetwork.outputMapping = calculateOutputs(neuralNetwork);
            statesBuilder.setupParentStateNumbers(neuralNetwork);
        }

        int sharedNeuronsCount = statesBuilder.makeSharedStatesRecursively(neuralNetwork);

        neuralNetwork.setSharedNeuronsCount(sharedNeuronsCount);

        return neuralNetwork;
    }


    public Map<BaseNeuron, LinkedMapping> calculateOutputs(TopologicNetwork<State.Neural.Structure> network) {
        Map<BaseNeuron, LinkedMapping> outputMapping = new HashMap<>();

        for (BaseNeuron parent : network.allNeuronsTopologic) {
            Iterator<BaseNeuron> inputs = network.getInputs(parent);
            BaseNeuron child;
            while ((child = inputs.next()) != null) {
                LinkedMapping parentMapping = outputMapping.computeIfAbsent(child, f -> new NeuronMapping());
                parentMapping.addLink(child);
            }
        }
        return outputMapping;
    }


    public NeuronMaps getNeuronMaps() {
        return neuralBuilder.neuronFactory.neuronMaps;
    }

    public void setNeuronMaps(NeuronMaps neuronMaps) {
        this.neuralBuilder.neuronFactory.neuronMaps = neuronMaps;
    }
}