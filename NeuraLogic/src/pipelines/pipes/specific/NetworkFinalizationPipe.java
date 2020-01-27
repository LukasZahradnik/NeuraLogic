package pipelines.pipes.specific;

import networks.structure.building.NeuralProcessingSample;
import networks.structure.building.builders.NeuralNetBuilder;
import networks.structure.components.neurons.BaseNeuron;
import networks.structure.components.neurons.Neurons;
import networks.structure.components.types.DetailedNetwork;
import networks.structure.metadata.states.State;
import networks.structure.transforming.NetworkReducing;
import networks.structure.transforming.ParentsExtractor;
import pipelines.Pipe;
import settings.Settings;

import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class NetworkFinalizationPipe extends Pipe<Stream<NeuralProcessingSample>, Stream<NeuralProcessingSample>> {
    private static final Logger LOG = Logger.getLogger(NetworkFinalizationPipe.class.getName());
    private final NeuralNetBuilder neuralNetBuilder;

    public NetworkFinalizationPipe(Settings settings, NeuralNetBuilder neuralNetBuilder) {
        super("NetworksFinalizationPipe", settings);
        this.neuralNetBuilder = neuralNetBuilder;
    }

    @Override
    public Stream<NeuralProcessingSample> apply(Stream<NeuralProcessingSample> neuralProcessingSampleStream) {

        //if we do Global (KB) mode with a single Shared network, and we require topologic iteration for each query, extract many small networks then
        if (settings.groundingMode == Settings.GroundingMode.GLOBAL && settings.iterationMode == Settings.IterationMode.TOPOLOGIC) {
            neuralProcessingSampleStream = neuralProcessingSampleStream.map(sample -> {
                DetailedNetwork emptyCopy = sample.detailedNetwork.emptyCopy(sample.getId());
                NetworkReducing.supervisedNetReconstruction(emptyCopy, (BaseNeuron<Neurons, State.Neural>) sample.query.neuron);
                sample.detailedNetwork = emptyCopy;
                sample.query.evidence = emptyCopy;
                return sample;
            });
        }

        //if we need to access the parentStates multiple times in sequence or at once, we need to correct for the parentscounts of shared neurons that did not know they are shared at time of creation
        if (settings.parentCounting && (settings.parallelTraining || !settings.neuralStreaming)) {
            ParentsExtractor parentsExtractor = new ParentsExtractor();
            //in the case of the need for parents extraction of shared neurons, we NEED TO TERMINATE THE STREAM
            List<NeuralProcessingSample> processingSamples = neuralProcessingSampleStream.collect(Collectors.toList());
            processingSamples.forEach(sample -> parentsExtractor.extractSharedNeuronsParents(sample.detailedNetwork));
            neuralProcessingSampleStream = processingSamples.stream();
        }

        //if we need to either store extra inputs, or store (possibly varying) parents for shared neurons, create a Neural Cache
        return neuralProcessingSampleStream.map(sample -> {
            sample.detailedNetwork = neuralNetBuilder.neuralBuilder.statesBuilder.setupFinalStatesCache(sample.detailedNetwork);
            return sample;
        });
    }
}
