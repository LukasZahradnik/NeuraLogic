package networks.structure.transforming;

import networks.structure.NeuralNetwork;

/**
 * Created by gusta on 14.3.17.
 */
public class IsoValueNetworkCompressor extends NetworkReducing implements NetworkMerging {
    @Override
    public NeuralNetwork merge(NeuralNetwork a, NeuralNetwork b) {
        return null;
    }

    @Override
    public NeuralNetwork reduce(NeuralNetwork inet) {
        return null;
    }
}
