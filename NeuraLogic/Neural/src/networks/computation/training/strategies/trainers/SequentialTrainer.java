package networks.computation.training.strategies.trainers;

import networks.computation.iteration.actions.Evaluation;
import networks.computation.evaluation.results.Result;
import networks.computation.iteration.visitors.states.Dropouter;
import networks.computation.iteration.visitors.states.Invalidator;
import networks.computation.iteration.actions.Backpropagation;
import networks.computation.iteration.actions.IndependentNeuronProcessing;
import networks.computation.training.NeuralModel;
import networks.computation.training.NeuralSample;
import settings.Settings;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Created by gusta on 8.3.17.
 * <p>
 * Simple training on a single example one-by-one. Contains subclasses for sequential training from an input List but also a Strem of samples!
 */
public class SequentialTrainer extends Trainer {

    IndependentNeuronProcessing dropout;
    IndependentNeuronProcessing invalidation;
    Evaluation evaluation;
    Backpropagation backpropagation;

    public SequentialTrainer(Settings settings, NeuralModel neuralModel) {
        this(settings, neuralModel, -1);
    }

    public SequentialTrainer(Settings settings, NeuralModel neuralModel, int index) {
        evaluation = new Evaluation(settings, index);
        backpropagation = new Backpropagation(settings, neuralModel, index);
        invalidation = new IndependentNeuronProcessing(settings, new Invalidator(index));
        dropout = new IndependentNeuronProcessing(settings, new Dropouter(settings, index));
    }

    protected SequentialTrainer() {
    }

    public class SequentialListTrainer extends SequentialTrainer implements ListTrainer {

        @Override
        public List<Result> learnEpoch(NeuralModel neuralModel, List<NeuralSample> sampleList) {
            List<Result> resultList = new ArrayList<>(sampleList.size());
            for (NeuralSample neuralSample : sampleList) {
                Result result = learnFromSample(neuralModel, neuralSample, dropout, invalidation, evaluation, backpropagation);
                resultList.add(result);
            }
            return resultList;
        }
    }

    public class SequentialStreamTrainer extends SequentialTrainer implements StreamTrainer {

        @Override
        public Stream<Result> learnEpoch(NeuralModel neuralModel, Stream<NeuralSample> sampleStream) {
            Stream<Result> resultStream = sampleStream.map(sample -> learnFromSample(neuralModel, sample, dropout, invalidation, evaluation, backpropagation));
            return resultStream;
        }
    }
}