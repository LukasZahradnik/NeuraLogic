package cz.cvut.fel.ida.neural.networks.computation.training.strategies.trainers;

import cz.cvut.fel.ida.learning.results.Result;
import cz.cvut.fel.ida.neural.networks.computation.training.NeuralModel;
import cz.cvut.fel.ida.neural.networks.computation.training.NeuralSample;
import cz.cvut.fel.ida.neural.networks.computation.training.optimizers.Optimizer;
import cz.cvut.fel.ida.neural.networks.computation.training.strategies.debugging.NeuralDebugging;
import cz.cvut.fel.ida.setup.Settings;

import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The experimental unsynchronized parallel minibatch/SGD with random gradient overwriting.
 * <p>
 * This is roughly like minibatch training where the batch size is app. the number of available threads/cores,
 * but it can vary dynamically as the samples to be processed in parallel are selected by the underlying {@link java.util.Spliterator} of the parallel stream.
 * <p>
 * The correct way of processing is to perform weight updates BEFORE any subsequent evaluation and next weight updates,
 * but in this case we do not wait until the weight update is finished, so the updates may overwrite each other.
 * But since there is no synchronization it should be fastest and it was shown to work in practice on our datasets.
 */
public class AsyncParallelTrainer extends SequentialTrainer {
    private static final Logger LOG = Logger.getLogger(AsyncParallelTrainer.class.getName());

    //todo correctly there should be as many trainers as there are samples, i.e. no two samples should be trained with the same trainer in parallel
    // otherwise they might be overwriting/mixing up their results in shared neurons, but this trainer is kind of flawed anyway

    public AsyncParallelTrainer(Settings settings, Optimizer optimizer, NeuralModel neuralModel) {
        super(settings, optimizer, neuralModel);
    }

    protected AsyncParallelTrainer() {
    }

    public class AsyncListTrainer implements ListTrainer {


        @Override
        public List<Result> learnEpoch(NeuralModel neuralModel, List<NeuralSample> sampleList) {
            List<Result> resultList = sampleList.parallelStream().
                    map(neuralSample -> learnFromSample(neuralModel, neuralSample, dropout, invalidation, evaluation, backpropagation)).collect(Collectors.toList());
            return resultList;
        }

        @Override
        public List<Result> evaluate(List<NeuralSample> sampleList) {
            List<Result> resultList = sampleList.parallelStream().
                    map(neuralSample -> evaluateSample(evaluation, neuralSample)).collect(Collectors.toList());
            return resultList;
        }

        @Override
        public void restart(Settings settings) {
            AsyncParallelTrainer.this.optimizer.restart(settings);
        }

        @Override
        public void setupDebugger(NeuralDebugging trainingDebugger) {
            neuralDebugger = trainingDebugger;
        }
    }

    public class AsyncStreamTrainer implements StreamTrainer {

        @Override
        public Stream<Result> learnEpoch(NeuralModel neuralModel, Stream<NeuralSample> sampleStream) {
            Stream<Result> resultStream = sampleStream.parallel().map(sample -> learnFromSample(neuralModel, sample, dropout, invalidation, evaluation, backpropagation));
            return resultStream;
        }

        @Override
        public void setupDebugger(NeuralDebugging trainingDebugger) {
            neuralDebugger = trainingDebugger;
        }
    }
}

