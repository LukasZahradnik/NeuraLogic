package cz.cvut.fel.ida.neural.networks.computation.training.strategies;

import cz.cvut.fel.ida.algebra.values.Value;
import cz.cvut.fel.ida.algebra.values.inits.ValueInitializer;
import cz.cvut.fel.ida.learning.results.DetailedClassificationResults;
import cz.cvut.fel.ida.learning.results.Progress;
import cz.cvut.fel.ida.algebra.values.ScalarValue;
import cz.cvut.fel.ida.learning.results.Result;
import cz.cvut.fel.ida.neural.networks.computation.iteration.actions.Accumulating;
import cz.cvut.fel.ida.neural.networks.computation.iteration.actions.Evaluation;
import cz.cvut.fel.ida.neural.networks.computation.iteration.visitors.states.neurons.SaturationChecker;
import cz.cvut.fel.ida.neural.networks.computation.iteration.visitors.weights.WeightUpdater;
import cz.cvut.fel.ida.neural.networks.computation.training.NeuralModel;
import cz.cvut.fel.ida.neural.networks.computation.training.NeuralSample;
import cz.cvut.fel.ida.neural.networks.computation.training.optimizers.Optimizer;
import cz.cvut.fel.ida.neural.networks.computation.training.strategies.Hyperparameters.LearnRateDecayStrategy;
import cz.cvut.fel.ida.neural.networks.computation.training.strategies.debugging.NeuralDebugging;
import cz.cvut.fel.ida.neural.networks.computation.training.strategies.trainers.*;
import cz.cvut.fel.ida.setup.Settings;
import cz.cvut.fel.ida.utils.exporting.Exporter;
import cz.cvut.fel.ida.utils.generic.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class PythonTrainingStrategy extends TrainingStrategy {

    transient List<NeuralSample> samplesSet;

    transient SequentialTrainer trainer;

    transient ListTrainer listTrainer;

    MiniBatchTrainer miniBatchTrainer;

    ListTrainer minibatchListTrainer;

    ValueInitializer valueInitializer;

    Evaluation evaluation;

    LearnRateDecayStrategy learnRateDecay;

    int epochCount = 0;

    public static final String METRIC_AUC_ROC = "auc_roc";
    public static final String METRIC_AUC_PR = "auc_pr";
    public static final String METRIC_BEST_ACCURACY = "best_accuracy";
    public static final String METRIC_BEST_THRESHOLD = "best_threshold";
    public static final String METRIC_SATURATION = "saturation";

    /**
     * The threshold picked from the last training set, passed to validation so
     * that the reported best accuracy is the same quantity IterativeTrainingStrategy
     * computes: validation accuracy at the training-chosen threshold.
     */
    transient Value lastBestThreshold = new ScalarValue(0.5);

    public PythonTrainingStrategy(Settings settings, NeuralModel model, Optimizer optimizer, LearnRateDecayStrategy learnRateDecay) {
        super(settings, model);

        this.trainer = new SequentialTrainer(settings, optimizer, currentModel);
        this.listTrainer = this.trainer.new SequentialListTrainer();
        this.valueInitializer = ValueInitializer.getInitializer(settings);
        this.evaluation = this.trainer.getEvaluation();

        this.miniBatchTrainer = new MiniBatchTrainer(settings, optimizer, currentModel, 0);
        this.minibatchListTrainer = this.miniBatchTrainer.new MinibatchListTrainer();

        this.learnRateDecay = learnRateDecay;
    }

    public SequentialTrainer getTrainer() {
        return this.trainer;
    }

    public NeuralModel getCurrentModel() {
        return this.currentModel;
    }

    public void setSamples(List<NeuralSample> samples) {
        this.samplesSet = samples;
    }

    public void resetParameters() {
        if (learnRateDecay != null) {
            learnRateDecay.restart();
        }

        epochCount = 0;
        listTrainer.restart(settings);
        currentModel.resetWeights(valueInitializer);
    }

    @Override
    public Pair<NeuralModel, Progress> train() {
        return null;
    }

    @Override
    public void setupDebugger(NeuralDebugging neuralDebugger) {
    }

    public List<Result> learnSamples(int epochs, int minibatchSize) {
        return learnSamples(samplesSet, epochs, minibatchSize);
    }

    public List<Result> learnSamples(List<NeuralSample> samples, int epochs, int minibatchSize) {
        List<Result> results = null;

        if (epochs <= 0) {
            return new ArrayList<>();
        }

        ListTrainer trainer = listTrainer;

        if (minibatchSize > 1) {
            miniBatchTrainer.setMinibatchSize(minibatchSize);
            trainer = minibatchListTrainer;
        }

        for (int i = 0; i < epochs; i++) {
            epochCount++;

            if (learnRateDecay != null) {
                learnRateDecay.decay(epochCount);
            }

            results = trainer.learnEpoch(currentModel, samples);

//            if (settings.checkNeuronSaturation) {
//                saturationCheck(samples);
//            }
        }

        return results;
    }

    /**
     * Trains exactly like {@link #learnSamples(List, int, int)} and additionally
     * computes the requested deep metrics over the final epoch's results. The
     * default path is untouched - call this only when metrics are wanted.
     */
    public PythonTrainingMetrics learnSamplesWithMetrics(List<NeuralSample> samples, int epochs, int minibatchSize, List<String> metrics) {
        List<Result> results = learnSamples(samples, epochs, minibatchSize);
        return new PythonTrainingMetrics(results, computeMetrics(samples, results, metrics, true));
    }

    /**
     * Evaluates exactly like {@link #validateSamples(List, int)} and additionally
     * computes the requested deep metrics. Validation best-accuracy is measured
     * at the threshold last picked on the training set, matching
     * IterativeTrainingStrategy's recalculateResults().
     */
    public PythonTrainingMetrics validateSamplesWithMetrics(List<NeuralSample> samples, int minibatchSize, List<String> metrics) {
        List<Result> results = validateSamples(samples, minibatchSize);
        return new PythonTrainingMetrics(results, computeMetrics(samples, results, metrics, false));
    }

    /**
     * Compute deep metrics from results that have already been produced (for
     * callers, e.g. the static-graph path, that run their own loop over samples).
     * The results list is never reordered here; classification metrics work on a
     * defensive copy because computeBestAccuracyThreshold sorts in place.
     */
    public Map<String, Double> computeMetrics(List<NeuralSample> samples, List<Result> results, List<String> metrics, boolean isTraining) {
        Map<String, Double> output = new HashMap<>();
        if (results == null || results.isEmpty() || metrics == null) {
            return output;
        }

        DetailedClassificationResults detailed = null;
        boolean bestAccuracyComputed = false;

        for (String metric : metrics) {
            if (metric == null) {
                continue;
            }
            switch (metric) {
                case METRIC_AUC_ROC:
                    if (detailed == null) {
                        detailed = classificationResults(results);
                    }
                    if (detailed.AUCroc != null) {
                        output.put(METRIC_AUC_ROC, detailed.AUCroc);
                    }
                    break;
                case METRIC_AUC_PR:
                    if (detailed == null) {
                        detailed = classificationResults(results);
                    }
                    if (detailed.getAUCpr() != null) {
                        output.put(METRIC_AUC_PR, detailed.getAUCpr());
                    }
                    break;
                case METRIC_BEST_ACCURACY:
                case METRIC_BEST_THRESHOLD:
                    if (detailed == null) {
                        detailed = classificationResults(results);
                    }
                    if (!bestAccuracyComputed) {
                        if (isTraining) {
                            detailed.computeBestAccuracyThreshold(detailed.evaluations);
                            lastBestThreshold = detailed.bestThreshold;
                        } else {
                            detailed.computeBestAccuracy(detailed.evaluations, lastBestThreshold);
                        }
                        bestAccuracyComputed = true;
                    }
                    if (METRIC_BEST_ACCURACY.equals(metric) && detailed.bestAccuracy != null) {
                        output.put(METRIC_BEST_ACCURACY, detailed.bestAccuracy);
                    }
                    if (METRIC_BEST_THRESHOLD.equals(metric) && detailed.bestThreshold != null) {
                        output.put(METRIC_BEST_THRESHOLD, ((ScalarValue) detailed.bestThreshold).value);
                    }
                    break;
                case METRIC_SATURATION:
                    output.put(METRIC_SATURATION, saturationRatio(samples));
                    break;
                default:
                    break;
            }
        }

        return output;
    }

    private DetailedClassificationResults classificationResults(List<Result> results) {
        return new DetailedClassificationResults(new ArrayList<>(results), settings);
    }

    private double saturationRatio(List<NeuralSample> samples) {
        if (samples == null || samples.isEmpty()) {
            return 0.0;
        }

        ScalarValue percentage = new ScalarValue(settings.saturationPercentage);
        Accumulating accumulating = new Accumulating(settings, new SaturationChecker());
        List<Pair<Value, Value>> pairs = accumulating.accumulateStats(samples);
        int saturatedNetworks = 0;

        for (Pair<Value, Value> pair : pairs) {
            Value saturated = pair.r;
            Value all = pair.s;
            if (saturated.greaterThan(all.times(percentage))) {
                saturatedNetworks++;
            }
        }

        return (double) saturatedNetworks / samples.size();
    }

    public static class PythonTrainingMetrics {
        private final List<Result> results;
        private final Map<String, Double> metrics;

        public PythonTrainingMetrics(List<Result> results, Map<String, Double> metrics) {
            this.results = results;
            this.metrics = metrics;
        }

        public List<Result> getResults() {
            return results;
        }

        public Map<String, Double> getMetrics() {
            return metrics;
        }
    }

    public Result learnSample(NeuralSample sample) {
        trainer.invalidateSample(trainer.getInvalidation(), sample);
        Result result = trainer.evaluateSample(trainer.getEvaluation(), sample);

        WeightUpdater weightUpdater = trainer.backpropSample(trainer.getBackpropagation(), result, sample);
        //this reimplements Trainer.learnFromSample rather than calling it, so anything that path gains has to
        //be repeated here or the two descend different functions - which they did, until the reduction was
        //added to only one of them and a single sample under MEAN stepped as if under SUM
        trainer.reduceAndClip(weightUpdater, Collections.singletonList(result));
        trainer.updateWeights(currentModel, weightUpdater);

        return result;
    }

    public Value evaluateSample(NeuralSample sample) {
        trainer.invalidateSample(trainer.getInvalidation(), sample);
        return evaluation.evaluate(sample.query);
    }

    public List<Value> evaluateSamples(List<NeuralSample> samples, int minibatchSize) {
        List<Value> output = new ArrayList<>(samples.size());

        if (minibatchSize > 1) {
            miniBatchTrainer.setMinibatchSize(minibatchSize);

            for (Result result : minibatchListTrainer.evaluate(samples)) {
                output.add(result.getOutput());
            }

            return output;
        }

        for (NeuralSample sample : samples) {
            trainer.invalidateSample(trainer.getInvalidation(), sample);
            output.add(evaluation.evaluate(sample.query));
        }

        return output;
    }

    /**
     * The loss of a whole dataset, reduced the way the error function says - so the single number torch's
     * criterion would hand back, and the quantity the optimizer is descending.
     * <p>
     * The per-sample values from {@link #validateSamples} are deliberately *not* reduced across the batch;
     * they are each summed over their own components, which is torch's reduction="none". This is the one
     * that applies the batch reduction, through the same {@link Result#reductionDivisor} the trainers scale
     * the gradient by, so the reported loss and the descended one cannot drift apart.
     */
    public Value reducedError(List<NeuralSample> samples, int minibatchSize) {
        List<Result> results = validateSamples(samples, minibatchSize);
        double total = 0;
        for (Result result : results) {
            for (double component : result.errorValue()) {
                total += component;
            }
        }
        return new ScalarValue(total / Result.reductionDivisor(results, settings.errorReduction));
    }

    public Result validateSample(NeuralSample sample) {
        trainer.invalidateSample(trainer.getInvalidation(), sample);
        return trainer.evaluateSample(evaluation, sample);
    }

    public List<Result> validateSamples(List<NeuralSample> samples, int minibatchSize) {
        List<Result> output = new ArrayList<>(samples.size());

        if (minibatchSize > 1) {
            miniBatchTrainer.setMinibatchSize(minibatchSize);

            for (Result result : minibatchListTrainer.evaluate(samples)) {
                output.add(result);
            }

            return output;
        }

        for (NeuralSample sample : samples) {
            trainer.invalidateSample(trainer.getInvalidation(), sample);
            output.add(trainer.evaluateSample(evaluation, sample));
        }

        return output;
    }

    @Override
    public void export(Exporter exporter) {
    }

    @Override
    public String exportToJson() {
        return null;
    }
}