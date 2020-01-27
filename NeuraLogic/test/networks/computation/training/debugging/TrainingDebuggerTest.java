package networks.computation.training.debugging;

import org.junit.Test;
import settings.Settings;
import utils.logging.Logging;

import java.util.logging.Level;

public class TrainingDebuggerTest {

    @Test
    public void family() {
        Logging logging = Logging.initLogging(Level.FINE);
        String[] args = "-path ./resources/datasets/simple/family".split(" ");
        Settings settings = new Settings();
        settings.maxCumEpochCount = 2;
        settings.intermediateDebug = true;
        settings.undoWeightTrainingChanges = true;
        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void mutaMini() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = "-e ./resources/datasets/relational/molecules/muta_mini/examples -t ./resources/datasets/relational/molecules/muta_mini/template_old_init".split(" ");

        Settings settings = new Settings();
        settings.maxCumEpochCount = 1000;
        settings.oneQueryPerExample = true;
        settings.neuralNetsPostProcessing = false;
        settings.chainPruning = true;
        settings.isoValueCompression = true;
        settings.intermediateDebug = false;
        settings.debugPipeline = true;
        settings.storeNotShow = true;
        settings.debugTemplateTraining = false; //too big
        settings.debugTemplate = false;
        settings.optimizer = Settings.OptimizerSet.SGD;
        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void mutagen_zero_init() {   //todo now remove these weird templates and do some cleaning in the templates
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-e ./resources/datasets/relational/molecules/mutagenesis/trainExamples.txt " +
                "-q ./resources/datasets/relational/molecules/mutagenesis/trainQueries.txt " +
                "-t ./resources/datasets/relational/molecules/mutagenesis/template_old.txt").split(" ");

        Settings settings = new Settings();
        settings.seed = 0;
        settings.initLearningRate = 0.1;
        settings.maxCumEpochCount = 100;
        settings.resultsRecalculationEpochae = 20;
        settings.appLimitSamples = 10;
        settings.stratification = true;
        settings.oneQueryPerExample = true;
        settings.neuralNetsPostProcessing = true;
        settings.chainPruning = true;
        settings.isoValueCompression = true;
        settings.intermediateDebug = false;
        settings.debugPipeline = false;
        settings.storeNotShow = true;
        settings.debugTemplateTraining = false; //too big
        settings.debugTemplate = false;
        settings.optimizer = Settings.OptimizerSet.ADAM;
        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    /**
     * This setting works successfuly on template with no offsets with SGD after 10000 steps
     */
    @Test
    public final void mutagen_sgd_very_slow() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-e ./resources/datasets/relational/molecules/mutagenesis/trainExamples.txt " +
                "-q ./resources/datasets/relational/molecules/mutagenesis/trainQueries.txt " +
                "-t ./resources/datasets/relational/molecules/mutagenesis/template_new_fast.txt").split(" ");

        Settings settings = new Settings();

        settings.seed = 0;
        settings.initLearningRate = 0.05;
        settings.maxCumEpochCount = 10000;
        settings.resultsRecalculationEpochae = 100;
        settings.debugSampleOutputs = true;
//        settings.appLimitSamples = 100;
        settings.calculateBestThreshold = true;
        settings.stratification = true;
        settings.oneQueryPerExample = true;
        settings.neuralNetsPostProcessing = true;
        settings.chainPruning = true;
        settings.isoValueCompression = false;
        settings.intermediateDebug = false;
        settings.debugPipeline = false;
        settings.storeNotShow = true;
        settings.debugTemplateTraining = false; //too big
        settings.debugTemplate = false;
        settings.initializer = Settings.InitSet.SIMPLE;
        settings.optimizer = Settings.OptimizerSet.SGD;
        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void mutagen_fast() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-e ./resources/datasets/relational/molecules/mutagenesis/trainExamples.txt " +
                "-q ./resources/datasets/relational/molecules/mutagenesis/trainQueries.txt " +
                "-t ./resources/datasets/relational/molecules/mutagenesis/template_fastest.txt").split(" ");

        Settings settings = new Settings();

        settings.seed = 0;
        settings.initLearningRate = 0.3;
        settings.maxCumEpochCount = 3000;
        settings.resultsRecalculationEpochae = 10;
        settings.shuffleEachEpoch = false;
        settings.debugSampleOutputs = false;
//        settings.appLimitSamples = 100;
        settings.calculateBestThreshold = false;
        settings.initializer = Settings.InitSet.SIMPLE;
        settings.optimizer = Settings.OptimizerSet.SGD;


        settings.stratification = false;
        settings.oneQueryPerExample = true;
        settings.neuralNetsPostProcessing = true;
        settings.chainPruning = true;
        settings.isoValueCompression = false;
        settings.intermediateDebug = false;
        settings.debugPipeline = false;
        settings.storeNotShow = true;
        settings.debugTemplateTraining = false; //too big
        settings.debugTemplate = false;
        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void mutagen_diffcheck() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-e ./resources/datasets/relational/molecules/mutagenesis/examples3 " +
//                "-q ./resources/datasets/relational/molecules/mutagenesis/trainQueries.txt " +
                "-t ./resources/datasets/relational/molecules/mutagenesis/template_new.txt").split(" ");

        Settings settings = new Settings();

        settings.initDistribution = Settings.InitDistribution.CONSTANT;

        settings.seed = 0;  //the seed shouldn't matter here!
        settings.shuffleBeforeFoldSplit = false;
        settings.shuffleBeforeTraining = false;
        settings.shuffleEachEpoch = false;
        settings.debugSampleOutputs = true;

        settings.initLearningRate = 0.3;
        settings.maxCumEpochCount = 100;
        settings.resultsRecalculationEpochae = 1;
        settings.calculateBestThreshold = false;
//        settings.appLimitSamples = 100;
        settings.initializer = Settings.InitSet.SIMPLE;
        settings.optimizer = Settings.OptimizerSet.SGD;
        settings.iterationMode = Settings.IterationMode.DFS_RECURSIVE;

        settings.oneQueryPerExample = true;
        settings.neuralNetsPostProcessing = true;
        settings.isoValueCompression = false;
        settings.chainPruning = true;
        settings.storeNotShow = true;

        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void mutagen_diffcheck_mods() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-e ./resources/datasets/relational/molecules/mutagenesis/examples " +
                "-q ./resources/datasets/relational/molecules/mutagenesis/queries " +
//                "-t ./resources/datasets/relational/molecules/mutagenesis/template_new.txt").split(" ");
                "-t ./resources/datasets/relational/molecules/mutagenesis/template_vector_cross").split(" ");

        Settings settings = new Settings();

        settings.initDistribution = Settings.InitDistribution.CONSTANT;

        settings.seed = 0;  //the seed shouldn't matter here!
        settings.shuffleBeforeFoldSplit = false;
        settings.shuffleBeforeTraining = false;
        settings.shuffleEachEpoch = false;

        settings.debugSampleOutputs = true;
        settings.storeNotShow = true;

        settings.initLearningRate = 0.3;
        settings.maxCumEpochCount = 100;
        settings.resultsRecalculationEpochae = 1;
        settings.calculateBestThreshold = false;
//        settings.appLimitSamples = 100;
        settings.initializer = Settings.InitSet.SIMPLE;
        settings.optimizer = Settings.OptimizerSet.SGD;
        settings.iterationMode = Settings.IterationMode.TOPOLOGIC;  //tested as equivalent to DFS

        settings.oneQueryPerExample = true;
        settings.neuralNetsPostProcessing = true;
        settings.isoValueCompression = false;
        settings.chainPruning = true;

        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void mutagen_diffcheck_uniform() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-e ./resources/datasets/relational/molecules/mutagenesis/examples " +
                "-q ./resources/datasets/relational/molecules/mutagenesis/queries " +
//                "-t ./resources/datasets/relational/molecules/mutagenesis/template_new.txt").split(" ");
                "-t ./resources/datasets/relational/molecules/mutagenesis/template_vector_cross").split(" ");

        Settings settings = new Settings();

        settings.initDistribution = Settings.InitDistribution.UNIFORM;

        settings.seed = 0;  //the seed shouldn't matter here!
        settings.shuffleBeforeFoldSplit = false;
        settings.shuffleBeforeTraining = true;
        settings.shuffleEachEpoch = true;

        settings.debugSampleOutputs = false;
        settings.storeNotShow = true;

        settings.initLearningRate = 0.3;
        settings.maxCumEpochCount = 100;
        settings.resultsRecalculationEpochae = 1;
        settings.calculateBestThreshold = false;
//        settings.appLimitSamples = 100;
        settings.initializer = Settings.InitSet.SIMPLE;
        settings.optimizer = Settings.OptimizerSet.SGD;
        settings.iterationMode = Settings.IterationMode.TOPOLOGIC;  //tested as equivalent to DFS

        settings.oneQueryPerExample = true;
        settings.neuralNetsPostProcessing = true;
        settings.isoValueCompression = false;
        settings.chainPruning = true;

        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void mutagen_standard() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-e ./resources/datasets/relational/molecules/mutagenesis/examples.txt " +
                "-q ./resources/datasets/relational/molecules/mutagenesis/queries.txt " +
                "-t ./resources/datasets/relational/molecules/mutagenesis/template_new.txt").split(" ");

        Settings settings = new Settings();

        settings.initDistribution = Settings.InitDistribution.UNIFORM;

        settings.seed = 0;
        settings.initLearningRate = 0.3;
        settings.maxCumEpochCount = 1000;
        settings.resultsRecalculationEpochae = 10;
        settings.shuffleEachEpoch = true;
        settings.debugSampleOutputs = false;
        settings.calculateBestThreshold = true;
//        settings.appLimitSamples = 2;
        settings.initializer = Settings.InitSet.SIMPLE;
        settings.optimizer = Settings.OptimizerSet.SGD;
        settings.iterationMode = Settings.IterationMode.TOPOLOGIC;

        settings.oneQueryPerExample = true;
        settings.neuralNetsPostProcessing = true;
        settings.chainPruning = true;

        settings.isoValueCompression = true;
        settings.losslessIsoCompression = false;
        settings.isoValueInits = 1;
        settings.isoDecimals = 12;


        settings.storeNotShow = true;

        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void mutagen_vector_element() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-e ./resources/datasets/relational/molecules/mutagenesis/trainExamples.txt " +
                "-q ./resources/datasets/relational/molecules/mutagenesis/trainQueries.txt " +
                "-t ./resources/datasets/relational/molecules/mutagenesis/template_vectorized.txt").split(" ");

        Settings settings = new Settings();

        settings.initDistribution = Settings.InitDistribution.UNIFORM;

        settings.seed = 0;
        settings.initLearningRate = 0.01;
        settings.maxCumEpochCount = 100;
        settings.resultsRecalculationEpochae = 10;
        settings.shuffleEachEpoch = true;
        settings.debugSampleOutputs = false;
        settings.calculateBestThreshold = true;
//        settings.appLimitSamples = 100;
        settings.initializer = Settings.InitSet.SIMPLE;
        settings.optimizer = Settings.OptimizerSet.ADAM;
        settings.iterationMode = Settings.IterationMode.TOPOLOGIC;

        settings.oneQueryPerExample = true;
        settings.neuralNetsPostProcessing = true;
        settings.chainPruning = true;
        settings.isoValueCompression = true;
        settings.storeNotShow = true;

        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void mutagen_vector_cross() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-e ./resources/datasets/relational/molecules/mutagenesis/examples.txt " +
                "-q ./resources/datasets/relational/molecules/mutagenesis/queries.txt " +
                "-t ./resources/datasets/relational/molecules/mutagenesis/template_vector_cross.txt" +
                " -out ./out/expo").split(" ");

        Settings settings = new Settings();

        settings.initDistribution = Settings.InitDistribution.UNIFORM;

        settings.seed = 2;
        settings.initLearningRate = 0.3;   //todo now make default initLearningRate change based on optimizer
        settings.maxCumEpochCount = 10;
        settings.resultsRecalculationEpochae = 10;
        settings.shuffleEachEpoch = true;
        settings.debugSampleOutputs = false;
        settings.calculateBestThreshold = true;
//        settings.appLimitSamples = 10;
        settings.initializer = Settings.InitSet.SIMPLE;
        settings.optimizer = Settings.OptimizerSet.SGD;
        settings.iterationMode = Settings.IterationMode.TOPOLOGIC;

        settings.oneQueryPerExample = true;
        settings.neuralNetsPostProcessing = true;
        settings.chainPruning = true;

        settings.isoValueCompression = true;
        settings.losslessIsoCompression = false;
        settings.isoValueInits = 1;
        settings.isoDecimals = 12;

        settings.errorFunction = Settings.ErrorFcn.CROSSENTROPY;

        settings.storeNotShow = true;

        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void mutagen_gnns() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-e ./resources/datasets/relational/molecules/mutagenesis/examples " +
                "-q ./resources/datasets/relational/molecules/mutagenesis/queries " +
                "-t ./resources/datasets/relational/molecules/mutagenesis/template_unified_gnn_bad" +
                " -out ./out/expo").split(" ");

        Settings settings = new Settings();

        settings.initDistribution = Settings.InitDistribution.UNIFORM;

        settings.seed = 0;
        settings.initLearningRate = 0.01;   //todo now make default initLearningRate change based on optimizer
        settings.maxCumEpochCount = 1000;
        settings.resultsRecalculationEpochae = 10;
        settings.shuffleEachEpoch = true;
        settings.debugSampleOutputs = false;
        settings.calculateBestThreshold = true;
//        settings.appLimitSamples = 10;
        settings.initializer = Settings.InitSet.SIMPLE;
        settings.optimizer = Settings.OptimizerSet.ADAM;
        settings.iterationMode = Settings.IterationMode.TOPOLOGIC;

        settings.oneQueryPerExample = true;
        settings.neuralNetsPostProcessing = true;
        settings.chainPruning = true;

        settings.isoValueCompression = true;
        settings.losslessIsoCompression = false;
        settings.isoValueInits = 1;
        settings.isoDecimals = 3;

        settings.storeNotShow = true;

        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void jair_mda_dataset() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-e ./resources/datasets/relational/molecules/MDA_MB_231_ATCC/examples " +
                "-q ./resources/datasets/relational/molecules/MDA_MB_231_ATCC/queries " +
                "-t ./resources/datasets/relational/molecules/template_unified_bad").split(" ");

        Settings settings = new Settings();

        settings.initDistribution = Settings.InitDistribution.UNIFORM;

        settings.seed = 0;
        settings.initLearningRate = 0.01;   //todo now make default initLearningRate change based on optimizer
        settings.maxCumEpochCount = 1000;
        settings.resultsRecalculationEpochae = 10;
        settings.shuffleEachEpoch = true;
        settings.debugSampleOutputs = false;
        settings.calculateBestThreshold = true;
//        settings.appLimitSamples = 100;
        settings.initializer = Settings.InitSet.SIMPLE;
        settings.optimizer = Settings.OptimizerSet.ADAM;
        settings.iterationMode = Settings.IterationMode.TOPOLOGIC;

        settings.oneQueryPerExample = true;
        settings.neuralNetsPostProcessing = true;
        settings.chainPruning = true;
        settings.isoValueCompression = true;
        settings.isoDecimals = 1;
        settings.storeNotShow = true;

        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void nations() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-q ./resources/datasets/relational/kbs/nations/trainQueries.txt " +
                "-e ./resources/datasets/relational/kbs/nations/facts " +
                "-t ./resources/datasets/relational/kbs/nations/template_advanced"+
                " -out ./out/nations").split(" ");

        Settings settings = new Settings();

//        settings.neuralNetsPostProcessing = false;
        settings.chainPruning = true;
        settings.isoValueCompression = true;
        settings.losslessIsoCompression = false;

        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void kinships() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-q ./resources/datasets/relational/kbs/kinships/trainQueries.txt " +
                "-e ./resources/datasets/relational/kbs/kinships/facts " +
                "-t ./resources/datasets/relational/kbs/kinships/template_embeddings"+
                " -out ./out/kinships").split(" ");

        Settings settings = new Settings();

//        settings.neuralNetsPostProcessing = false;
        settings.chainPruning = true;
        settings.isoValueCompression = true;
        settings.losslessIsoCompression = false;

//        settings.iterationMode = Settings.IterationMode.DFS_STACK;

        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }

    @Test
    public void umls() {
        Logging logging = Logging.initLogging(Level.FINER);
        String[] args = ("-q ./resources/datasets/relational/kbs/umls/trainQueries_small.txt " +
                "-e ./resources/datasets/relational/kbs/umls/facts " +
                "-t ./resources/datasets/relational/kbs/umls/template_embeddings"+
                " -out ./out/umls").split(" ");

        Settings settings = new Settings();

//        settings.neuralNetsPostProcessing = false;
        settings.chainPruning = true;
        settings.isoValueCompression = true;
        settings.losslessIsoCompression = false;

        TrainingDebugger trainingDebugger = new TrainingDebugger(args, settings);
        trainingDebugger.executeDebug();
    }
}