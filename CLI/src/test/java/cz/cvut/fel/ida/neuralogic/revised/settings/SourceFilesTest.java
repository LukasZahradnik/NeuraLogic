package cz.cvut.fel.ida.neuralogic.revised.settings;

import cz.cvut.fel.ida.neuralogic.cli.utils.Runner;
import cz.cvut.fel.ida.pipelines.debugging.TrainingDebugger;
import cz.cvut.fel.ida.setup.Settings;
import cz.cvut.fel.ida.utils.generic.TestAnnotations;
import cz.cvut.fel.ida.utils.generic.Utilities;

public class SourceFilesTest {

    @TestAnnotations.Fast
    public void multipleTemplates() throws Exception {
        Settings settings = Settings.forFastTest();
        String[] args = Utilities.getDatasetArgs("relational/molecules/mutagenesis", "-t ./templates/embeddings.txt,./templates/template_partial.txt");
        TrainingDebugger trainingDebugger = new TrainingDebugger(Runner.getSources(args, settings), settings);
        trainingDebugger.executeDebug();
    }
}