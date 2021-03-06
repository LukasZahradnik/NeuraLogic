package cz.cvut.fel.ida.logic.parsing.examples;

import cz.cvut.fel.ida.logic.parsing.antlr.NeuralogicParser;
import cz.cvut.fel.ida.logic.parsing.grammarParsing.PlainParseTree;

import java.io.IOException;
import java.io.Reader;
import java.util.logging.Logger;

public class PlainExamplesParseTree extends PlainParseTree<NeuralogicParser.ExamplesFileContext> {
    private static final Logger LOG = Logger.getLogger(PlainExamplesParseTree.class.getName());

    public PlainExamplesParseTree(Reader reader) throws IOException {
        super(reader);
    }

    @Override
    public NeuralogicParser.ExamplesFileContext getRoot() {
        return parseTree.examplesFile();
    }
}
