package cz.cvut.fel.ida.pipelines.pipes.generic;

import cz.cvut.fel.ida.pipelines.Branch;
import cz.cvut.fel.ida.utils.generic.Pair;

import java.util.logging.Logger;

public class PairBranch<O1,O2> extends Branch<Pair<O1,O2>,O1,O2> {
    private static final Logger LOG = Logger.getLogger(PairBranch.class.getName());

    public PairBranch(){
        super("PairBranch");
    }

    public PairBranch(String id) {
        super(id);
    }

    @Override
    protected Pair<O1, O2> branch(Pair<O1, O2> outputFromInputPipe) {
        return outputFromInputPipe;
    }
}
