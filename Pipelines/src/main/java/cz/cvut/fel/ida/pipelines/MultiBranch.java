package cz.cvut.fel.ida.pipelines;

import cz.cvut.fel.ida.pipelines.pipes.generic.IdentityGenPipe;
import cz.cvut.fel.ida.setup.Settings;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * This method must be terminating, since it is not possible to fork streams,
 * hence the return value is List to signify this
 */
public abstract class MultiBranch<I, O> extends Block implements ConnectBefore<I> {
    private static final Logger LOG = Logger.getLogger(MultiBranch.class.getName());

    public ConnectAfter<I> input;
    public List<IdentityGenPipe<O>> outputs;

    List<O> outputReady;

    boolean parallelBranching = false;

    protected MultiBranch(String id, int count, Settings settings) {
        this.ID = id;
        this.settings = settings;

        outputs = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            outputs.add(new IdentityGenPipe<>(id + "Output" + i));
        }
    }

    /**
     * Self computation and Parallel invocation of successor Consumers of O
     *
     * @param outputFromInputPipe
     */
    public void accept(I outputFromInputPipe) throws Exception {
        LOG.finer("Entering: " + ID);
        outputReady = branch(outputFromInputPipe);
        if (outputs.size() != outputReady.size()) {
            LOG.severe("MultiBranch output dimension mismatches with subsequent consumers!");
        }
        if (parallelBranching) {
            //todo next
        } else {
            for (int i = 0; i < outputs.size(); i++) {
                outputs.get(i).accept(outputReady.get(i));
            }
        }
    }

    /**
     * Output List size must match the outputs!
     * @param outputFromInputPipe
     * @return
     */
    protected abstract List<O> branch(I outputFromInputPipe);

    @Override
    public ConnectAfter<I> getInput() {
        return input;
    }

    @Override
    public void setInput(ConnectAfter<I> prev) {
        input = prev;
    }

    public <T extends ConnectBefore<O>> List<T> connectAfter(List<T> next){
        if (next.size() != outputs.size()){
            LOG.severe("MultiBranch output dimension mismatches with subsequent consumers!");
        }
        for (int i = 0; i < outputs.size(); i++) {
            outputs.get(i).connectAfter(next.get(i));
        }
        return next;
    }
}
