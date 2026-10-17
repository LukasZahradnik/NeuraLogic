package cz.cvut.fel.ida.pipelines.debuging.drawing;

import cz.cvut.fel.ida.drawing.DotBuilder;
import cz.cvut.fel.ida.drawing.Drawer;
import cz.cvut.fel.ida.pipelines.*;
import cz.cvut.fel.ida.pipelines.pipes.generic.IdentityGenPipe;
import cz.cvut.fel.ida.setup.Settings;
//import sun.reflect.generics.reflectiveObjects.TypeVariableImpl;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.logging.Logger;

public class PipelineDrawer<S, T> extends Drawer<Pipeline<S, T>> {
    private static final Logger LOG = Logger.getLogger(PipelineDrawer.class.getName());

    public PipelineDrawer(Settings settings) {
        super(settings);
    }

    @Override
    public void loadGraph(Pipeline<S, T> pipeline) {
        this.graphviz.addln("strict digraph {");
        graphviz.addln("compound=true");
        addGraphDefaults();
        loadPipeline(pipeline);

        this.graphviz.end_graph();
    }


    private void loadPipeline(Pipeline<S, T> pipeline) {

        for (Map.Entry<String, Pipe> entry : pipeline.pipes.entrySet()) {
            draw(entry.getValue());
        }

        for (Map.Entry<String, Branch> entry : pipeline.branches.entrySet()) {
            draw(entry.getValue());
        }

        for (Map.Entry<String, Merge> entry : pipeline.merges.entrySet()) {
            draw(entry.getValue());
        }

        for (Map.Entry<String, ParallelPipe> entry : pipeline.multiPipes.entrySet()) {
            draw(entry.getValue());
        }

        for (Map.Entry<String, MultiBranch> entry : pipeline.multiBranches.entrySet()) {
            draw(entry.getValue());
        }

        for (Map.Entry<String, MultiMerge> entry : pipeline.multiMerges.entrySet()) {
            draw(entry.getValue());
        }

        for (Map.Entry<String, Pipeline> entry : pipeline.pipelines.entrySet()) {
            drawInnerPipeline(entry.getValue());
        }

        drawStart(getStartElement(pipeline));
        drawTerminal(getTerminalElement(pipeline));
    }

    private void drawStart(Object start) {
        graphviz.addln(dot.node(getNodeId(start), start.toString(), "start"));
    }

    private void drawTerminal(Object terminal) {
        if (terminal != null)
            graphviz.addln(dot.node(getNodeId(terminal), terminal.toString(), "terminal"));
    }

    private void draw(Pipe pipe) {
        ArrayList<String> types = GenericTypeGetter.getTypes(pipe); //todo next replace reflection with explicit Class info storing within the pipes - e.g. to show "Sources" type as the input

        if (pipe instanceof IdentityGenPipe) {
            graphviz.addln(dot.node(getNodeId(pipe), "", "identityPipe"));
        } else {
            graphviz.addln(dot.node(getNodeId(pipe), pipe.toString(), "pipe"));
        }
        if (pipe.input != null && types != null && types.size() > 0)
            drawEdge(pipe.input, pipe, "pipelineTypeEdge", types.get(0));
        if (pipe.output != null && types != null && types.size() > 1)
            drawEdge(pipe, pipe.output, "pipelineTypeEdge", types.get(1));
    }

    private void draw(Branch branch) {
        graphviz.addln(dot.node(getNodeId(branch), branch.toString(), "branch"));
        if (branch.input != null)
            drawEdge(branch.input, branch);
        if (branch.output1 != null)
            drawEdge(branch, branch.output1);
        if (branch.output2 != null)
            drawEdge(branch, branch.output2);
    }

    private void draw(Merge merge) {
        graphviz.addln(dot.node(getNodeId(merge), merge.toString(), "merge"));
        if (merge.input1 != null) {
            graphviz.addln(dot.node(getNodeId(merge.input1), "", "identityPipe"));
            drawEdge(merge.input1, merge);
        }
        if (merge.input2 != null) {
            graphviz.addln(dot.node(getNodeId(merge.input2), "", "identityPipe"));
            drawEdge(merge.input2, merge);
        }
        if (merge.output != null)
            drawEdge(merge, merge.output);
    }

    private void draw(ParallelPipe pipe) {
        graphviz.addln(dot.node(pipe.hashCode(), getNodeId(pipe.toString()), "pipe"));
        for (Object input : pipe.inputs) {
            drawEdge(input, pipe);
        }
        for (Object output : pipe.outputs) {
            drawEdge(pipe, output);
        }
    }

    private void draw(MultiBranch branch) { //todo now test some multi-pipeline (xval)
        graphviz.addln(dot.node(getNodeId(branch), branch.toString(), "branch"));
        if (branch.input != null)
            drawEdge(branch.input, branch);
        for (Object output : branch.outputs) {
            drawEdge(branch, output);
        }
    }

    private void draw(MultiMerge merge) {
        graphviz.addln(dot.node(getNodeId(merge), merge.toString(), "merge"));
        for (Object input : merge.inputs) {
            drawEdge(input, merge);
        }
        if (merge.output != null)
            drawEdge(merge, merge.output);
    }

    private void drawInnerPipeline(Pipeline pipeline) {
        graphviz.addln("subgraph cluster_" + pipeline.hashCode() + " {");
        this.loadPipeline(pipeline);
        graphviz.addln("label=" + DotBuilder.quote(pipeline.toString()) + "}");

        if (pipeline.input != null)
            drawEdge(pipeline.input, pipeline);
        if (pipeline.output != null)
            drawEdge(pipeline, pipeline.output);
    }

    private void drawEdge(Object obj1, Object obj2) {
        drawEdge(obj1, obj2, "pipelineEdge", "");
    }

    private void drawEdge(Object obj1, Object obj2, String styleKey, String label) {
        Map<String, String> linkAttributes = new LinkedHashMap<>();
        if (obj1 instanceof Pipeline) {
            linkAttributes.put("ltail", getNodeId(obj1));
        }
        if (obj2 instanceof Pipeline) {
            linkAttributes.put("lhead", getNodeId(obj2));
        }
        String node1 = getEnd(obj1);
        String node2 = getStart(obj2);
        graphviz.addln(dot.edge(node1, node2, styleKey, label, linkAttributes));
    }

    private String getStart(Object obj) {
        if (obj instanceof Pipeline) {
            Pipeline pipeline = (Pipeline) obj;
            return getNodeId(pipeline.start);
        } else {
            return getNodeId(obj);
        }
    }

    private String getEnd(Object obj) {
        if (obj instanceof Pipeline) {
            Pipeline pipeline = (Pipeline) obj;
            if (pipeline.terminal != null)
                return getNodeId(pipeline.terminal);
            else {
                LOG.severe("Pipeline without terminal");
                return getNodeId(pipeline.start);   // todo next
            }
        } else {
            return getNodeId(obj);
        }
    }

    private String getNodeId(Object object) {
        int hashCode = object.hashCode();
        if (object instanceof Pipeline) {
            return "cluster_" + hashCode;
        } else {
            return String.valueOf(hashCode);
        }
    }

    private Object getStartElement(Object object) {
        if (object instanceof Pipeline) {
            Pipeline pipeline = (Pipeline) object;
            return getStartElement(pipeline.start);
        } else {
            return object;
        }
    }

    private Object getTerminalElement(Object object) {
        if (object instanceof Pipeline) {
            Pipeline pipeline = (Pipeline) object;
            return getTerminalElement(pipeline.terminal);
        } else {
            return object;
        }
    }

    private static class GenericTypeGetter {

        public static ArrayList<String> getTypes(Object object) {
            try {
                Type genericSuperclass = object.getClass().getGenericSuperclass();
                Type[] actualTypeArguments = ((ParameterizedType) genericSuperclass).getActualTypeArguments();
                return recurseTypes(actualTypeArguments);
            } catch (Exception e) {
                LOG.warning("Java error while hacking around with reflection for recovering generics at runtime.");
            }
            return null;
        }

        private static ArrayList<String> recurseTypes(Type[] actualTypeArguments) {
            ArrayList<String> types = new ArrayList<>();
            for (Type actualTypeArgument : actualTypeArguments) {
                types.add(getCompoundType(actualTypeArgument) + ",");
            }
            return types;
        }

        private static String getCompoundType(Type actualTypeArgument) {
            String simpleName = "";
            try {
                if (actualTypeArgument instanceof Class) {
                    Class aClass = (Class) actualTypeArgument;
                    simpleName = aClass.getSimpleName();
                } else if (actualTypeArgument instanceof ParameterizedType) {
                    Type[] actualTypeArguments = ((ParameterizedType) actualTypeArgument).getActualTypeArguments();
                    simpleName = Arrays.toString(recurseTypes(actualTypeArguments).toArray());
                }
//                else if (actualTypeArgument instanceof TypeVariableImpl) {
//                    TypeVariableImpl typeVariable = (TypeVariableImpl) actualTypeArgument;
//                    simpleName = typeVariable.getName();
//                }


            } catch (ClassCastException e) {
                System.out.println();
            }
            return simpleName;
        }
    }
}
