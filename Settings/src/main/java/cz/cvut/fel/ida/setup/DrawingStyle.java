package cz.cvut.fel.ida.setup;

import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Programmatic drawing style, intended to be configured directly from PyNeuraLogic
 * (or any other embedder) by mutating the public fields below.
 *
 * <p>Example from PyNeuraLogic:</p>
 * <pre>
 * style = settings.drawingStyle
 * style.valueDecimals = 4
 * style.displayValues = False
 * style.graph.put("bgcolor", "white")
 * style.nodeStyles.get("fact").put("fillcolor", "#fff2cc")
 * </pre>
 *
 * Attribute maps are keyed by semantic element names ("fact", "rule", "atomNeuron", ...),
 * and values are Graphviz attributes ("shape", "color", "style", "fontsize", ...).
 */
public class DrawingStyle implements Serializable {

    /**
     * Graph-level Graphviz attributes, e.g. bgcolor, rankdir, splines, fontname, nodesep, ranksep.
     */
    public Map<String, String> graph = new LinkedHashMap<>();

    /**
     * Semantic node style key -> Graphviz node attributes.
     */
    public Map<String, Map<String, String>> nodeStyles = new LinkedHashMap<>();

    /**
     * Semantic edge style key -> Graphviz edge attributes.
     */
    public Map<String, Map<String, String>> edgeStyles = new LinkedHashMap<>();

    /**
     * Number of decimal places used for numeric values in drawings. 0 = integers only.
     */
    public int valueDecimals = 2;

    /**
     * Master switch for all numeric values in drawings. If false, values/gradients/dimensions/functions/weights are omitted.
     */
    public boolean displayValues = true;

    public boolean displayGradients = true;

    public boolean displayDimensions = true;

    public boolean displayFunctions = true;

    public boolean displayWeights = true;

    public DrawingStyle() {
        setDefaults();
    }

    /**
     * Defaults reproduce the previously hardcoded look, so the default behavior is unchanged.
     */
    private void setDefaults() {
        // ---- TemplateDrawer ----
        nodeStyle("literal");
        nodeStyle("fact").put("shape", "house");
        nodeStyle("rule").put("shape", "larrow");
        nodeStyle("rule").put("color", "green");

        edgeStyle("ruleToBody").put("color", "black");
        edgeStyle("negatedRuleToBody").put("color", "red");
        edgeStyle("ruleToHead").put("color", "green");
        edgeStyle("ruleToHead").put("style", "dashed");
        edgeStyle("bodyToFact").put("dir", "both");
        edgeStyle("bodyToFact").put("style", "dotted");
        edgeStyle("bodyToFact").put("color", "blue");

        // ---- GroundingDrawer ----
        nodeStyle("groundLiteral").put("color", "blue");
        nodeStyle("groundLiteral").put("fontcolor", "blue");
        nodeStyle("groundHeadRule").put("shape", "larrow");
        nodeStyle("groundHeadRule").put("fontsize", "10");
        nodeStyle("groundHeadRule").put("color", "green");
        nodeStyle("groundRule").put("shape", "lpromoter");
        nodeStyle("groundRule").put("color", "red");
        nodeStyle("target").put("shape", "tripleoctagon");

        edgeStyle("headToGroundRule").put("color", "blue");
        edgeStyle("groundHeadRuleToGroundRule").put("style", "dashed");
        edgeStyle("groundHeadRuleToGroundRule").put("color", "green");
        edgeStyle("groundRuleToLiteral").put("color", "red");
        edgeStyle("groundRuleToLiteral").put("fontcolor", "red");
        edgeStyle("compactRuleEdge").put("fontsize", "10");
        edgeStyle("compactRuleEdge").put("color", "blue");
        edgeStyle("compactRuleEdge").put("fontcolor", "green");

        // ---- NeuralNetDrawer ----
        nodeStyle("atomNeuron").put("shape", "ellipse");
        nodeStyle("atomNeuron").put("color", "blue");
        nodeStyle("weightedAtomNeuron").put("shape", "ellipse");
        nodeStyle("weightedAtomNeuron").put("color", "blue");
        nodeStyle("negationNeuron").put("shape", "ellipse");
        nodeStyle("negationNeuron").put("color", "red");
        nodeStyle("aggregationNeuron").put("shape", "box");
        nodeStyle("aggregationNeuron").put("color", "green");
        nodeStyle("ruleNeuron").put("shape", "ellipse");
        nodeStyle("ruleNeuron").put("color", "red");
        nodeStyle("weightedRuleNeuron").put("shape", "ellipse");
        nodeStyle("weightedRuleNeuron").put("color", "red");
        nodeStyle("factNeuron").put("shape", "house");
        nodeStyle("factNeuron").put("color", "black");
        nodeStyle("targetNeuron").put("shape", "tripleoctagon");

        edgeStyle("unweightedInput").put("style", "dashed");
        edgeStyle("weightedInput");
        edgeStyle("fixedWeightInput").put("style", "dashed");

        // ---- PipelineDrawer ----
        nodeStyle("pipe").put("shape", "cds");
        nodeStyle("identityPipe").put("shape", "point");
        nodeStyle("identityPipe").put("width", "0.1");
        nodeStyle("identityPipe").put("height", "0.1");
        nodeStyle("branch").put("shape", "trapezium");
        nodeStyle("merge").put("shape", "invtrapezium");
        nodeStyle("start").put("color", "green");
        nodeStyle("terminal").put("color", "red");

        edgeStyle("pipelineEdge");
        edgeStyle("pipelineTypeEdge");
    }

    /**
     * Creates a mutable attribute map for the given semantic node key, if absent.
     */
    public Map<String, String> nodeStyle(String key) {
        return nodeStyles.computeIfAbsent(key, k -> new LinkedHashMap<>());
    }

    /**
     * Creates a mutable attribute map for the given semantic edge key, if absent.
     */
    public Map<String, String> edgeStyle(String key) {
        return edgeStyles.computeIfAbsent(key, k -> new LinkedHashMap<>());
    }

    /**
     * PyNeuraLogic-friendly convenience: {@code style.setGraphAttribute("bgcolor", "white")}.
     */
    public DrawingStyle setGraphAttribute(String attribute, String value) {
        graph.put(attribute, value);
        return this;
    }

    /**
     * PyNeuraLogic-friendly convenience: {@code style.setNodeAttribute("fact", "fillcolor", "#fff2cc")}.
     */
    public DrawingStyle setNodeAttribute(String styleKey, String attribute, String value) {
        nodeStyle(styleKey).put(attribute, value);
        return this;
    }

    /**
     * PyNeuraLogic-friendly convenience: {@code style.setEdgeAttribute("ruleToHead", "color", "red")}.
     */
    public DrawingStyle setEdgeAttribute(String styleKey, String attribute, String value) {
        edgeStyle(styleKey).put(attribute, value);
        return this;
    }

    /**
     * Returns the attributes for a node key, or an empty map (never null).
     */
    public Map<String, String> nodeAttributes(String key) {
        Map<String, String> attrs = nodeStyles.get(key);
        return attrs != null ? attrs : java.util.Collections.emptyMap();
    }

    /**
     * Returns the attributes for an edge key, or an empty map (never null).
     */
    public Map<String, String> edgeAttributes(String key) {
        Map<String, String> attrs = edgeStyles.get(key);
        return attrs != null ? attrs : java.util.Collections.emptyMap();
    }
}
