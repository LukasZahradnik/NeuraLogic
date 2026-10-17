package cz.cvut.fel.ida.drawing;

import cz.cvut.fel.ida.setup.DrawingStyle;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Renders DOT node/edge statements from semantic style keys defined in {@link DrawingStyle}.
 * Handles quoting of ids, labels and attribute values so drawers no longer concatenate
 * raw Graphviz strings.
 */
public class DotBuilder {

    private final DrawingStyle style;

    public DotBuilder(DrawingStyle style) {
        this.style = style;
    }

    public String node(Object id, String label, String styleKey) {
        return node(id, label, styleKey, null);
    }

    public String node(Object id, String label, String styleKey, Map<String, String> extraAttributes) {
        List<String> attributes = new ArrayList<>();
        attributes.add("label=" + quote(label));
        addAll(attributes, style.nodeAttributes(styleKey));
        addAll(attributes, extraAttributes);
        return quote(String.valueOf(id)) + joinAttributes(attributes);
    }

    public String edge(Object from, Object to, String styleKey, String label) {
        return edge(from, to, styleKey, label, null);
    }

    public String edge(Object from, Object to, String styleKey, String label, Map<String, String> extraAttributes) {
        List<String> attributes = new ArrayList<>();
        if (label != null && !label.isEmpty()) {
            attributes.add("label=" + quote(label));
        }
        addAll(attributes, style.edgeAttributes(styleKey));
        addAll(attributes, extraAttributes);
        return quote(String.valueOf(from)) + " -> " + quote(String.valueOf(to)) + joinAttributes(attributes);
    }

    private void addAll(List<String> target, Map<String, String> attributes) {
        if (attributes == null) {
            return;
        }
        for (Map.Entry<String, String> attribute : attributes.entrySet()) {
            target.add(attribute.getKey() + "=" + quote(attribute.getValue()));
        }
    }

    private String joinAttributes(List<String> attributes) {
        if (attributes.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder(" [");
        for (int i = 0; i < attributes.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(attributes.get(i));
        }
        return sb.append("]").toString();
    }

    /**
     * Quotes a DOT id/label/attribute value and escapes backslashes and double quotes.
     */
    public static String quote(String value) {
        if (value == null) {
            return "\"\"";
        }
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }
}
