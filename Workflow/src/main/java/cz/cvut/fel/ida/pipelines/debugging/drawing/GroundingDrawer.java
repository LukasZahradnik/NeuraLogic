package cz.cvut.fel.ida.pipelines.debugging.drawing;

import cz.cvut.fel.ida.drawing.Drawer;
import cz.cvut.fel.ida.logic.Literal;
import cz.cvut.fel.ida.logic.constructs.template.components.GroundHeadRule;
import cz.cvut.fel.ida.logic.constructs.template.components.GroundRule;
import cz.cvut.fel.ida.logic.grounding.GroundTemplate;
import cz.cvut.fel.ida.logic.grounding.GroundingSample;
import cz.cvut.fel.ida.setup.Settings;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.logging.Logger;

public class GroundingDrawer extends Drawer<GroundingSample> {
    private static final Logger LOG = Logger.getLogger(GroundingDrawer.class.getName());

    boolean compact = true;

    public GroundingDrawer(Settings settings) {
        super(settings);
        this.compact = settings.compactGroundingDrawing;
    }

    @Override
    public void loadGraph(GroundingSample sample) {
        startStyledGraph();

        nodeGraph(sample);

        if (sample.query.headAtom != null) {
            Map<String, String> targetAttributes = new LinkedHashMap<>();
            targetAttributes.put("xlabel", "\n\n\ntarget = " + sample.target + "  ");
            graphviz.addln(dot.node(sample.query.headAtom.literal.toString(), "", "target", targetAttributes));
        }

        graphviz.end_graph();
    }

    public void nodeGraph(GroundingSample obj) {
        GroundTemplate groundTemplate = obj.groundingWrap.getGroundTemplate();
        LinkedHashMap<Literal, LinkedHashMap<GroundHeadRule, Collection<GroundRule>>> groundRules = groundTemplate.groundRules;

        for (Map.Entry<Literal, LinkedHashMap<GroundHeadRule, Collection<GroundRule>>> mapEntry : groundRules.entrySet()) {
            Literal groundHead = mapEntry.getKey();
            graphviz.addln(draw(groundHead));
            for (Map.Entry<GroundHeadRule, Collection<GroundRule>> entry : mapEntry.getValue().entrySet()) {
                GroundHeadRule groundHeadRule = entry.getKey();
                if (!compact) {
                    graphviz.addln(draw(groundHeadRule));
                    graphviz.addln(drawEdge(groundHead, groundHeadRule));
                }
                Collection<GroundRule> bodyGroundings = entry.getValue();
                for (GroundRule bodyGrounding : bodyGroundings) {
                    graphviz.addln(draw(bodyGrounding));
                    if (compact) {
                        graphviz.addln(drawEdge(groundHead, bodyGrounding, groundHeadRule));
                    } else
                        graphviz.addln(drawEdge(groundHeadRule, bodyGrounding));
                    for (Literal literal : bodyGrounding.groundBody) {
                        graphviz.addln(draw(literal));
                        graphviz.addln(drawEdge(bodyGrounding, literal));
                    }
                }
            }
        }
    }

    private String drawEdge(Literal groundHead, GroundRule bodyGrounding, GroundHeadRule groundHeadRule) {
        return dot.edge(bodyGrounding.hashCode(), groundHead.toString(), "compactRuleEdge", groundHeadRule.toFullString());
    }

    private String draw(GroundHeadRule groundHeadRule) {
        return dot.node(groundHeadRule.hashCode(), groundHeadRule.toFullString(), "groundHeadRule");
    }

    private String draw(GroundRule groundRule) {
        return dot.node(groundRule.hashCode(), groundRule.toString(), "groundRule");
    }

    public String draw(Literal groundHead) {
        String name = "null";
        if (groundHead != null) {
            name = groundHead.toString();
        }
        return dot.node(name, name, "groundLiteral");
    }

    private String drawEdge(GroundRule bodyGrounding, Literal literal) {
        return dot.edge(literal == null ? "null" : literal.toString(), bodyGrounding.hashCode(), "groundRuleToLiteral", "");
    }

    public String drawEdge(GroundHeadRule groundHeadRule, GroundRule bodyGrounding) {
        return dot.edge(bodyGrounding.hashCode(), groundHeadRule.hashCode(), "groundHeadRuleToGroundRule", "");
    }

    public String drawEdge(Literal groundHead, GroundHeadRule groundHeadRule) {
        return dot.edge(groundHeadRule.hashCode(), groundHead.toString(), "headToGroundRule", "");
    }

}
