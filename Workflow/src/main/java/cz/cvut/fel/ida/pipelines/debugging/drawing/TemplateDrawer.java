package cz.cvut.fel.ida.pipelines.debugging.drawing;

import cz.cvut.fel.ida.drawing.Drawer;
import cz.cvut.fel.ida.logic.Clause;
import cz.cvut.fel.ida.logic.Literal;
import cz.cvut.fel.ida.logic.Predicate;
import cz.cvut.fel.ida.logic.constructs.example.ValuedFact;
import cz.cvut.fel.ida.logic.constructs.template.Template;
import cz.cvut.fel.ida.logic.constructs.template.components.BodyAtom;
import cz.cvut.fel.ida.logic.constructs.template.components.WeightedRule;
import cz.cvut.fel.ida.logic.constructs.template.types.GraphTemplate;
import cz.cvut.fel.ida.logic.subsumption.Matching;
import cz.cvut.fel.ida.setup.Settings;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by gusta on 8.3.17.
 */
public class TemplateDrawer extends Drawer<Template> {

    Set<String> graph;

    public TemplateDrawer(Settings settings) {
        super(settings);
    }

    @Override
    public void loadGraph(Template obj) {
        graph = new LinkedHashSet<>();
        startStyledGraph();
        if (obj instanceof GraphTemplate) {
            loadGraph((GraphTemplate) obj);
        } else {
            loadGraph(new GraphTemplate(obj));
        }
        graph.forEach(graphviz::addln);
        graphviz.end_graph();
    }

    public void loadGraph(GraphTemplate obj) {

        Map<Literal, Set<WeightedRule>> atom2rules = obj.atom2rules;
        LinkedHashSet<ValuedFact> facts = obj.facts;
        Matching matching = new Matching(Collections.singletonList(new Clause(facts.stream().map(f -> f.literal).collect(Collectors.toSet()))));
        Map<Predicate, Set<ValuedFact>> pred2facts = new HashMap<>();

        for (ValuedFact fact : facts) {
            graph.add(draw(fact));
            Set<ValuedFact> factSet = pred2facts.computeIfAbsent(fact.literal.predicate(), k -> new HashSet<>());
            factSet.add(fact);
        }

        for (Map.Entry<Literal, Set<WeightedRule>> entry : atom2rules.entrySet()) {
            Literal literal = entry.getKey();
            Set<WeightedRule> rules = entry.getValue();
            graph.add(draw(literal));
            for (WeightedRule rule : rules) {
                graph.add(draw(rule)); //rule with subsumable head by this literal
                graph.add(draw(literal, rule));
                for (BodyAtom bodyAtom : rule.getBody()) {
                    graph.add(draw(bodyAtom));
                    graph.add(draw(rule, bodyAtom)); //body literals from the rule

                    Set<ValuedFact> matchedFacts;
                    if ((matchedFacts = pred2facts.get(bodyAtom.literal.predicate())) != null)
                        for (ValuedFact matchedFact : matchedFacts) {
                            if (matching.subsumption(new Clause(matchedFact.literal), new Clause(bodyAtom.literal))
                                    || matching.subsumption(new Clause(bodyAtom.literal), new Clause(matchedFact.literal))) {
                                graph.add(draw(bodyAtom, matchedFact));
                            }
                        }
                }
            }
        }
    }

    private String draw(WeightedRule rule, BodyAtom bodyAtom) {
        String styleKey = bodyAtom.isNegated() ? "negatedRuleToBody" : "ruleToBody";
        String weight = "";
        if (style.displayValues && style.displayWeights && bodyAtom.getConjunctWeight() != null) {
            weight = bodyAtom.getConjunctWeight().toString(numberFormat);
        }
        return dot.edge(bodyAtom.literal.liftedHashCode(), rule.hashCode(), styleKey, weight);
    }

    private String draw(BodyAtom bodyAtom) {
        return draw(bodyAtom.literal);
    }

    private String draw(Literal literal, WeightedRule rule) {
        String weight = "";
        if (style.displayValues && style.displayWeights) {
            weight = rule.getWeight() != null ? rule.getWeight().toString(numberFormat) : "null : 1";
        }
        return dot.edge(rule.hashCode(), literal.liftedHashCode(), "ruleToHead", weight);
    }

    private String draw(BodyAtom bodyAtom, ValuedFact matchedFact) {
        return dot.edge(matchedFact.literal.liftedHashCode(), bodyAtom.literal.liftedHashCode(), "bodyToFact", "");
    }

    private String draw(Literal literal) {
        String name = literal.toString();
        if (literal.isNegated()) {
            name = name.substring(1);
        }
        return dot.node(literal.liftedHashCode(), name, "literal");
    }

    private String draw(WeightedRule rule) {
        return dot.node(rule.hashCode(), rule.getOriginalString(), "rule");
    }

    private String draw(ValuedFact fact) {
        return dot.node(fact.literal.liftedHashCode(), fact.toString(), "fact");
    }

}
