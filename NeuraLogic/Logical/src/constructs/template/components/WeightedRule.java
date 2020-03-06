package constructs.template.components;

import constructs.template.metadata.RuleMetadata;
import evaluation.functions.Activation;
import ida.ilp.logic.Clause;
import ida.ilp.logic.HornClause;
import ida.ilp.logic.Literal;
import ida.ilp.logic.Term;
import networks.structure.components.weights.Weight;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by Gusta on 04.10.2016.
 * <p>
 */
public class WeightedRule {

    /**
     * changable by structure learning?
     */
    boolean isEditable = false;

    private Weight weight;
    private Weight offset;

    private HeadAtom head;
    private List<BodyAtom> body;

    private Activation aggregationFcn;
    private Activation activationFcn;

    private RuleMetadata metadata;
    private String originalString;

    private int hashCode = -1;

    private boolean crossProduct;

    public WeightedRule() {

    }

    /**
     * This does not really clone the rule, only references
     *
     * @param other
     */
    public WeightedRule(WeightedRule other) {
        this.setWeight(other.getWeight());
        this.setHead(other.getHead());
        this.setBody(new ArrayList<>(other.getBody().size()));
        this.getBody().addAll(other.getBody());
        this.setOffset(other.getOffset());
        this.setAggregationFcn(other.getAggregationFcn());
        this.setActivationFcn(other.getActivationFcn());
        this.setMetadata(other.getMetadata());
        this.setOriginalString(other.getOriginalString());
        this.isEditable = other.isEditable;
    }

    public HornClause toHornClause() {
        List<Literal> collected = getBody().stream().map(bodyLit -> bodyLit.getLiteral()).collect(Collectors.toList());
        return new HornClause(getHead().getLiteral(), new Clause(collected));
    }

    /**
     * Grounding of individual atoms will create new copies of them.
     *
     * @param terms
     * @return
     */
    public GroundRule groundRule(Term[] terms) {

        Literal groundHead = head.literal.subsCopy(terms);

        int size = getBody().size();
        Literal[] groundBody = new Literal[size];

        for (int i = 0; i < size; i++) {
            groundBody[i] = getBody().get(i).literal.subsCopy(terms);
        }

        GroundRule groundRule = new GroundRule(this, groundHead, groundBody);

        return groundRule;
    }

    public GroundHeadRule groundHeadRule(Literal groundHead) {
        GroundHeadRule groundRule = new GroundHeadRule(this, groundHead);
        return groundRule;
    }

    public String signatureString() {
        StringBuilder sb = new StringBuilder();
        sb.append(getHead().getPredicate()).append(":-");
        for (BodyAtom bodyAtom : getBody()) {
            sb.append(bodyAtom.getPredicate()).append(",");
        }
        return sb.toString();
    }

    public boolean detectWeights() {
        boolean hasWeights = hasOffset();
        for (BodyAtom bodyAtom : getBody()) {
            if (bodyAtom.weight != null) {
                hasWeights = true;
            }
        }
        if (hasWeights) {
            for (BodyAtom bodyAtom : getBody()) {
                if (bodyAtom.weight == null) {
                    bodyAtom.weight = Weight.unitWeight;
                }
            }
        }
        return hasWeights;
    }

    @Override
    public int hashCode() {
        if (hashCode != -1)
            return hashCode;
        hashCode = head.hashCode() + body.hashCode();
        return hashCode;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) {   //this should catch absolute majority of all calls (due to factory creation and unique hash)
            return true;
        }
        if (!(obj instanceof WeightedRule)) {
            return false;
        }
        WeightedRule other = (WeightedRule) obj;
        if (getWeight() == null && other.getWeight() != null || getWeight() != null && other.getWeight() == null) {
            return false;
        }
        if (getOffset() == null && other.getOffset() != null || getOffset() != null && other.getOffset() == null) {
            return false;
        }
        if (getWeight() != null && !getWeight().equals(other.getWeight()) || getOffset() != null && !getOffset().equals(other.getOffset())) {
            return false;
        }
        if (getAggregationFcn() != null && !getAggregationFcn().equals(other.getAggregationFcn()) || getActivationFcn() != null && !getActivationFcn().equals(other.getActivationFcn())) {
            return false;
        }
        if (!getHead().equals(other.getHead()) || !getBody().equals(other.getBody())) {
            return false;
        }
        return true;
    }

    public String toRuleNeuronString() {
        StringBuilder sb = new StringBuilder();
        sb.append(getHead().toString()).append(":-");
        for (BodyAtom bodyAtom : getBody()) {
            sb.append(bodyAtom.toString()).append(",");
        }
        sb.setCharAt(sb.length() - 1, '.');
        return sb.toString();
    }

    public Weight getWeight() {
        return weight;
    }

    public void setWeight(Weight weight) {
        this.weight = weight;
    }

    public boolean hasOffset() {
        return offset != null;
    }

    public Weight getOffset() {
        return offset;
    }

    public void setOffset(Weight offset) {
        this.offset = offset;
    }

    public HeadAtom getHead() {
        return head;
    }

    public void setHead(HeadAtom head) {
        this.head = head;
    }

    public List<BodyAtom> getBody() {
        return body;
    }

    public void setBody(List<BodyAtom> body) {
        this.body = body;
    }

    public Activation getAggregationFcn() {
        return aggregationFcn;
    }

    public void setAggregationFcn(Activation aggregationFcn) {
        this.aggregationFcn = aggregationFcn;
    }

    public Activation getActivationFcn() {
        return activationFcn;
    }

    public void setActivationFcn(Activation activationFcn) {
        this.activationFcn = activationFcn;
    }

    public RuleMetadata getMetadata() {
        return metadata;
    }

    public void setMetadata(RuleMetadata metadata) {
        this.metadata = metadata;
    }

    public String getOriginalString() {
        return originalString;
    }

    public void setOriginalString(String originalString) {
        this.originalString = originalString;
    }

    /**
     * Apply {@link networks.computation.evaluation.functions.CrossProduct} activation on the inputs of the rule?
     */
    public boolean isCrossProduct() {
        return crossProduct;
    }

    public void setCrossProduct(boolean crossProduct) {
        this.crossProduct = crossProduct;
    }
}