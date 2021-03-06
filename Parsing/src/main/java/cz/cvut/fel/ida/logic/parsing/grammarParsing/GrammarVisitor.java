package cz.cvut.fel.ida.logic.parsing.grammarParsing;

import cz.cvut.fel.ida.logic.constructs.building.LogicSourceBuilder;

public abstract class GrammarVisitor {

    /**
     * LogicSourceBuilder creates basic logic constructs.building blocks (variables, predicates, contants) as the visitor walks the parse tree
     */
    LogicSourceBuilder builder;

    public GrammarVisitor(LogicSourceBuilder builder) {
        this.builder = builder;
    }
}
