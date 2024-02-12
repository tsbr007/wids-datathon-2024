package com.wids.datathon.datathon;

import java.util.ArrayList;
import java.util.List;

public class DecisionTreeNode {
    private String label;
    private List<DecisionTreeNode> children;

    public DecisionTreeNode() {
        this.label = null;
        this.children = new ArrayList<>();
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public List<DecisionTreeNode> getChildren() {
        return children;
    }

    public void addChild(DecisionTreeNode child) {
        children.add(child);
    }
}

