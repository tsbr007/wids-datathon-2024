package com.wids.datathon.datathon;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

public class CustomDecisionTree extends AbstractClassifier {

	private Node root;

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		root = buildTree(instances);
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		return classify(instance, root);
	}

	private double classify(Instance instance, Node node) {
		if (node.isLeaf()) {
			return node.getLabel();
		} else {
			Attribute attribute = node.getAttribute();
			double attributeValue = instance.value(attribute);

			Node nextNode = node.getChild(attributeValue);
			return classify(instance, nextNode);
		}
	}

	private Node buildTree(Instances instances) {
		if (instances.numInstances() == 0) {
			return new LeafNode(0); // Default class for an empty dataset
		}

		if (allInstancesSameClass(instances)) {
			return new LeafNode(instances.instance(0).classValue());
		}

		Attribute bestAttribute = chooseBestAttribute(instances);

		if (bestAttribute == null) {
			return new LeafNode(getMajorityClass(instances));
		}

		Node node = new InternalNode(bestAttribute);

		for (int i = 0; i < bestAttribute.numValues(); i++) {
			Instances subset = getSubset(instances, bestAttribute, i);
			Node childNode = buildTree(subset);
			node.addChild(i, childNode);
		}

		return node;
	}

	private boolean allInstancesSameClass(Instances instances) {
		double firstClass = instances.instance(0).classValue();
		for (int i = 1; i < instances.numInstances(); i++) {
			if (instances.instance(i).classValue() != firstClass) {
				return false;
			}
		}
		return true;
	}

	private Attribute chooseBestAttribute(Instances instances) {
		double currentEntropy = calculateEntropy(instances);
		double bestInfoGain = 0.0;
		Attribute bestAttribute = null;

		Enumeration<Attribute> attributes = instances.enumerateAttributes();
		while (attributes.hasMoreElements()) {
			Attribute attribute = attributes.nextElement();
			if (attribute.isNominal()) {
				double infoGain = currentEntropy - calculateConditionalEntropy(instances, attribute);
				if (infoGain > bestInfoGain) {
					bestInfoGain = infoGain;
					bestAttribute = attribute;
				}
			}
		}

		return bestAttribute;
	}

	private double calculateEntropy(Instances instances) {
		int numInstances = instances.numInstances();
		Map<Double, Integer> classCounts = new HashMap<>();

		// Count the occurrences of each class
		for (int i = 0; i < numInstances; i++) {
			double classValue = instances.instance(i).classValue();
			classCounts.put(classValue, classCounts.getOrDefault(classValue, 0) + 1);
		}

		// Calculate entropy
		double entropy = 0.0;
		for (double classValue : classCounts.keySet()) {
			double probability = classCounts.get(classValue) / (double) numInstances;
			entropy -= probability * log2(probability);
		}

		return entropy;
	}

	private double calculateConditionalEntropy(Instances instances, Attribute attribute) {
		int numInstances = instances.numInstances();
		Map<Double, Map<Double, Integer>> attributeClassCounts = new HashMap<>();

		// Count the occurrences of each attribute value and class combination
		for (int i = 0; i < numInstances; i++) {
			double attributeValue = instances.instance(i).value(attribute);
			double classValue = instances.instance(i).classValue();

			attributeClassCounts.putIfAbsent(attributeValue, new HashMap<>());
			Map<Double, Integer> classCounts = attributeClassCounts.get(attributeValue);

			classCounts.put(classValue, classCounts.getOrDefault(classValue, 0) + 1);
		}

		// Calculate conditional entropy
		double conditionalEntropy = 0.0;
		for (double attributeValue : attributeClassCounts.keySet()) {
			Map<Double, Integer> classCounts = attributeClassCounts.get(attributeValue);
			int totalAttributeCount = classCounts.values().stream().mapToInt(Integer::intValue).sum();

			double attributeEntropy = 0.0;
			for (double classValue : classCounts.keySet()) {
				double probability = classCounts.get(classValue) / (double) totalAttributeCount;
				attributeEntropy -= probability * log2(probability);
			}

			double probabilityAttribute = totalAttributeCount / (double) numInstances;
			conditionalEntropy += probabilityAttribute * attributeEntropy;
		}

		return conditionalEntropy;
	}

	private Instances getSubset(Instances instances, Attribute attribute, int valueIndex) {
		Instances subset = new Instances(instances, 0);

		for (int i = 0; i < instances.numInstances(); i++) {
			if (instances.instance(i).value(attribute) == valueIndex) {
				subset.add(instances.instance(i));
			}
		}

		return subset;
	}

	private double getMajorityClass(Instances instances) {
		Map<Double, Integer> classCounts = new HashMap<>();

		for (int i = 0; i < instances.numInstances(); i++) {
			double classValue = instances.instance(i).classValue();
			classCounts.put(classValue, classCounts.getOrDefault(classValue, 0) + 1);
		}

		double majorityClass = -1;
		int maxCount = 0;

		for (double classValue : classCounts.keySet()) {
			int count = classCounts.get(classValue);
			if (count > maxCount) {
				majorityClass = classValue;
				maxCount = count;
			}
		}

		return majorityClass;
	}

	private double log2(double value) {
		return Math.log(value) / Math.log(2);
	}

	private abstract static class Node {
		private Attribute attribute;
		private Map<Double, Node> children;

		public Node(Attribute attribute) {
			this.attribute = attribute;
			this.children = new HashMap<>();
		}

		public Attribute getAttribute() {
			return attribute;
		}

		public boolean isLeaf() {
			return false;
		}

		public Node getChild(double value) {
			return children.get(value);
		}

		public void addChild(double value, Node child) {
			children.put(value, child);
		}

		public abstract double getLabel();
	}

	private static class InternalNode extends Node {
		private Attribute splitAttribute;

		public InternalNode(Attribute splitAttribute) {
			super(null); // Internal nodes have no attribute
			this.splitAttribute = splitAttribute;
		}

		@Override
		public double getLabel() {
			throw new UnsupportedOperationException("Internal nodes do not have labels.");
		}
	}

	private static class LeafNode extends Node {
		private double label;

		public LeafNode(double label) {
			super(null); // Leaf nodes have no attribute
			this.label = label;
		}

		@Override
		public boolean isLeaf() {
			return true;
		}

		@Override
		public double getLabel() {
			return label;
		}
	}

}
