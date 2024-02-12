package com.wids.datathon.datathon;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToNominal;

public class BreastCancerDiagnosisPredictor implements Classifier  {
	
	private CustomDecisionTree classifier;
    
    public static Instances loadCSV(String filePath, boolean isTraining) throws IOException {
        InputStream inputStream = new FileInputStream(filePath);
        CSVLoader loader = new CSVLoader();
        loader.setSource(inputStream);
        Instances data = loader.getDataSet();

        // Check if 'DiagPeriodL90D' is a valid attribute in the dataset
        Attribute classAttribute1 = data.attribute("﻿patient_id");
        Attribute classAttribute = data.attribute("DiagPeriodL90D");

        if (isTraining) {
            // For training data, ensure that the class attribute is present
            if (classAttribute == null) {
                throw new IllegalArgumentException("Class attribute 'DiagPeriodL90D' not found in the training dataset.");
            }
            // Set the class index for the training data
            data.setClassIndex(classAttribute1.index());
         } else {
            // For test data, do not set the class index
            if (classAttribute != null) {
                System.out.println("Warning: 'DiagPeriodL90D' attribute found in test data. Ignoring it.");
            }
            
        }
        data.setClassIndex(classAttribute1.index());
        inputStream.close();
        return data;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // Assuming 'DiagPeriodL90D' is the class attribute
        data.setClassIndex(data.attribute("DiagPeriodL90D").index());

        // Convert string attributes to nominal
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);
            if (attribute.isString()) {
                StringToNominal stringToNominalFilter = new StringToNominal();
                stringToNominalFilter.setAttributeRange(Integer.toString(i + 1));
                stringToNominalFilter.setInputFormat(data);
                data = Filter.useFilter(data, stringToNominalFilter);
            }
        }

        // Convert numeric class attribute to nominal
        if (data.attribute("DiagPeriodL90D").isNumeric()) {
            NumericToNominal classToNominalFilter = new NumericToNominal();
            classToNominalFilter.setAttributeIndices("last");
            classToNominalFilter.setInputFormat(data);
            data = Filter.useFilter(data, classToNominalFilter);
        }

        // Instantiate the classifier (you can replace J48 with your preferred algorithm)
       // classifier = new J48();
        classifier =  new CustomDecisionTree();

        // Build the classifier using the training data
        classifier.buildClassifier(data);
    }
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        // Ensure that the classifier has been trained
        if (classifier == null) {
            throw new IllegalStateException("The classifier has not been trained yet.");
        }

        // Make a prediction using the trained classifier
        return classifier.classifyInstance(instance);
    }

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}
}

