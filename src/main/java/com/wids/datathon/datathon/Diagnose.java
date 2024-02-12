package com.wids.datathon.datathon;

import java.io.File;
import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVSaver;

public class Diagnose {
    public static void main(String[] args) {
        try {
            // Load training data
        	
        	// Load training data
            Instances trainingData = BreastCancerDiagnosisPredictor.loadCSV("training.csv",true);

            // Train the classifier
            BreastCancerDiagnosisPredictor predictor = new BreastCancerDiagnosisPredictor();
            predictor.buildClassifier(trainingData);

            // Load test data
            Instances testData = BreastCancerDiagnosisPredictor.loadCSV("test.csv",false);

            // Create an attribute for predictions
            Attribute predictionAttribute = new Attribute("DiagPeriodL90D");

            // Create a new Instances object to store predictions
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(trainingData.attribute("﻿patient_id"));
            attributes.add(predictionAttribute);

            Instances predictions = new Instances("Predictions", attributes, 0);

            // Make predictions on test data and add to the predictions Instances
            for (int i = 0; i < testData.numInstances(); i++) {
                Instance testInstance = testData.instance(i);
                double prediction = predictor.classifyInstance(testInstance);

                // Create a DenseInstance instead of using the Instance constructor
                DenseInstance predictionInstance = new DenseInstance(2);
                predictionInstance.setDataset(predictions);

                // Set values for the attributes
                predictionInstance.setValue(testData.attribute("﻿patient_id"), testInstance.value(testData.attribute("﻿patient_id")));
                predictionInstance.setValue(predictionAttribute, prediction);

                predictions.add(predictionInstance);
            }

            // Save predictions to CSV
            CSVSaver csvSaver = new CSVSaver();
            csvSaver.setInstances(predictions);
         // Check if the file already exists and delete it
            File submissionFile = new File("submission.csv");
            if (submissionFile.exists()) {
                if (submissionFile.delete()) {
                    System.out.println("Existing submission.csv deleted.");
                } else {
                    System.err.println("Failed to delete existing submission.csv.");
                }
            }

            csvSaver.setFile(new File("submission.csv"));
            csvSaver.writeBatch();

            System.out.println("Predictions saved to submission.csv");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}


