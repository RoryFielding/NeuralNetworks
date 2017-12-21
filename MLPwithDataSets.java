package uk.ac.reading.rfielding.NeuralNets;

import java.util.ArrayList;

/**
 * @author rfielding
 * Class of a multi layer perceptron network with training, unseen and validation data sets
 * MLP has hidden layer of sigmnoidally activated neurons and then output layer(s)
 * Such a network can learn using the training set, and be tested on the unseen set
 * In addition, it can use the validation set to decide when to stop learning
 */
public class MLPwithDataSets extends MultiLayerNetwork {

	private ArrayList<Double> tempSSE; //temporarily storing the SSE
	private double sumSSE; // sum of the SSE
	private double prevSum; // previous sum of the SSE
	private int epochCounter; //counter that increments with each epoch 
	protected DataSet unseenData;			// unseen data set
	protected DataSet validationData;		// validation set : is set to null if that set is not being used
	
	/**
	 * Constructor for the MLP
	 * @param numIns			number of inputs	of hidden layer
	 * @param numOuts			number of outputs	of hidden layer
	 * @param data				training data set used
	 * @param nextL				next layer		
	 * @param unseen			unseen data set
	 * @param valid				validation data set
	 */
	MLPwithDataSets (int numIns, int numOuts, DataSet data, LinearLayerNetwork nextL,
						DataSet unseen, DataSet valid) {
		super(numIns, numOuts, data, nextL);	// create the MLP
												// and store the data sets
		unseenData = unseen;
		validationData = valid;
	}

	/** 
	 * initialise network before learning ...
	 */
	public void doInitialise() {
		super.doInitialise();
		sumSSE = 0; // Set sum of SSE's to 0
		prevSum = Double.MAX_VALUE; // Sets previous valid SSE to high value
		epochCounter = 0; //Set epoch counter to 0 to increment from
	}
	/**
	 * present the data to the set and return a String describing results
	 * Here it returns the performance when the training, unseen (and if available) validation
	 * sets are passed - typically responding with SSE and if appropriate % correct classification
	 */
	public String doPresent() {
		String S;
		computeNetwork(trainData);
		S = "Train: " +  trainData.dataAnalysis();
		computeNetwork(unseenData);
		S = S + " Unseen: " + unseenData.dataAnalysis();
		if (validationData != null) {
			computeNetwork(validationData);
			S = S + " Valid: " + validationData.dataAnalysis();
		}
		return S;
	}

	/**
	 * learn training data, printing SSE at 10 of the epochs, evenly spaced
	 * if a validation set available, learning stops when SSE on validation set rises
	 * this check is done by summing SSE over 10 epochs
	 * @param numEpochs		number of epochs
	 * @param lRate			learning rate
	 * @param momentum		momentum
	 * @return				String with data about learning eg SSEs at relevant epoch
	 */
	public String doLearn (int numEpochs, double lRate, double momentum) {
		String s = "";
		if (validationData==null) 
		{	s = super.doLearn(numEpochs, lRate, momentum); 
		}
				// if no validation set, just use normal doLearn
	else {
		for (int epochCT = 1; epochCT <= numEpochs; epochCT++) //Counter that increments through the number of epochs
		{
			epochCounter++; //Increments
			adaptNetwork(trainData, lRate, momentum); //Adapt the network to the training data
			super.computeNetwork(validationData); //Pass validation set to the network
			validationData.addToSSELog(); //Adds the SSE of the validation data set to its SSE Log
			tempSSE = validationData.getSSE(); //Store the validation data in temp arraylist.
			sumSSE += tempSSE.get(0);//Add the last SSE to the SSE sum
			if((epochCT % 10) == 0) { //For every 10th epoch
				s = s + addEpochString(epochCounter) + " : " +  //It stops learning and adds to return string info that learning has stopped at this epoch
						 trainData.dataAnalysis()+"\n"; 
				if (sumSSE > prevSum) { //If the sum of the SSEs has risen
					s = s + addEpochString(epochCounter ) + " : " +  //Stop learning and adds to return string info that learning has completed at this epoch
							 trainData.dataAnalysis()+ "The network has learned this problem" + "\n"; 
					epochCT = numEpochs; //Set the counter to max
				}
				else 
				{
					prevSum = sumSSE; //Else remembers the sum of SSEs 
					sumSSE=0; //Set the sum to 0
				}
			}
		}
	}
	return s; //Returns the string.
}
}
