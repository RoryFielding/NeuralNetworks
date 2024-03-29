package uk.ac.reading.rfielding.NeuralNets;

/**
 * 
 */

import java.util.ArrayList;
import java.util.Random;

/**
 * @author rfielding
 *         This is a class for a layer of neurons with linear
 *         activation All such neurons share the same inputs.
 */

public class LinearLayerNetwork {
	
	/**
	 * data are arraylists of weights and the change in weights and of the
	 * outputs and deltas and also how many inputs, neurons and weights also has
	 * data set used with network
	 */
	
	protected ArrayList<Double> weights;
	protected ArrayList<Double> changeInWeights;
	protected ArrayList<Double> outputs;
	protected ArrayList<Double> deltas;
	protected int numInputs, numNeurons, numWeights;
	protected DataSet trainData;

	/**
	 * Constructor for neuron
	 * 
	 * @param numIns
	 *            how many inputs there are (hence how many weights needed)
	 * @param numOuts
	 *            how many outputs there are (hence how many neurons needed)
	 * @param data
	 *            the data set used to train the network
	 */
	
	public LinearLayerNetwork(int numIns, int numOuts, DataSet data) {
		numInputs = numIns; // store number inputs
		numNeurons = numOuts; // and of outputs in object
		numWeights = (numInputs + 1) * numNeurons; // for convenience calculate
													// number of weights
		// each neuron as numInputs + 1 weights (+1 because of bias weight)

		weights = new ArrayList<Double>(); // create array list for weights
		changeInWeights = new ArrayList<Double>(); // and for the change in
													// weights
		outputs = new ArrayList<Double>(); // create array list for outputs
		deltas = new ArrayList<Double>(); // and for the change in deltas
		for (int ct = 0; ct < numWeights; ct++) { // for each weight
			weights.add(0.0); // add next weight as 0
			changeInWeights.add(0.0); // add next change in weight, value 0
		}
		for (int ct = 0; ct < numNeurons; ct++) { // for each neuron
			outputs.add(0.0); // add a zero output
			deltas.add(0.0); // add a zero delta
		}
		trainData = data; // remember data set used for training
	}

	/**
	 * calcOutputs of neuron
	 * 
	 * @param nInputs
	 *            arraylist with the neuron inputs Calculates weighted sum being
	 *            weight(0) + inputs(0..n) * weights(1..n+1)
	 */
	
	protected void calcOutputs(ArrayList<Double> nInputs) {
		int wtIndex = 0; // used to index weights in order
		double output;
		for (int neuronct = 0; neuronct < numNeurons; neuronct++) {
			output = weights.get(wtIndex++); // start with bias weight( * 1)
			for (int inputct = 0; inputct < numInputs; inputct++)
				// for remaining weights
				output += nInputs.get(inputct) * weights.get(wtIndex++);
			// add weight*appropriate input and move to next weight
			outputs.set(neuronct, output); // set calculated output as the
											// neuron output
		}
	}

	/**
	 * depositOutputs to the given data set
	 * 
	 * @param ct
	 *            which item in the data set
	 * @param d
	 *            the data set
	 */
	
	protected void depositOutputs(int ct, DataSet d) {
		d.storeOutputs(ct, outputs); // just store outputs in data set
	}

	/**
	 * compute outputs of network by passing it each item in data set in turn,
	 * these outputs are put back into the data set
	 * 
	 * @param d
	 *            data set
	 */
	
	public void computeNetwork(DataSet d) {
		for (int ct = 0; ct < d.numInSet(); ct++) { // for each item in data set
			calcOutputs(d.getIns(ct)); // calculate output
			depositOutputs(ct, d); // and put in data set
		}
	}

	/**
	 * find deltas using the errors passed
	 *
	 * @param errors
	 */
	
	protected void findDeltas(ArrayList<Double> errors) {
		for (int i = 0; i < deltas.size(); i++) { // iterate over array list for
													// number of deltas
			deltas.set(i, errors.get(i)); // replace delta with specified error
		}

	}

	/**
	 * change all the weights in the layer of neurons
	 * 
	 * @param ins
	 *            array list of the inputs to the neuron layer
	 * @param learnRate
	 *            learning rate: change is learning rate * input * delta
	 * @param momentum
	 *            momentum constant : change is also momentun * change in weight
	 *            last time
	 */
	
	protected void changeTheWeights(ArrayList<Double> ins, double learnRate,
			double momentum) {
		double theIn;
		int ct = 0;
		for (int neuron = 0; neuron < numNeurons; neuron++) // first loop for
															// incrementing
															// between each
															// neuron
		{
			for (int wct = 0; wct < numInputs + 1; wct++) { // second loop for
															// weight counter
				if (wct == 0) {
					theIn = 1.0;
				} // if weight counter is equal to 0 input will be 0
				else {
					theIn = ins.get(wct - 1); // if weight counter no equal to
												// 0, input equal inputs with
												// weight counter -1
				}

				ct = (neuron * (numInputs + 1)) + wct; // counter is neuron x
														// numinputs +1 + other
														// count

				changeInWeights.set(ct, theIn * deltas.get(neuron) * learnRate
						+ changeInWeights.get(ct) * momentum);

				weights.set(ct, weights.get(ct) + changeInWeights.get(ct));
			}
		} // note sets theIn to either 1 or relevant input
	}

	/**
	 * adapt the network, by inputting each item from the data set in turn,
	 * calculating the output, the error and delta, and adjusting all the
	 * weights
	 * 
	 * @param d
	 *            data set
	 * @param learnRate
	 *            learning rate constant
	 * @param momentum
	 *            momentum constant
	 */
	
	public void adaptNetwork(DataSet d, double learnRate, double momentum) {
		for (int ct = 0; ct < d.numInSet(); ct++) { // for each item in set
			calcOutputs(d.getIns(ct)); // calc outputs
			depositOutputs(ct, d); // put in data set
			findDeltas(d.getErrors(ct)); // calc deltas, from the errors
			changeTheWeights(d.getIns(ct), learnRate, momentum);// change the
																// weights
		}
		d.addToSSELog();
	}

	/**
	 * return the array list containing the outputs of this layer of neurons
	 * 
	 * @return
	 */
	
	protected ArrayList<Double> getOutputs() {
		return outputs;
	}

	/**
	 * Calculate the errors in the previous layer, being the deltas in this
	 * layer * associated weights this is used in the back propagation algorithm
	 * There is one error for each neuron in previous layer, those neurons
	 * provide inputs to this layer
	 * 
	 * @return arraylist of errors
	 */
	
	public ArrayList<Double> weightedDeltas() {
		ArrayList<Double> wtDeltas = new ArrayList<Double>(); // create array
																// for answer
		for (int i = 0; i < numInputs; i++){ //for each neuron in the previous layer
			double res = 0.0;
			for (int j = 0; j < numNeurons; j++){ //ad the delta * weight (+1 bias)
				res += deltas.get(j) * weights.get(j * (numInputs +1) + i + 1);
			}
			wtDeltas.add(res);
		}
		return wtDeltas;
	}

	/**
	 * Load weights with the values in the array of strings wtsSplit
	 * 
	 * @param wtsSplit
	 */

	protected void setWeights(String[] wtsSplit) {
		for (int ct = 0; ct < weights.size(); ct++)
			weights.set(ct, Double.parseDouble(wtsSplit[ct]));
	} // for each item, set weight by converting string to double

	/**
	 * Load the weights with the values in the String wts
	 * 
	 * @param wts
	 */

	public void setWeights(String wts) {
		setWeights(wts.split(" ")); // split string into array of string and so
									// set weights
	}

	/**
	 * Load the weights with random values in range -1 to 1
	 * 
	 * @param rgen
	 *            random number generator
	 */

	public void setWeights(Random rgen) {
		for (int ct = 0; ct < weights.size(); ct++)
			weights.set(ct, 2.0 * rgen.nextDouble() - 1);
	}

	/**
	 * return how many weights there are in the neuron
	 * 
	 * @return
	 */

	public int numWeights() {
		return weights.size();
	}

	/**
	 * return the weights in the neuron as a string
	 * 
	 * @return the string
	 */

	public String getWeights() {
		String s = ""; // set string to empty
		for (int i = 0; i < weights.size(); i++)
			s = s + String.format("%.5f", weights.get(i)) + " ";
		return s; // return the result
	}

	/**
	 * initialise network before running
	 */
	
	public void doInitialise() {
		for (int ct = 0; ct < changeInWeights.size(); ct++)
			changeInWeights.set(ct, 0.0);
		// set the change in weights to be 0
		trainData.clearSSELog();
	}

	/**
	 * present the data to the network and return string describing result
	 * 
	 * @return
	 */
	
	public String doPresent() {
		computeNetwork(trainData);
		return trainData.toString(true, true) + "\nOver Set : "
				+ trainData.dataAnalysis() + "\n";
	}

	/**
	 * create string which says Epoch then adds the actual epoch in a fixed
	 * width field
	 * 
	 * @param epoch
	 * @return
	 */
	
	protected String addEpochString(int epoch) {
		return "Epoch " + String.format("%4d", epoch);
	}

	/**
	 * get network to learn for numEpochs
	 * 
	 * @param numEpochs
	 *            number of epochs to learn
	 * @param lRate
	 *            learning rate
	 * @param momentum
	 *            momentum
	 * @return String with data about learning eg SSEs at relevant epochs At
	 *         each epoch if numEpochs low, or do so at 10 of the epochs
	 */
	
	public String doLearn(int numEpochs, double lRate, double momentum) {
		int epochsSoFar = trainData.sizeSSELog(); // SSE log indicates how many
													// epochs so far
		String s = "";
		for (int ct = 1; ct <= numEpochs; ct++) { // for n epochs
			adaptNetwork(trainData, lRate, momentum); // present data and adapt
														// weights
			if (numEpochs < 20 || ct % (numEpochs / 10) == 0) // print
																// appropriate
																// number of
																// times
				s = s + addEpochString(ct + epochsSoFar) + " : "
						+ trainData.dataAnalysis() + "\n";
		} // Epoch, and SSE, and if appropriate % correctly classified
		return s;
	}

	public static void main(String[] args) {
		// Test network on example data set
		DataSet AndOrXor = new DataSet(
				"2 3 %.0f %.0f %.3f;x1 x2 AND OR XOR;0 0 0 0 0;0 1 0 1 1;1 0 0 1 1;1 1 1 1 0");
		LinearLayerNetwork LN = new LinearLayerNetwork(2, 3, AndOrXor);
		LN.setWeights("0.2 0.5 0.3 0.3 0.5 0.1 0.4 0.1 0.2");
		LN.doInitialise();
		System.out.println(LN.doPresent());
		System.out.println("Weights " + LN.getWeights());
		System.out.println(LN.doLearn(7, 0.1, 0.1));
		System.out.println(LN.doPresent());
		System.out.println("Weights " + LN.getWeights());

	}

}
