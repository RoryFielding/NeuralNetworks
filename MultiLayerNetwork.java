package uk.ac.reading.rfielding.NeuralNets;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;


/**
 * @author rfielding
 * This a multi layer network, comprising a hidden layer of neurons with sigmoid activation
 * Followed by another layer with linear/sigmoid activation, or be another multi layer network
 * A layer is defined as a set of neurons which have the same inputs
 */
public class MultiLayerNetwork extends SigmoidLayerNetwork {
	LinearLayerNetwork nextLayer;			// this is the next layer of neurons
	
	/**
	 * Constructor for neuron
	 * @param numIns	how many inputs there are (hence how many weights needed)
	 * @param numOuts	how many outputs there are (hence how many neurons needed)
	 * @param data		the data set used to train the network
	 * @param nextL		the next layer in the network
	 */
	public MultiLayerNetwork(int numIns, int numOuts, DataSet data, LinearLayerNetwork nextL) {
		super(numIns, numOuts, data);			// construct the current layer
		nextLayer = nextL;						// store link to next layer
	}
	/**
	 * calcOutputs of network
	 * @param nInputs	arraylist with the neuron inputs
	 * Calculates outputs of this layer and then of next layer
	 * get the outputs from super taking its inputs as an argument and gets the next layers outputs using the getoutput 
	 * function from this class
	 */
	protected void calcOutputs(ArrayList<Double> nInputs) {
		super.calcOutputs(nInputs); //Uses super to calc outputs using inputs
		nextLayer.calcOutputs(getOutputs()); //Gets next layer outputs using this classes outputs
		
	}
	/**
	 * depositOutputs of the output layer of the network to the data set
	 * @param ct	which item in the data set
	 * @param d		the data set
	 * 
	 */
	protected void depositOutputs (int ct, DataSet d) {
		/**Stores outputs with a counter and the next layers outputs**/
		nextLayer.depositOutputs(ct, d);
		
	}
	
	/**
	 * find the deltas in the whole network from the errors passed
	 * needs to find the deltas in the next layer, and then calculate them in this layer
	 *	@param errors in the output layer	
	 */
	protected void findDeltas(ArrayList<Double> errors) {
		/**Gets the next layers deltas and uses its deltas with the super finding deltas**/
		nextLayer.findDeltas(errors);
		super.findDeltas(nextLayer.weightedDeltas());
	}
	
	/**
	 * change all the weights in the network, in this layer and the next
	 * @param ins		array list of the inputs to the neuron
	 * @param learnRate	learning rate: change is learning rate * input * delta
	 * @param momentum	momentum constant : change is also momentun * change in weight last time
	 */
	protected void changeTheWeights(ArrayList<Double> ins, double learnRate, double momentum) {
		/**Changes the weights with arguments of inputs, learning rate and momentum in super class and
		 * then next layer changes the weights with the outputs, learning rate and momentum
		 */
		super.changeTheWeights(ins, learnRate, momentum);
		nextLayer.changeTheWeights(getOutputs(), learnRate, momentum);
	}	
	/**
	 * Load weights with the values in the array of strings wtsSplit
	 * @param wtsSplit
	 */
	protected void setWeights (String[] wtsSplit) {
		super.setWeights(wtsSplit);					// copy relevant weights in this layer
		nextLayer.setWeights(Arrays.copyOfRange(wtsSplit, weights.size(), wtsSplit.length));
				// copy remaining strings in wtsSplit and pass to next layer
	}
	/**
	 * Load the weights with random values
	 * @param rgen	random number generator
	 */
	public void setWeights (Random rgen) {
		super.setWeights(rgen);			// do so in this layer
		nextLayer.setWeights(rgen);		// and in next
	}
	/**
	 * return how many weights there are in the network
	 * @return returns the number of weights in super and in next layer
	 */
	public int numWeights() {
		
			numWeights = super.numWeights() + nextLayer.numWeights(); //numweights will equal supers number of weights and the next layers number of weights
			return numWeights;
	}
	/**
	 * return the weights in the whole network as a string
	 * @return returns the first layers weights and the next layers weights
	 */
	public String getWeights() {
		return  super.getWeights() +  nextLayer.getWeights(); //returns first layer and next layers weights
	}

	/**
	 * initialise network before running
	 */
	public void doInitialise() {
		super.doInitialise();					// initialise this layer 
		nextLayer.doInitialise();				// and then initialise next layer
	}
	
	/**
	 * function to test MLP on xor problem
	 */
	public static void TestXOR() {
		DataSet Xor = new DataSet("2 1 %.0f %.0f %.3f;x1 x2 XOR;0 0 0;0 1 1;1 0 1;1 1 0");
		MultiLayerNetwork MLN = new MultiLayerNetwork(2, 2, Xor, new SigmoidLayerNetwork(2, 1, Xor));
		MLN.setWeights("0.862518 -0.155797 0.282885 0.834986 -0.505997 -0.864449 0.036498 -0.430437 0.481210");
		MLN.doInitialise();
		System.out.println(MLN.doPresent());
		System.out.println("Weights " + MLN.getWeights());
		System.out.println(MLN.doLearn(1000, 0.4,  0.7));
		System.out.println(MLN.doPresent());
		System.out.println("Weights " + MLN.getWeights());
		System.out.println("Number of weights: " + MLN.numWeights());

	
	}
	/**
	 * function to test MLP on other non linear separable problem
	 */
	public static void TestOther() {
		DataSet Other = new DataSet("2 2 %.1f %.0f %.3f;0.1 1.2 1 0;0.7 1.8 1 0;0.8 1.6 1 0;1 0.8 0 0;"+
									 "0.3 0.5 1 1;0 0.2 1 1;-0.3 0.8 1 1;-0.5 -1.5 0 1;-1.5 -1.3 0 1");
	//	DataSet Other = new DataSet(DataSet.GetFile("other.txt"));
		MultiLayerNetwork MLN = new MultiLayerNetwork(2, 2, Other, new SigmoidLayerNetwork(2, 2, Other));
			MLN.computeNetwork(Other);
			MLN.doInitialise();
			System.out.println(MLN.doPresent());
			System.out.println("Weights " + MLN.getWeights());
			System.out.println(MLN.doLearn(2000,  0.5,  0.8));
			System.out.println(MLN.doPresent());
			System.out.println("Weights " + MLN.getWeights());
			
		
	}
	/**
	 * function to test MLP on other non linear separable problem using three layers
	 */
	public static void TestThree() {
	 DataSet Other = new DataSet(DataSet.GetFile("other.txt"));
	 MultiLayerNetwork MLN = new MultiLayerNetwork(2, 4, Other,
	new MultiLayerNetwork (4, 3, Other,
	new SigmoidLayerNetwork(3, 2, Other)) );
	MLN.computeNetwork(Other);
	MLN.doInitialise();
	 System.out.println(MLN.doPresent());
	 System.out.println("Weights " + MLN.getWeights());
	 System.out.println(MLN.doLearn(1000, 0.2, 0.6));
	 System.out.println(MLN.doPresent());
	 System.out.println("Weights " + MLN.getWeights());
	 } 
	/**
	 * @param args
	 */
	public static void main(String[] args) {
	//	TestXOR();				// test MLP on the XOR problem
	//	TestOther();			// test MLP on the other problem
		TestThree();
	}

}
