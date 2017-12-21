package uk.ac.reading.rfielding.NeuralNets;

import java.util.ArrayList;


/**
 * @author rfielding
 * This is a class for a layer of neurons with sigmoidal activation
 * All such neurons share the same inputs.
 */
public class SigmoidLayerNetwork extends LinearLayerNetwork {

	/**
	 * Constructor for neuron
	 * @param numIns	how many inputs there are (hence how many weights needed)
	 * @param numOuts	how many outputs there are (hence how many neurons needed)
	 * @param data		the data set used to train the network
	 */
	public SigmoidLayerNetwork(int numIns, int numOuts, DataSet data) {
		super(numIns, numOuts, data);
	}
	
	/**
	 * calcOutputs of neuron
	 * @param nInputs	arraylist with the neuron inputs
	 * Calcs Sigmoid(weighted sum) where weighted sum being weight(0) + inputs(0..n) * weights(1..n+1)
	 */
	protected void calcOutputs(ArrayList<Double> nInputs) {
		super.calcOutputs(nInputs); //inherit function from linearlayernetwork
		for (int i = 0; i < this.numNeurons; i++){ //iterate over number of neurons
			double sigmoid = 1/(1+Math.exp(-outputs.get(i))); //sigmoid function for each output
			outputs.set(i,  sigmoid); //set outputs 
		}
	}
	/**
	 * find deltas, being errors (which are passed to function) * outputs * (1 - outputs)
	 *	@param errors	
	 */
	protected void findDeltas(ArrayList<Double> errors) {
		for (int i = 0; i < errors.size(); i++){ //iterate over arraylist of errors
			deltas.set(i, errors.get(i)*outputs.get(i)*(1-outputs.get(i))); //calc deltas using formula
		}
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// test with and or xor
		DataSet AndOrXor = new DataSet("2 3 %.0f %.0f %.4f;x1 x2 AND OR XOR;0 0 0 0 0;0 1 0 1 1;1 0 0 1 1;1 1 1 1 0");
		SigmoidLayerNetwork SN = new SigmoidLayerNetwork(2, 3, AndOrXor);
		SN.setWeights("0.2 0.5 0.3 0.3 0.5 0.1 0.4 0.1 0.2");
		SN.doInitialise();
		System.out.println(SN.doPresent());
		System.out.println("Weights " + SN.getWeights());
		System.out.println(SN.doLearn(1000,  0.15,  0.4));
		System.out.println(SN.doPresent());
		System.out.println("Weights " + SN.getWeights());

	}

}
