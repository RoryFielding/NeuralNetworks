package uk.ac.reading.rfielding.NeuralNets;
/**
 * 
 */

import java.util.ArrayList;
import java.util.Random;

/**
 * @author rfielding
 *
 */
public class SigmoidNeuron extends LinearNeuron {
	/**
	 * Constructor for neuron with Sigmoid Activation
	 * @param numIns	number of inputs
	 * @param data		data set to be used
	 */
	public SigmoidNeuron(int numIns, DataSet data) {
		super(numIns, data);
	}
	
	/**
	 * calcOutput of neuron
	 * @param nInputs	arraylist with the neuron inputs
	 */
	protected void calcOutput(ArrayList<Double> nInputs) {
		super.calcOutput(nInputs);					// calc weighted sum of inputs
		output = 1.0/(1.0 + Math.exp(-output));		// convert to sigmoid(weightedsum)
	}
	
	/**
	 * find delta
	 *	@param error	compute the delta from the error passed
	 */
	protected void findDelta(double error) {
		delta = error * output * (1.0 - output);
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// Simple main to test class
		DataSet AndData = new DataSet("2 1 %.0f %.0f %.2f;x1 x2 AND;0 0 0;0 1 0;1 0 0;1 1 1");
		SigmoidNeuron SN = new SigmoidNeuron(2, AndData);
		Random rgen = new Random();
		SN.setWeights(rgen);
		System.out.print(SN.doPresent());
		System.out.print(SN.doLearn(1000, 0.2, 0.1));
		System.out.print(SN.doPresent());

	}

}
