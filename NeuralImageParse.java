package A3NeuralImageParse;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class NeuralImageParse {
	private class Record {
		private int[] input;
		private double[] output;

		private Record(int[] input, double[] output) {
			this.input = input;
			this.output = output;
		}
	}

	private int numberOfRecords;
	private int numberOfInputs;
	private int numberOfOutputs;

	private int numberOfMiddle;
	private int numberOfIterations;
	private double rate;

	private ArrayList<Record> records;

	private double[] input;
	private double[] middle;
	private double[] output;

	private double[] errorMiddle;
	private double[] errorOutput;

	private double[] thetaMiddle;
	private double[] thetaOutput;

	private double[][] matrixMiddle;
	private double[][] matrixOutput;

	public NeuralImageParse() {
		numberOfRecords = 0;
		numberOfInputs = 0;
		numberOfOutputs = 0;
		numberOfMiddle = 0;
		numberOfIterations = 0;
		rate = 0;

		records = null;
		input = null;
		middle = null;
		output = null;
		errorMiddle = null;
		errorOutput = null;
		thetaMiddle = null;
		thetaOutput = null;
		matrixMiddle = null;
		matrixOutput = null;

	}

	public void loadTrainingData(String trainingFile) throws IOException {
		Scanner file = new Scanner(new File(trainingFile));

		// read number of records, inputs, and outputs
		numberOfRecords = file.nextInt();
		numberOfInputs = file.nextInt();
		numberOfOutputs = file.nextInt();

		// create empty list of records
		records = new ArrayList<Record>();

		// for each training record
		for (int i = 0; i < numberOfRecords; i++) {
			// read inputs
			int[] input = new int[numberOfInputs];
			for (int j = 0; j < numberOfInputs; j++) {
				input[j] = file.nextInt();
			}

			// read outputs
			double[] output = new double[numberOfOutputs];
			for (int j = 0; j < numberOfOutputs; j++) {
				output[j] = file.nextInt();
			}

			// create record and add record to array
			Record record = new Record(input, output);
			records.add(record);
		}
		file.close();
	}

	public void setParameters(int numberOfMiddle, int numberOfIterations, int seed, double rate) {
		// set hidden nodes, iterations, and rate
		this.numberOfMiddle = numberOfMiddle;
		this.numberOfIterations = numberOfIterations;
		this.rate = rate;

		// create random gen
		Random random = new Random(seed);

		// create input and output arrays
		input = new double[numberOfInputs];
		middle = new double[numberOfMiddle];
		output = new double[numberOfOutputs];

		// create error arrays
		errorMiddle = new double[numberOfMiddle];
		errorOutput = new double[numberOfOutputs];

		// create theta arrays
		thetaMiddle = new double[numberOfMiddle];
		thetaOutput = new double[numberOfOutputs];

		// initialize thetas at hidden nodes
		for (int i = 0; i < numberOfMiddle; i++) {
			thetaMiddle[i] = 2 * random.nextDouble() - 1;
		}

		// initialize thetas at output nodes
		for (int i = 0; i < numberOfOutputs; i++) {
			thetaOutput[i] = 2 * random.nextDouble() - 1;
		}

		// initialize weights between input and hidden nodes
		matrixMiddle = new double[numberOfInputs][numberOfMiddle];
		for (int i = 0; i < numberOfInputs; i++) {
			for (int j = 0; j < numberOfMiddle; j++) {
				matrixMiddle[i][j] = 2 * random.nextDouble() - 1;
			}
		}

		// initialize weights between hidden and output nodes
		matrixOutput = new double[numberOfMiddle][numberOfOutputs];
		for (int i = 0; i < numberOfInputs; i++) {
			for (int j = 0; j < numberOfMiddle; j++) {
				matrixMiddle[i][j] = 2 * random.nextDouble() - 1;
			}
		}
	}

	public void train() {
		// repeat for the specified number of iterations
		for (int i = 0; i < numberOfIterations; i++) {
			// for each training record
			for (int j = 0; j < numberOfRecords; j++) {
				// calculate inputs and outputs
				forwardCalculation(records.get(j).input);
				// compute errors, update weights and thetas
				backwardCalculation(records.get(j).output);
			}
		}
	}

	private void forwardCalculation(int[] input2) {
		// feed inputs of record
		for (int i = 0; i < numberOfInputs; i++) {
			input[i] = input2[i];
		}
		// for each hidden node
		for (int i = 0; i < numberOfMiddle; i++) {
			double sum = 0;

			// compute input at hidden node
			for (int j = 0; j < numberOfInputs; j++) {
				sum += input[j] * matrixMiddle[j][i];
			}

			// add theta
			sum += thetaMiddle[i];

			// compute output at hidden node
			middle[i] = 1 / (1 + Math.exp(-sum));

		}

		// for each output node
		for (int i = 0; i < numberOfOutputs; i++) {
			double sum = 0;

			// compute input at output node
			for (int j = 0; j < numberOfMiddle; j++) {
				sum += middle[j] * matrixOutput[j][i];
			}

			// add theta
			sum += thetaOutput[i];

			// compute output at output node
			output[i] = 1 / (1 + Math.exp(-sum));
		}
	}

	private void backwardCalculation(double[] trainingOutput) {
		// compute error at every output node
		for (int i = 0; i < numberOfOutputs; i++) {
			errorOutput[i] = output[i] * (1 - output[i]) * (trainingOutput[i] - output[i]);
		}

		// computer error at every hidden node
		for (int i = 0; i < numberOfMiddle; i++) {
			double sum = 0;

			for (int j = 0; j < numberOfOutputs; j++) {
				sum += matrixOutput[i][j] * errorOutput[j];
			}

			errorMiddle[i] = middle[i] * (1 - middle[i]) * sum;
		}

		// update weights between hidden and output nodes
		for (int i = 0; i < numberOfMiddle; i++) {
			for (int j = 0; j < numberOfOutputs; j++) {
				matrixOutput[i][j] += rate * middle[i] * errorOutput[j];
			}
		}

		// update weights between input and hidden nodes
		for (int i = 0; i < numberOfInputs; i++) {
			for (int j = 0; j < numberOfMiddle; j++) {
				matrixMiddle[i][j] += rate * input[i] * errorMiddle[j];
			}
		}

		// update thetas at output nodes
		for (int i = 0; i < numberOfOutputs; i++) {
			thetaOutput[i] += rate * errorOutput[i];
		}

		// update thetas at hidden nodes
		for (int i = 0; i < numberOfMiddle; i++) {
			thetaMiddle[i] += rate * errorMiddle[i];
		}
	}

	private double[] test(int[] input) {
		// forward pass
		forwardCalculation(input);
		// back pass
		return output;
	}

	// reads from input file and writes to output file
	public double[][] testData(String inputFile, String outputFile) throws IOException {
		Scanner inFile = new Scanner(new File(inputFile));

		int numberOfRecords = inFile.nextInt();
		double[][] outputToProcess = new double[numberOfRecords][numberOfOutputs];

		// for each record
		for (int i = 0; i < numberOfRecords; i++) {
			int[] input = new int[numberOfInputs];

			// read input from input file
			for (int j = 0; j < numberOfInputs; j++) {
				input[j] = inFile.nextInt();
			}

			// find output using the neural net
			double[] output = test(input);

			for (int j = 0; j < numberOfOutputs; j++) {
				outputToProcess[i][j] = output[j];
			}
		}
		inFile.close();
		return outputToProcess;
	}

	// reads from input file and writes to output file
	public void validate(String validationFile) throws IOException {
		Scanner scanner = new Scanner(new File(validationFile));

		int numberOfRecords = scanner.nextInt();
		int numberOfInputs = scanner.nextInt();
		int numberOfOutputs = scanner.nextInt();

		System.out.println(numberOfRecords);
		// for each record
		for (int i = 0; i < numberOfRecords; i++) {
			// read inputs
			int[] input = new int[numberOfInputs];
			for (int j = 0; j < numberOfInputs; j++) {
				input[j] = scanner.nextInt();
			}

			// read actual outputs
			double[] expectedOutput = new double[numberOfOutputs];
			for (int j = 0; j < numberOfOutputs; j++) {
				expectedOutput[j] = scanner.nextDouble();
			}

			// find predicted output
			double[] predictedOutput = test(input);

			// write actual and predicted output
			for (int j = 0; j < numberOfOutputs; j++) {
				System.out.println("validation output from file: " + expectedOutput[j]);
				System.out.println("output from neural network:  " + predictedOutput[j]);
				System.out.println();
			}
		}
		scanner.close();
	}

	// method finds error between actual and predicted outputs
	private double computeError(double[] actualOutput, double[] predictedOutput) {
		double error = 0;

		// sum of squares of errors
		for (int i = 0; i < actualOutput.length; i++) {
			error += Math.pow(actualOutput[i] - predictedOutput[i], 2);
		}
		return Math.sqrt(error / actualOutput.length);

	}
}
