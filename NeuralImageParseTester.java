package A3NeuralImageParse;

import java.io.IOException;

public class NeuralImageParseTester {
	public static void main(String[] args) throws IOException {
		String trainingFile = "C:\\Users\\a\\Desktop\\Class\\COSC461 Heuristics\\src\\A3NeuralImageParse\\image_train.txt";
		String testFile = "C:\\Users\\a\\Desktop\\Class\\COSC461 Heuristics\\src\\A3NeuralImageParse\\image_test.txt";
		String validationFile = "C:\\Users\\a\\Desktop\\Class\\COSC461 Heuristics\\src\\A3NeuralImageParse\\image_validate.txt";
		String finalOutputFile = "C:\\Users\\a\\Desktop\\Class\\COSC461 Heuristics\\src\\A3NeuralImageParse\\image_out.txt";

		// create network
		NeuralImageParse network = new NeuralImageParse();

		// data post processor setup

		DataProcessor processor = new DataProcessor(testFile);
		network.loadTrainingData(trainingFile);

		// set params of network
		network.setParameters(175, 250, 54678, 0.5);

		// train network
		network.train();

		// validate
		network.validate(validationFile);

		// test network
		double[][] outputFromNetwork = network.testData(testFile, finalOutputFile);

		// post process and print test data
		processor.postprocessOutputs(outputFromNetwork);
		processor.printTestOutputs(finalOutputFile);
	}
}
