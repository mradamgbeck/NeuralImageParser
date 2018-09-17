package A3NeuralImageParse;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;

class DataProcessor {
	int numberOfRecords;
	int numberOfInputs;
	int numberOfOutputs;
	int[][] postprocessedOutputs;

	public DataProcessor(String fileToProcess) throws FileNotFoundException {
		Scanner scanner = new Scanner(new File(fileToProcess));
		this.numberOfRecords = scanner.nextInt();
		this.numberOfInputs = scanner.nextInt();
		this.numberOfOutputs = scanner.nextInt();
		scanner.close();
	}

	public void postprocessOutputs(double[][] outputsToProcess) {
		this.postprocessedOutputs = new int[numberOfRecords][numberOfOutputs];
		// postprocess outputs
		for (int i = 0; i < numberOfOutputs; i++) {
			for (int j = 0; j < numberOfRecords; j++) {
				// scale output to its proper size
				if (outputsToProcess[j][i] >= 0.5) {
					postprocessedOutputs[j][i] = 1;
				} else {
					postprocessedOutputs[j][i] = 0;
				}
			}
		}
	}

	public void printTestOutputs(String finalOutputFile) throws IOException {
		PrintWriter outfile = new PrintWriter(new FileWriter(finalOutputFile));

		for (int i = 0; i < numberOfRecords; i++) {
			for (int j = 0; j < numberOfOutputs; j++) {
				outfile.print(postprocessedOutputs[i][j] + " ");
			}
			outfile.println();
		}
		outfile.close();
	}
}
