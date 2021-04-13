package example;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

/**
 * Created by u624 on 6/1/17.
 */
public class GenerateTrainingData {
    public static final int MAX_X = 300;
    public static final int MAX_Y = 200;

    public static void main(String[] args) {
        Random random = new Random(System.currentTimeMillis());
        try (BufferedWriter bufferedWriter = Files.newBufferedWriter(Paths.get("./resources/training_data.csv"))) {
            for (int i = 0; i < 100; i++) {
                bufferedWriter.write((-random.nextInt(MAX_X)) + "," + random.nextInt(MAX_Y) + ",-1\n");
            }
            for (int i = 0; i < 99; i++) {
                bufferedWriter.write((random.nextInt(MAX_X)) + "," + -random.nextInt(MAX_Y) + ",1\n");
            }
            bufferedWriter.write((random.nextInt(MAX_X)) + "," + -random.nextInt(MAX_Y) + ",1");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
