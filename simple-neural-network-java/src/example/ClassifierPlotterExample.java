package example;

import javax.swing.*;
import java.awt.*;
import java.nio.file.Paths;
import java.util.Random;

/**
 * Created by u624 on 6/1/17.
 */
public class ClassifierPlotterExample extends JFrame {
    public static final String TRAINING_DATA_FILE = "./resources/training_data.csv";
    private static final String JFRAME_TITLE = "Perceptron Classifier Example";
    private PerceptronClassifierExample perceptronClassifierExample = new PerceptronClassifierExample();

    private ClassifierPlotterExample() {
        perceptronClassifierExample.train(Paths.get(TRAINING_DATA_FILE));
        setLayout(new BorderLayout());
        add(new Graph(), BorderLayout.CENTER);
    }

    public static void main(String[] args) {
        JFrame jFrame = new ClassifierPlotterExample();
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jFrame.setTitle(JFRAME_TITLE);
        jFrame.setSize(GenerateTrainingData.MAX_X * 2, GenerateTrainingData.MAX_Y * 2);
        jFrame.setLocationRelativeTo(null);
        jFrame.setVisible(true);
    }

    class Graph extends JPanel {
        protected void paintComponent(Graphics graphics) {
            super.paintComponent(graphics);

            Graphics2D graphics2D = (Graphics2D) graphics;
            Dimension dimension = getSize();
            Random random = new Random(System.currentTimeMillis());

            for (int i = 0; i < 1000; i++) {
                drawRandomPoint(graphics2D, random);
            }
        }

        private void drawRandomPoint(Graphics2D graphics2D, Random random) {
            int x = GenerateTrainingData.MAX_X - random.nextInt(GenerateTrainingData.MAX_X) * 2;
            int y = GenerateTrainingData.MAX_Y - random.nextInt(GenerateTrainingData.MAX_Y) * 2;
            drawSolidPoint(graphics2D, x, y);
        }

        private void drawSolidPoint(Graphics2D graphics2D, int x, int y) {
            int pointX = x + GenerateTrainingData.MAX_X;
            int pointY = GenerateTrainingData.MAX_Y - y;
            if (perceptronClassifierExample.classify(x, y) == 1) {
                graphics2D.setColor(Color.red);
                drawPoint(graphics2D, pointX, pointY);
            } else {
                graphics2D.setColor(Color.blue);
                drawPoint(graphics2D, pointX, pointY);
            }
        }

        private void drawPoint(Graphics2D graphics2D, int pointX, int pointY) {
            graphics2D.fillOval(pointX, pointY, 5, 5);
        }
    }
}
