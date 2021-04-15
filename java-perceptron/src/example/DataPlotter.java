package example;

import readers.CSVReader;

import javax.swing.*;
import java.awt.*;
import java.nio.file.Paths;
import java.util.List;

import static example.ClassifierPlotterExample.TRAINING_DATA_FILE;

/**
 * Created by u624 on 6/1/17.
 */
public class DataPlotter extends JFrame {
    private static final String JFRAME_TITLE = "Data Plot";

    private DataPlotter() {
        setLayout(new BorderLayout());
        add(new Graph(), BorderLayout.CENTER);
    }

    public static void main(String[] args) {
        JFrame jFrame = new DataPlotter();
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
            CSVReader.readFile(Paths.get(TRAINING_DATA_FILE))
                    .forEach(row -> drawSolidPoint(graphics2D, row));
        }

        private void drawSolidPoint(Graphics2D graphics2D, List<String> row) {
            int pointX = Integer.parseInt(row.get(0)) + GenerateTrainingData.MAX_X;
            int pointY = GenerateTrainingData.MAX_Y - Integer.parseInt(row.get(1));
            if (Integer.parseInt(row.get(2)) == 1) {
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
