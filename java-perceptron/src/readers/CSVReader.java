package readers;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by u624 on 5/2/17.
 */
public class CSVReader {
    public static List<List<String>> readFile(Path filePath) {
        List<List<String>> result = null;
        try (BufferedReader bufferedReader = Files.newBufferedReader(filePath)) {
            result = bufferedReader.lines()
                    .map(s -> Arrays.asList(s.split("[,]")))
                    .collect(Collectors.toList());
        } catch (IOException e) {
            e.printStackTrace();
        }
        return result;
    }
}
