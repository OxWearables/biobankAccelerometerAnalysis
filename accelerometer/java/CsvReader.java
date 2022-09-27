
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.Reader;
import java.io.InputStreamReader;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * Calculates epoch summaries from an AX3 .CWA file.
 */
public class CsvReader extends DeviceReader {

    private static int NO_TEMPERATURE = -99;

    public static void readCSVEpochs(
            String accFile,
            EpochWriter epochWriter,
            int csvStartRow,
            List<Integer> csvTimeXYZTempColsIndex,
            DateTimeFormatter csvTimeFormat,
            Boolean verbose) {

        BufferedReader accReader;
        try {
            // check if .csv or .csv.gz, then setup reader apprioriately
            if (accFile.toLowerCase().endsWith(".csv.gz")){
                FileInputStream accStream = new FileInputStream(accFile);
                GZIPInputStream gzipStream = new GZIPInputStream(accStream);
                Reader decoder = new InputStreamReader(gzipStream);
                accReader = new BufferedReader(decoder);
            } else{ // i.e. endsWith(".csv")
                accReader = new BufferedReader(new FileReader(accFile));
            }

            String line = "";
            int lineNumber = 0;
            String csvSplitBy = ",";
            int timeCol = csvTimeXYZTempColsIndex.get(0);
            int xCol = csvTimeXYZTempColsIndex.get(1);
            int yCol = csvTimeXYZTempColsIndex.get(2);
            int zCol = csvTimeXYZTempColsIndex.get(3);
            int temperatureCol = NO_TEMPERATURE;
            if (csvTimeXYZTempColsIndex.size() == 5){
                temperatureCol = csvTimeXYZTempColsIndex.get(4);
            }

            String[] cols;
            long time = Long.MIN_VALUE;
            double x;
            double y;
            double z;
            double temperature = 0;
            while (true) {
                line = accReader.readLine();

                // skip unto we reach appropriate start line number
                if (lineNumber < csvStartRow){
                    lineNumber++;
                    continue;
                }
                if (line==null) {
                    accReader.close();
                    break;
                }

                // read csv line
                cols = line.split(csvSplitBy);
                time = getEpochMillis(LocalDateTime.parse(cols[timeCol],
                        csvTimeFormat));
                x = Double.parseDouble(cols[xCol]);
                y = Double.parseDouble(cols[yCol]);
                z = Double.parseDouble(cols[zCol]);
                if (temperatureCol != NO_TEMPERATURE){
                    temperature = Double.parseDouble(cols[temperatureCol]);
                }
                epochWriter.newValues(time, x, y, z, temperature, new int[] {0});
            }

        } catch (Exception excep) {
            excep.printStackTrace(System.err);
            System.err.println("error reading/writing file " + accFile + ": "
                 + excep.toString());
            System.exit(-2);
        }
    }


}
