
//BSD 2-Clause (c) 2014: A.Doherty (Oxford), D.Jackson, N.Hammerla (Newcastle)
import java.io.BufferedReader;
import java.io.FileReader;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;

/**
 * Calculates epoch summaries from an AX3 .CWA file.
 */
public class CsvReader extends DeviceReader {


    public static void readCSVEpochs(
            String accFile,
            EpochWriter epochWriter,
            LocalDateTime startTime,
            double csvSampleRate,
            DateTimeFormatter csvTimeFormat,
            int csvStartRow,
            List<Integer> csvXYZTCols) {

        try {
            BufferedReader accStream =  new BufferedReader(new FileReader(accFile));
            String line = "";
            int lineNumber = 0;
            String csvSplitBy = ",";
            int numColsRequired = Collections.max(csvXYZTCols);
            System.out.println("This is a special .csv reading version made for Alex-Rowlands dataset!\n"
                                + "parsing .csv file using:\n"
                                + (startTime!=null ? "csvStartTime = " + startTime.toString() + "\n"
                                + "csvSampleRate = " + csvSampleRate + "Hz" : "csvTimeFormat = " + csvTimeFormat.toString())+"\n"
                                + "minimum number of columns per row: " + numColsRequired);
            int xCol = csvXYZTCols.get(0);
            int yCol = csvXYZTCols.get(1);
            int zCol = csvXYZTCols.get(2);
            boolean readTime = false;
            long time = Long.MIN_VALUE;
            if (csvXYZTCols.size()>3) {
                readTime = true;
            } else {
                time = getEpochMillis(startTime);
            }

            while (true) {
                line = accStream.readLine();
                if (lineNumber++ <= csvStartRow) continue;

                if (line==null) {
                    accStream.close();
                    break;
                    // uncomment to loop forever (useful for testing performance)
                    // accStream =  new BufferedReader(new FileReader(accFile));
                    // continue;
                }
                String[] cols = line.split(csvSplitBy);
                if (cols.length>numColsRequired) {
                        if (readTime) {
                            String timeStr = "";
                            for(int i=3; i<csvXYZTCols.size(); i++) {
                                timeStr += cols[csvXYZTCols.get(i)];
                                if (i!=csvXYZTCols.size()-1) {
                                    timeStr += ",";
                                }
                            }
                            time = getEpochMillis(LocalDateTime.parse(timeStr, csvTimeFormat));
                            // System.out.println(epochMillisToLocalDateTime(time).toString() + " - " + timeStr);
                        }

                        double x = Double.parseDouble(cols[xCol]);
                        double y = Double.parseDouble(cols[yCol]);
                        double z = Double.parseDouble(cols[zCol]);
                        epochWriter.newValues(time, x, y, z, 0, new int[] {0});
                        if (!readTime) {
                            time += 1000/csvSampleRate;
                        }
                } else {
                    System.err.println(".csv line " + lineNumber + " had too few columns :\n" + line);
                }
            }

        } catch (Exception excep) {
            excep.printStackTrace(System.err);
            System.err.println("error reading/writing file " + accFile + ": " + excep.toString());
            System.exit(-2);
        }
    }


}