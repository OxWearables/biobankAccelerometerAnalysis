
import java.io.BufferedReader;
import java.io.FileReader;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;


/**
 * Calculates epoch summaries from an GENEActiv .bin file.
 */
public class GENEActivReader extends DeviceReader {

    /**
     * Read GENEA bin file pages, then call method to write epochs from raw
     * data. Epochs will be written to epochFileWriter.
     */
    public static void readGeneaEpochs(
        String accFile,
        String timeZone,
        int timeShift,
        EpochWriter epochWriter,
        Boolean verbose) {

        setTimeSettings(timeZone, timeShift);

        int fileHeaderSize = 59;
        int linesToAxesCalibration = 47;
        int pageHeaderSize = 9;
        // epoch creation support variables
        int[] errCounter = new int[] { 0 }; // store val if updated in other
                                            // method

        try {
            BufferedReader rawAccReader = new BufferedReader(new FileReader(accFile));
            // Read header to determine mfrGain and mfrOffset values
            double[] mfrGain = new double[3];
            int[] mfrOffset = new int[3];
            // memory size in pages
            int memSizePages = parseBinFileHeader(rawAccReader, fileHeaderSize, linesToAxesCalibration, mfrGain, mfrOffset);

            int pageCount = 1;
            String header;
            LocalDateTime blockTime = LocalDateTime.of(1999, 1, 1, 1, 1, 1);
            double temperature = 0.0;
            double sampleFreq = 0.0;
            String dataBlock;
            String timeFmtStr = "yyyy-MM-dd HH:mm:ss:SSS";
            DateTimeFormatter timeFmt = DateTimeFormatter.ofPattern(timeFmtStr);
            while ((readLine(rawAccReader)) != null) {
                // header: "Recorded Data" (0), serialCode (1), seq num (2),
                // pageTime (3), unassigned (4), temp (5), batteryVolt (6),
                // deviceStatus (7), sampleFreq (8),
                // Then: dataBlock (9)
                // line "page = readLine(..." above will read 1st header line
                // (c=0)
                for (int i = 1; i < pageHeaderSize; i++) {
                    try {
                        header = readLine(rawAccReader);
                        if (i == 3) {
                            blockTime = LocalDateTime.parse(header.split("Time:")[1], timeFmt);
                            blockTime = blockTime.plusMinutes(timeShift);

                            if (pageCount == 1) {
                                setSessionStart(blockTime);
                                System.out.println("Session start: " + sessionStart);
                            }
                        } else if (i == 5) {
                            temperature = Double.parseDouble(header.split(":")[1]);
                        } else if (i == 8) {
                            sampleFreq = Double.parseDouble(header.split(":")[1]);
                        }
                    } catch (Exception excep) {
                        System.err.println(excep.toString());
                        continue; // to keep reading sequence correct
                    }
                }

                // now process hex dataBlock
                dataBlock = readLine(rawAccReader);

                // raw reading values
                long t = 0;  // Unix time in millis
                int hexPosition = 0;
                int xRaw = 0;
                int yRaw = 0;
                int zRaw = 0;
                double x = 0.0;
                double y = 0.0;
                double z = 0.0;

                // loop through each reading in data block and check if it is
                // last in
                // epoch then write epoch summary to file
                // an epoch will have a start+end time, and fixed duration
                while (hexPosition < dataBlock.length()) {
                    try {
                        xRaw = getSignedIntFromHex(dataBlock, hexPosition, 3);
                        yRaw = getSignedIntFromHex(dataBlock, hexPosition + 3, 3);
                        zRaw = getSignedIntFromHex(dataBlock, hexPosition + 6, 3);
                    } catch (Exception excep) {
                        errCounter[0] += 1;
                        System.err.println("block err @ " + blockTime.toString() + ": " + excep.toString());
                        break; // rest of block/page could be corrupted
                    }
                    // todo *** read in light[36:46] (10 bits to signed int) and
                    // button[47] (bool) values...

                    // update values to calibrated measure (taken from GENEActiv
                    // manual)
                    x = (xRaw * 100.0d - mfrOffset[0]) / mfrGain[0];
                    y = (yRaw * 100.0d - mfrOffset[1]) / mfrGain[1];
                    z = (zRaw * 100.0d - mfrOffset[2]) / mfrGain[2];

                    t = zonedWithDSTCorrection(blockTime).toInstant().toEpochMilli();
                    epochWriter.newValues(t, x, y, z, temperature, errCounter);

                    hexPosition += 12;
                    blockTime = blockTime.plusNanos(secs2Nanos(1.0 / sampleFreq));
                }
                // option to provide status update to user...
                pageCount++;
                if (verbose && pageCount % 10000 == 0) {
                    System.out.print((pageCount * 100 / memSizePages) + "%\t");
                }
            }
            rawAccReader.close();
        } catch (Exception excep) {
            excep.printStackTrace(System.err);
            System.err.println("error reading/writing file " + accFile + ": " + excep.toString());
            System.exit(-2);
        }
    }


    /**
     * Replicates bin file header to epoch file, also calculates and returns
     * x/y/z gain/offset values along with number of pages of data in file bin
     * format described in GENEActiv manual ("Decoding .bin files", pg.27)
     * http://www.geneactiv.org/wp-content/uploads/2014/03/
     * geneactiv_instruction_manual_v1.2.pdf
     */
    private static int parseBinFileHeader(BufferedReader reader, int fileHeaderSize, int linesToAxesCalibration,
            double[] gainVals, int[] offsetVals) {
        // read first i lines in bin file to writer
        for (int i = 0; i < linesToAxesCalibration; i++) {
            readLine(reader);
        }
        // read axes calibration lines for gain and offset values
        // data like -> x gain:25548 \n x offset:574 ... Volts:300 \n Lux:800
        gainVals[0] = Double.parseDouble(readLine(reader).split(":")[1].trim()); // xGain
        offsetVals[0] = Integer.parseInt(readLine(reader).split(":")[1].trim()); // xOffset
        gainVals[1] = Double.parseDouble(readLine(reader).split(":")[1].trim()); // y
        offsetVals[1] = Integer.parseInt(readLine(reader).split(":")[1].trim()); // y
        gainVals[2] = Double.parseDouble(readLine(reader).split(":")[1].trim()); // z
        offsetVals[2] = Integer.parseInt(readLine(reader).split(":")[1].trim()); // z
        int volts = Integer.parseInt(readLine(reader).split(":")[1].trim()); // volts
        int lux = Integer.parseInt(readLine(reader).split(":")[1].trim()); // lux
        readLine(reader); // 9 blank
        readLine(reader); // 10 memory status header
        int memorySizePages = Integer.parseInt(readLine(reader).split(":")[1].trim()); // 11

        // ignore remaining header lines in bin file
        for (int i = 0; i < fileHeaderSize - linesToAxesCalibration - 11; i++) {
            readLine(reader);
        }
        return memorySizePages;
    }


    private static String readLine(BufferedReader fReader) {
        String line = "";
        try {
            line = fReader.readLine();
        } catch (Exception excep) {
            System.err.println(excep.toString());
        }
        return line;
    }

    
}
