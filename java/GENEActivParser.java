
//BSD 2-Clause (c) 2014: A.Doherty (Oxford), D.Jackson, N.Hammerla (Newcastle)
import java.io.BufferedReader;
import java.io.FileReader;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.concurrent.TimeUnit;
import java.time.LocalDateTime;
import java.time.ZoneOffset;


public class GENEActivParser {

    final private static int EXIT_SUCCESS = 0;
    final private static int EXIT_FAILURE = 1;
    private static final LinkedHashMap<String, String> ITEM_NAMES_AND_TYPES = getItemNamesAndTypes();


    public static int parse(
        String accFile,
        String outFile,
        boolean verbose) {

        int fileHeaderSize = 59;
        int linesToAxesCalibration = 47;
        int pageHeaderSize = 9;
        int[] errCounter = new int[] { 0 };

        NpyWriter writer = new NpyWriter(outFile, ITEM_NAMES_AND_TYPES);

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
                int hexPosition = 0;
                int xRaw = 0;
                int yRaw = 0;
                int zRaw = 0;
                double x = 0.0;
                double y = 0.0;
                double z = 0.0;

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

                    try {
                        writer.write(toItems(getEpochMillis(blockTime), x, y, z, temperature));
                    } catch (Exception e) {
                        System.err.println("Line write error: " + e.toString());
                    }

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
            return EXIT_FAILURE;
        } finally {
            try{
                writer.close();
            } catch (Exception e) {
                // ignore
            }
        }

        return EXIT_SUCCESS;

    }


    /**
     * Replicates bin file header, also calculates and returns
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
        gainVals[0] = Double.parseDouble(readLine(reader).split(":")[1]); // xGain
        offsetVals[0] = Integer.parseInt(readLine(reader).split(":")[1]); // xOffset
        gainVals[1] = Double.parseDouble(readLine(reader).split(":")[1]); // y
        offsetVals[1] = Integer.parseInt(readLine(reader).split(":")[1]); // y
        gainVals[2] = Double.parseDouble(readLine(reader).split(":")[1]); // z
        offsetVals[2] = Integer.parseInt(readLine(reader).split(":")[1]); // z
        int volts = Integer.parseInt(readLine(reader).split(":")[1]); // volts
        int lux = Integer.parseInt(readLine(reader).split(":")[1]); // lux
        readLine(reader); // 9 blank
        readLine(reader); // 10 memory status header
        int memorySizePages = Integer.parseInt(readLine(reader).split(":")[1]); // 11

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


    private static int getSignedIntFromHex(String dataBlock, int startPos, int length) {
        // input hex base is 16
        int rawVal = Integer.parseInt(dataBlock.substring(startPos, startPos + length), 16);
        int unsignedLimit = 4096; // 2^[length*4] #i.e. 3 hexBytes (12 bits)
                                    // limit = 4096
        int signedLimit = 2048; // 2^[length*(4-1)] #i.e. 3 hexBytes - 1 bit (11
                                // bits) limit = 2048
        if (rawVal > signedLimit) {
            rawVal = rawVal - unsignedLimit;
        }
        return rawVal;
    }


    // Convert LocalDateTime to epoch milliseconds (from 1970 epoch)
    private static long getEpochMillis(LocalDateTime date) {
        return date.toInstant(ZoneOffset.UTC).toEpochMilli();
    }


    private static long secs2Nanos(double num) {
        return (long) (TimeUnit.SECONDS.toNanos(1) * num);
    }


    private static HashMap<String, Object> toItems(long t, double x, double y, double z, double temperature) {
        HashMap<String, Object> items = new HashMap<String, Object>();
        items.put("time", t);
        items.put("x", x);
        items.put("y", y);
        items.put("z", z);
        items.put("T", temperature);
        // items.put("lux", light);
        return items;
    }


    private static LinkedHashMap<String, String> getItemNamesAndTypes() {
        LinkedHashMap<String, String> itemNamesAndTypes = new LinkedHashMap<String, String>();
        itemNamesAndTypes.put("time", "Long");
        itemNamesAndTypes.put("x", "Double");
        itemNamesAndTypes.put("y", "Double");
        itemNamesAndTypes.put("z", "Double");
        itemNamesAndTypes.put("T", "Double");
        // itemNamesAndTypes.put("lux", "Integer");
        return itemNamesAndTypes;
    }
    

}