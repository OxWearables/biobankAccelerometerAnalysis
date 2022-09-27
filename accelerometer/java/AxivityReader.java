
import java.io.FileInputStream;
import java.nio.ByteOrder;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.zone.ZoneRules;
import java.util.zip.GZIPInputStream;


/**
 * Calculates epoch summaries from an AX3 .CWA file.
 */
public class AxivityReader extends DeviceReader {

    /**
     * Read and process Axivity CWA file. Setup file reading infrastructure
     * and then call readCwaBuffer() method
    **/
    public static void readCwaEpochs(
        String accFile,
        String timeZone,
        int timeShift,
        EpochWriter epochWriter,
        Boolean verbose) {

        setTimeSettings(timeZone, timeShift);

        int[] errCounter = new int[] { 0 }; // store val if updated in other
                                            // method
        // Inter-block timstamp tracking
        LocalDateTime[] lastBlockTime = { null };
        int[] lastBlockTimeIndex = { 0 };

        // data block support variables
        String header = "";

        int bufSize = 512;
        ByteBuffer buf = ByteBuffer.allocate(bufSize);
        try ( FileInputStream accStream = new FileInputStream(accFile); ) {
            FileChannel rawAccReader = accStream.getChannel();

            // now read every page in CWA file
            int pageCount = 0;
            long memSizePages = rawAccReader.size() / bufSize;
            boolean USE_PRECISE_TIME = true; // true uses block fractional time
                                                // and
                                                // interpolates timestamp
                                                // between blocks.
            while (rawAccReader.read(buf) != -1) {
                readCwaBuffer(buf,
                    USE_PRECISE_TIME, lastBlockTime, lastBlockTimeIndex, header,
                    errCounter, epochWriter);
                buf.clear();
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
     * Read and process Axivity CWA.gz gzipped file. Setup file reading
     * infrastructure and then call readCwaBuffer() method
    **/
    public static void readCwaGzEpochs(
        String accFile,
        String timeZone,
        int timeShift,
        EpochWriter epochWriter,
        Boolean verbose) {

        setTimeSettings(timeZone, timeShift);

        int[] errCounter = new int[] { 0 }; // store val if updated in other
                                            // method
        // Inter-block timstamp tracking
        LocalDateTime[] lastBlockTime = { null };
        int[] lastBlockTimeIndex = { 0 };

        // data block support variables
        String header = "";

        int bufSize = 512;
        ByteBuffer buf = ByteBuffer.allocate(bufSize);
        try ( FileInputStream accStream = new FileInputStream(accFile); ) {
            GZIPInputStream in = new GZIPInputStream(accStream);
            ReadableByteChannel rawAccReader = Channels.newChannel(in);

            // now read every page in CWA file
            int pageCount = 0;
            long memSizePages = getUncompressedSizeofGzipFile(accFile) / bufSize;
            boolean USE_PRECISE_TIME = true; // true uses block fractional time
                                                // and
                                                // interpolates timestamp
                                                // between blocks.
            while (rawAccReader.read(buf) != -1) {
                readCwaBuffer(buf,
                    USE_PRECISE_TIME, lastBlockTime, lastBlockTimeIndex, header,
                    errCounter, epochWriter);
                buf.clear();
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
     * Read Axivity CWA file, then call method to write epochs from raw data.
     * Epochs will be written to epochFileWriter.
     * Read data block HEX values, store each raw reading, then continually test
     * if an epoch of data has been collected or not. Finally, write each epoch
     * to epochFileWriter. CWA format is described at:
     * https://github.com/digitalinteraction/openmovement/blob/master/Downloads/AX3/AX3-CWA-Format.txt
    **/
    private static void readCwaBuffer(ByteBuffer buf,
        boolean USE_PRECISE_TIME,
        LocalDateTime[] lastBlockTime, int[] lastBlockTimeIndex, String header,
        int[] errCounter, EpochWriter epochWriter)
    {
        buf.flip();
        buf.order(ByteOrder.LITTLE_ENDIAN);
        header = (char) buf.get() + "";
        header += (char) buf.get() + "";
        if (header.equals("MD")) {
            // Read first page (& data-block) to get time, temp,
            // measurement frequency, and start of epoch values
            try {
                LocalDateTime blockTime = cwaHeaderLoggingStartTime(buf);
                setSessionStart(blockTime);
                System.out.println("Device was programmed with delayed start time");
                System.out.println("Session start: " + sessionStart);
            } catch (Exception e) {
            }
        } else if (header.equals("AX")) {
            // read each individual page block, and process epochs...
            try {
                // read block header items
                long blockTimestamp = getUnsignedInt(buf, 14);
                int light = getUnsignedShort(buf, 18);
                double temperature = (getUnsignedShort(buf, 20) * 150.0 - 20500) / 1000;
                short rateCode = (short) (buf.get(24) & 0xff);
                short numAxesBPS = (short) (buf.get(25) & 0xff);
                int sampleCount = getUnsignedShort(buf, 28);
                short timestampOffset = 0;
                double sampleFreq = 0;
                int fractional = 0; // 1/65536th of a second fractions

                // check not very old file as pos 26=freq rather than
                // timestamp offset
                if (rateCode != 0) {
                    timestampOffset = buf.getShort(26); // timestamp
                                                        // offset ok
                    // if fractional offset, then timestamp offset was
                    // artificially
                    // modified for backwards-compatibility ...
                    // therefore undo this...
                    int oldDeviceId = getUnsignedShort(buf, 4);
                    if ((oldDeviceId & 0x8000) != 0) {
                        sampleFreq = 3200.0 / (1 << (15 - (rateCode & 15)));
                        if (USE_PRECISE_TIME) {
                            // Need to undo backwards-compatible shim:
                            // Take into account how many whole samples
                            // the fractional part of timestamp
                            // accounts for:
                            // relativeOffset = fifoLength -
                            // (short)(((unsigned long)timeFractional *
                            // AccelFrequency()) >> 16);
                            // nearest whole sample
                            // whole-sec | /fifo-pos@time
                            // | |/
                            // [0][1][2][3][4][5][6][7][8][9]
                            // use 15-bits as 16-bit fractional time
                            fractional = ((oldDeviceId & 0x7fff) << 1);
                            // frequency is truncated to int in firmware
                            timestampOffset += ((fractional * (int) sampleFreq) >> 16);
                        }
                    }
                } else {
                    sampleFreq = buf.getShort(26);
                    // very old format, where pos26 = freq
                }

                // calculate num bytes per sample...
                byte bytesPerSample = 4;
                int NUM_AXES_PER_SAMPLE = 3;
                if ((numAxesBPS & 0x0f) == 2) {
                    bytesPerSample = 6; // 3*16-bit
                } else if ((numAxesBPS & 0x0f) == 0) {
                    bytesPerSample = 4; // 3*10-bit + 2
                }

                // Limit values
                int maxSamples = 480 / bytesPerSample; //80 or 120 samples/block
                if (sampleCount > maxSamples) {
                    sampleCount = maxSamples;
                }
                if (sampleFreq <= 0) {
                    sampleFreq = 1;
                }

                // determine time for indexed sample within block
                LocalDateTime blockTime = getCwaTimestamp((int) blockTimestamp,
                                                    fractional);

                // if sessionStart not set yet, this is the first block
                if (sessionStart == null) {
                    setSessionStart(blockTime);
                    System.out.println("Session start: " + sessionStart);
                }

                // first & last sample. Actually, last = first sample in next block
                LocalDateTime firstSampleTime, lastSampleTime;
                // if no interval between times (or interval too large)
                long spanToSample = 0;
                if (lastBlockTime[0] != null) {
                    spanToSample = Duration.between(lastBlockTime[0], blockTime).toNanos();
                }
                if (!USE_PRECISE_TIME ||
                        lastBlockTime[0] == null ||
                        timestampOffset <= lastBlockTimeIndex[0] ||
                        spanToSample <= 0 ||
                        spanToSample > 1000000000.0 * 2 * maxSamples / sampleFreq
                    ) {
                    float offsetStart = (float) -timestampOffset / (float) sampleFreq;
                    firstSampleTime = blockTime.plusNanos(secs2Nanos(offsetStart));
                    lastSampleTime = firstSampleTime.plusNanos(secs2Nanos(sampleCount / sampleFreq));
                } else {
                    double gap = (double) spanToSample / (-lastBlockTimeIndex[0] + timestampOffset);
                    firstSampleTime = lastBlockTime[0].plusNanos((long) (-lastBlockTimeIndex[0] * gap));
                    lastSampleTime = lastBlockTime[0]
                            .plusNanos((long) ((-lastBlockTimeIndex[0] + sampleCount) * gap));
                }

                // Last block time
                lastBlockTime[0] = blockTime;
                // Advance last block time index for next block
                lastBlockTimeIndex[0] = timestampOffset - sampleCount;
                // Overall span of block
                long spanNanos = Duration.between(firstSampleTime, lastSampleTime).toNanos();

                // raw reading values
                long t = 0;  // Unix time in millis
                long value = 0; // x/y/z vals
                short xRaw = 0;
                short yRaw = 0;
                short zRaw = 0;
                double x = 0.0;
                double y = 0.0;
                double z = 0.0;

                // loop through each line in data block and check if it is last
                // in epoch, then write epoch summary to file.
                // An epoch will have a start+end time, and fixed duration
                for (int i = 0; i < sampleCount; i++) {
                    if (USE_PRECISE_TIME) {
                        // Calculate each sample's time, not successively adding
                        // so that we don't accumulate any errors
                        blockTime = firstSampleTime.plusNanos((long) (i * (double) spanNanos / sampleCount));
                    } else if (i == 0) {
                            blockTime = firstSampleTime; // emulate original behaviour
                    } else {
                            blockTime = blockTime.plusNanos(secs2Nanos(1.0 / sampleFreq));
                    }

                    if (bytesPerSample == 4) {
                        try {
                            value = getUnsignedInt(buf, 30 + 4 * i);
                        } catch (Exception excep) {
                            errCounter[0] += 1;
                            System.err.println("xyz reading err: " + excep.toString());
                            break; // rest of block/page may be corrupted
                        }
                        // Sign-extend 10-bit values, adjust for exponents
                        xRaw = (short) ((short) (0xffffffc0 & (value << 6)) >> (6 - ((value >> 30) & 0x03)));
                        yRaw = (short) ((short) (0xffffffc0 & (value >> 4)) >> (6 - ((value >> 30) & 0x03)));
                        zRaw = (short) ((short) (0xffffffc0 & (value >> 14)) >> (6 - ((value >> 30) & 0x03)));
                    } else if (bytesPerSample == 6) {
                        try {
                            xRaw = buf.getShort(30 + 2 * NUM_AXES_PER_SAMPLE * i + 0);
                            yRaw = buf.getShort(30 + 2 * NUM_AXES_PER_SAMPLE * i + 2);
                            zRaw = buf.getShort(30 + 2 * NUM_AXES_PER_SAMPLE * i + 4);
                        } catch (Exception excep) {
                            errCounter[0] += 1;
                            System.err.println("xyz read err: " + excep.toString());
                            break; // rest of block/page may be corrupted
                        }
                    } else {
                        xRaw = 0;
                        yRaw = 0;
                        zRaw = 0;
                    }
                    x = xRaw / 256.0;
                    y = yRaw / 256.0;
                    z = zRaw / 256.0;
                    t = zonedWithDSTCorrection(blockTime).toInstant().toEpochMilli();

                    epochWriter.newValues(t, x, y, z, temperature, errCounter);

                }
            } catch (Exception excep) {
                excep.printStackTrace(System.err);
                System.err.println(
                    "block err @ " + zonedWithDSTCorrection(lastBlockTime[0]).toString() + ": " + excep.toString()
                );
            }
        }
    }


    // Parse HEX values, CWA format is described at:
    // https://github.com/digitalinteraction/openmovement/blob/master/Downloads/AX3/AX3-CWA-Format.txt
    private static LocalDateTime getCwaTimestamp(int cwaTimestamp, int fractional) {
        LocalDateTime tStamp;
        int year = (int) ((cwaTimestamp >> 26) & 0x3f) + 2000;
        int month = (int) ((cwaTimestamp >> 22) & 0x0f);
        int day = (int) ((cwaTimestamp >> 17) & 0x1f);
        int hours = (int) ((cwaTimestamp >> 12) & 0x1f);
        int mins = (int) ((cwaTimestamp >> 6) & 0x3f);
        int secs = (int) ((cwaTimestamp) & 0x3f);
        tStamp = LocalDateTime.of(year, month, day, hours, mins, secs);
        // add 1/65536th fractions of a second
        tStamp = tStamp.plusNanos(secs2Nanos(fractional / 65536.0));
        tStamp = tStamp.plusMinutes(timeShift);
        return tStamp;
    }


    private static LocalDateTime cwaHeaderLoggingStartTime(ByteBuffer buf) {
        long delayedLoggingStartTime = getUnsignedInt(buf, 13);
        return getCwaTimestamp((int) delayedLoggingStartTime, 0);
    }

}
