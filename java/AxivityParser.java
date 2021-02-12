import java.io.FileInputStream;
import java.nio.ByteOrder;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.zone.ZoneRules;
import java.util.concurrent.TimeUnit;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;


public class AxivityParser {

    private static final boolean USE_PRECISE_TIME = true;
    private static final int BUFSIZE = 512;

    // Specification of items to be written
    private static final Map<String, String> ITEM_NAMES_AND_TYPES;
    static{
        Map<String, String> itemNamesAndTypes = new LinkedHashMap<String, String>();
        itemNamesAndTypes.put("time", "Long");
        itemNamesAndTypes.put("x", "Double");
        itemNamesAndTypes.put("y", "Double");
        itemNamesAndTypes.put("z", "Double");
        itemNamesAndTypes.put("T", "Double");
        // itemNamesAndTypes.put("lux", "Integer");
        ITEM_NAMES_AND_TYPES = Collections.unmodifiableMap(itemNamesAndTypes);
    }


    private String accFile;
    private String outFile;
    private String timeZone;
    private int timeShift;

    private ZonedDateTime sessionStart = null;
    private boolean isSessionStartDST;
    private ZoneId zoneId;
    private ZoneRules rules;

    private int errCounter = 0;
    private LocalDateTime lastBlockTime = null;
    private int lastBlockTimeIndex = 0;

    private NpyWriter writer;


    public AxivityParser(String accFile, String outFile, String timeZone, int timeShift) {
        this.accFile = accFile;
        this.outFile = outFile;
        this.timeZone = timeZone;
        this.timeShift = timeShift;

        setZoneIdAndRules();
    }


    public int parse() {

        try(FileInputStream accStream = new FileInputStream(accFile);
            FileChannel accChannel = accStream.getChannel();) {

            setupWriter();

            // now read every page in CWA file
            ByteBuffer buf = ByteBuffer.allocate(BUFSIZE);
            while (accChannel.read(buf) != -1) {
                parseCwaBuffer(buf);
                buf.clear();
            }

            return errCounter;

        } catch (Exception e) {
            e.printStackTrace(System.err);
            System.err.println("Error reading/writing file " + accFile + ": " + e.toString());
            return -1;

        } finally {
            closeWriter();
        }

    }


    public static int parse(String accFile, String outFile, String timeZone, int timeShift) {

        return new AxivityParser(accFile, outFile, timeZone, timeShift).parse();

    }


    /**
     * CWA format is described at:
     * https://github.com/digitalinteraction/openmovement/blob/master/Downloads/AX3/AX3-CWA-Format.txt
    **/
    private void parseCwaBuffer(ByteBuffer buf) {
        buf.flip();
        buf.order(ByteOrder.LITTLE_ENDIAN);
        String header = (char) buf.get() + "";
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
                // e.printStackTrace();
            }

        } else if (header.equals("AX")) {
            // read each individual page block, and process epochs...
            try {
                // read block header items
                long blockTimestamp = getUnsignedInt(buf, 14);
                // int light = getUnsignedShort(buf, 18);
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
                LocalDateTime blockTime = getCwaTimestamp((int) blockTimestamp, fractional);

                // if sessionStart not set yet, this is the first block
                if (sessionStart == null) {
                    setSessionStart(blockTime);
                    System.out.println("Session start: " + sessionStart);
                }

                // first & last sample. Actually, last = first sample in next block
                LocalDateTime firstSampleTime, lastSampleTime;
                // if no interval between times (or interval too large)
                long spanToSample = 0;
                if (lastBlockTime != null) {
                    spanToSample = Duration.between(lastBlockTime, blockTime).toNanos();
                }
                if (!USE_PRECISE_TIME ||
                        lastBlockTime == null ||
                        timestampOffset <= lastBlockTimeIndex ||
                        spanToSample <= 0 ||
                        spanToSample > 1000000000.0 * 2 * maxSamples / sampleFreq
                    ) {
                    float offsetStart = (float) -timestampOffset / (float) sampleFreq;
                    firstSampleTime = blockTime.plusNanos(secs2Nanos(offsetStart));
                    lastSampleTime = firstSampleTime.plusNanos(secs2Nanos(sampleCount / sampleFreq));
                } else {
                    double gap = (double) spanToSample / (-lastBlockTimeIndex + timestampOffset);
                    firstSampleTime = lastBlockTime.plusNanos((long) (-lastBlockTimeIndex * gap));
                    lastSampleTime = lastBlockTime
                            .plusNanos((long) ((-lastBlockTimeIndex + sampleCount) * gap));
                }

                // Last block time
                lastBlockTime = blockTime;
                // Advance last block time index for next block
                lastBlockTimeIndex = timestampOffset - sampleCount;
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
                            errCounter += 1;
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
                            errCounter += 1;
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

                    try {
                        writer.write(toItems(t, x, y, z, temperature));
                    } catch (Exception e) {
                        System.err.println("Line write error: " + e.toString());
                        System.exit(-1);
                    }

                }

            } catch (Exception e) {
                e.printStackTrace(System.err);
                System.err.println(
                    "Block err @ " + zonedWithDSTCorrection(lastBlockTime).toString() + ": " + e.toString()
                );
            }
        }
    }


    // CWA format is described at:
    // https://github.com/digitalinteraction/openmovement/blob/master/Downloads/AX3/AX3-CWA-Format.txt
    private LocalDateTime getCwaTimestamp(int cwaTimestamp, int fractional) {
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


    private LocalDateTime cwaHeaderLoggingStartTime(ByteBuffer buf) {
        long delayedLoggingStartTime = getUnsignedInt(buf, 13);
        return getCwaTimestamp((int) delayedLoggingStartTime, 0);
    }


    private ZonedDateTime zonedWithDSTCorrection(LocalDateTime ldt) {
        ZonedDateTime zdt = ldt.atZone(zoneId);
        boolean isDST = rules.isDaylightSavings(zdt.toInstant());
        if (isDST != isSessionStartDST) {  // DST crossover happened
            zdt = isDST ? ldt.plusHours(1).atZone(zoneId) : zdt.plusHours(-1);
        }
        return zdt;
    }


    private void setZoneIdAndRules() {
        zoneId = ZoneId.of(timeZone);
        rules = zoneId.getRules();
    }


    private void setSessionStart(LocalDateTime ldt) {
        sessionStart = ldt.atZone(zoneId);
        isSessionStartDST = rules.isDaylightSavings(sessionStart.toInstant());
    }


    public String getTimeZone() {
        return timeZone;
    }


    public String getOutFile() {
        return outFile;
    }


    private static long secs2Nanos(double num) {
        return (long) (TimeUnit.SECONDS.toNanos(1) * num);
    }


    // http://stackoverflow.com/questions/9883472/is-it-possiable-to-have-an-unsigned-bytebuffer-in-java
    private static long getUnsignedInt(ByteBuffer bb, int position) {
        return ((long) bb.getInt(position) & 0xffffffffL);
    }


    // http://stackoverflow.com/questions/9883472/is-it-possiable-to-have-an-unsigned-bytebuffer-in-java
    private static int getUnsignedShort(ByteBuffer bb, int position) {
        return (bb.getShort(position) & 0xffff);
    }


    private void setupWriter() {
        writer = new NpyWriter(outFile, ITEM_NAMES_AND_TYPES);
    }


    private void closeWriter() {
        try {
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private static Map<String, Object> toItems(long t, double x, double y, double z, double temperature) {
        Map<String, Object> items = new HashMap<String, Object>();
        items.put("time", t);
        items.put("x", x);
        items.put("y", y);
        items.put("z", z);
        items.put("T", temperature);
        // items.put("lux", light);
        return items;
    }


}
