import java.io.BufferedWriter;
import java.io.IOException;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.time.zone.ZoneRulesProvider;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

public class EpochWriter {

	private static final DecimalFormatSymbols decimalFormatSymbol = new DecimalFormatSymbols(Locale.ENGLISH);
	private static DecimalFormat DF6 = new DecimalFormat("0.000000", decimalFormatSymbol);
	private static DecimalFormat DF3 = new DecimalFormat("0.000", decimalFormatSymbol);
	private static DecimalFormat DF2 = new DecimalFormat("0.00", decimalFormatSymbol);
	private final long UNUSED_DATE = -1;

	// Storage variables setup:
	// store x/y/z values to pass into epochWriter
	private List<Long> timeVals = new ArrayList<Long>(); // offset into the epoch in milliseconds
	private List<Double> xVals = new ArrayList<Double>();
	private List<Double> yVals = new ArrayList<Double>();
	private List<Double> zVals = new ArrayList<Double>();
	private List<Double> temperatureVals = new ArrayList<Double>();
	private final int minSamplesForEpoch = 10;

	private long prevTimeVal = -1;
	private double[] prevXYZT = { -1, -1, -1, -1 }; // x/y/z/temp at prevTimeVal
	private boolean edgeInterpolation = true;

	// parameters to be initialised
	private long epochStartTime = UNUSED_DATE; // start point of current epoch (in milliseconds from 1970 epoch)
	private int epochPeriod; // duration of epoch (seconds)
    private DateTimeFormatter timeFormat;
    private String timeZone;
	private boolean getStationaryBouts;
	private double stationaryStd;
	private double[] xyzIntercept;
	private double[] xyzSlope;
	private double[] xyzSlopeT;
	private int intendedSampleRate;
    private String resampleMethod;
	private int range;
	private Filter filter;
	private long startTime; // milliseconds since epoch
	private long endTime;
	private boolean getFeatures;

	// file read/write objects
	private BufferedWriter epochFileWriter;
	private BufferedWriter rawWriter; // raw and fft are null if not used
    private NpyWriter npyWriter;

    private ZoneId zoneId;


	public EpochWriter(
		      BufferedWriter epochFileWriter,
		      BufferedWriter rawWriter,
		      NpyWriter npyWriter,
              DateTimeFormatter timeFormat,
              String timeZone,
		      int epochPeriod,
		      int intendedSampleRate,
              String resampleMethod,
		      int range,
		      double[] xyzIntercept,
		      double[] xyzSlope,
		      double[] xyzSlopeT,
		      Boolean getStationaryBouts,
		      double stationaryStd,
		      Filter filter,
		      long startTime,
		      long endTime,
		      boolean getFeatures)
    {
		this.epochFileWriter = epochFileWriter;
		this.rawWriter = rawWriter;
		this.npyWriter = npyWriter;
		this.timeFormat = timeFormat;
		this.epochPeriod = epochPeriod;
		this.intendedSampleRate = intendedSampleRate;
        this.resampleMethod = resampleMethod;
		this.range = range;
		this.xyzIntercept = xyzIntercept;
		this.xyzSlope = xyzSlope;
		this.xyzSlopeT = xyzSlopeT;
		this.getStationaryBouts = getStationaryBouts;
		this.stationaryStd = stationaryStd;
		this.filter = filter;
		this.startTime = startTime;
		this.endTime = endTime;
		this.getFeatures = getFeatures;

        this.zoneId = ZoneId.of(timeZone);

		// NaN's and infinity normally display as non-ASCII characters
		decimalFormatSymbol.setNaN("NaN");
		decimalFormatSymbol.setInfinity("inf");
		DF6.setDecimalFormatSymbols(decimalFormatSymbol);
		DF3.setDecimalFormatSymbols(decimalFormatSymbol);
		DF2.setDecimalFormatSymbols(decimalFormatSymbol);

		DF6.setRoundingMode(RoundingMode.CEILING);
		DF3.setRoundingMode(RoundingMode.HALF_UP); // To match mHealth Gt3x implementation
        DF2.setRoundingMode(RoundingMode.CEILING);

		String epochHeader = "time";
		epochHeader += "," + AccStats.getStatsHeader(getFeatures);
    	epochHeader += ",temp,samples";
		epochHeader += ",dataErrors,clipsBeforeCalibr,clipsAfterCalibr,rawSamples";

		writeLine(epochFileWriter, epochHeader);

		if (rawWriter!=null)
			writeLine(rawWriter, "time,x,y,z");

	}


	// Method which accepts raw values and writes them to an epoch (when enough values collected)
	// Returns true to continue processing, or false if endTime has been reached
	public boolean newValues(
			long time, // Unix time (milliseconds)
			double x,
			double y,
			double z,
			double temperature,
			int[] errCounter) {

		if (startTime!=UNUSED_DATE && time<startTime) {
			return true;
		}

		if (epochStartTime==UNUSED_DATE) { // if first good value start new epoch
			if (startTime==UNUSED_DATE) {
				epochStartTime = time;
			} else {
				// if -startTime option is set, ensure that the first epoch would start at that time
				epochStartTime = startTime;
				int numSkipped = 0;
				while(epochStartTime + epochPeriod * 1000 < time) {
					epochStartTime += epochPeriod * 1000;
					numSkipped++;
				}
				System.out.println("first epochtime set to startTime" +
				 	(numSkipped>0 ? " + " + numSkipped + " epochs" : "") +
				 	" at " + millisToZonedDateTime(epochStartTime));
			}
		}

		if (time<prevTimeVal && prevTimeVal != UNUSED_DATE) {
			errCounter[0] += 1;
			return true;
		}
		// check for large discontinuities in time intervals
		if (time-prevTimeVal >= epochPeriod * 2 * 1000 && prevTimeVal != UNUSED_DATE) {
			System.err.println(
                "Interrupt of length: " + (time-prevTimeVal)/1000.0 + "s, at epoch "
                + millisToZonedDateTime(epochStartTime) + " \n from: "
                + millisToZonedDateTime(prevTimeVal) + "\n to  : "
                + millisToZonedDateTime(time)
            );
			// log that an error occurred, and write epoch with previous values
			errCounter[0] += 1;
			if (timeVals.size()>minSamplesForEpoch) {
				writeEpochSummary(millisToZonedDateTime(epochStartTime), timeVals,
				// writeEpochSummary(millisToInstant(epochStartTime), timeVals,
					xVals, yVals, zVals, temperatureVals, errCounter);
			} else {
				System.err.println("not enough samples for an epoch.. discarding " +
					timeVals.size()+" samples");
				timeVals.clear();
				xVals.clear();
				yVals.clear();
				zVals.clear();
				temperatureVals.clear();
				errCounter[0] = 0;
			}
			// epoch times must be at regular (epochPeriod) intervals, so move forward
			while (epochStartTime<time-epochPeriod*1000) {
				epochStartTime += epochPeriod * 1000;
			}

        }

		// check to see if we have collected enough values to form an epoch
		if (time-epochStartTime >= epochPeriod * 1000 && xVals.size() > minSamplesForEpoch) {
			if (edgeInterpolation) {
				// this code adds the last sample of the next epoch so we can
				//correctly interpolate to the edges
				timeVals.add(time - epochStartTime);
				xVals.add(x);
				yVals.add(y);
				zVals.add(z);
				temperatureVals.add(temperature);
			}
			writeEpochSummary(millisToZonedDateTime(epochStartTime), timeVals,
			// writeEpochSummary(millisToInstant(epochStartTime), timeVals,
				xVals, yVals, zVals, temperatureVals, errCounter);

			epochStartTime = epochStartTime + epochPeriod * 1000;

			if (edgeInterpolation) {
				// this code adds the first sample of the previous epoch so we
				//can correctly interpolate to the edges
				timeVals.add(prevTimeVal - epochStartTime);
				xVals.add(prevXYZT[0]);
				yVals.add(prevXYZT[1]);
				zVals.add(prevXYZT[2]);
				temperatureVals.add(prevXYZT[3]);
			}
		}
		if (endTime!=UNUSED_DATE && time>endTime) {
			System.out.println("reached endTime at sample:" +
                millisToZonedDateTime(time));
			try {
				if (epochFileWriter!=null) epochFileWriter.close();
				if (rawWriter!=null) rawWriter.close();
				if (npyWriter!=null) npyWriter.close();
			} catch (Exception ex) {
				System.err.println("error closing output files");
			}
			System.exit(0); // end processing
		}
		// store axes + vector magnitude vals for every reading
		timeVals.add(time - epochStartTime);
		xVals.add(x);
		yVals.add(y);
		zVals.add(z);
		temperatureVals.add(temperature);
		prevTimeVal = time;
		prevXYZT = new double[]{x, y, z, temperature};
		return true;
	}


	/**
	 * Method used by all different file-types, to write a single line to the epochWriter.
	 * The method:
	 *  -resamples all data (x/y/z/time/temperatureVals) to the intendedSampleRate
	 *  -uses the calibration parameters and temperature to adjust the x/y/z values
	 *  -increments the errCounter (array length 1) for 'stuck values'
	 *  -writes the raw resampled data to the global rawWriter (unless null)
	 *  [the above does not apply if getSanDiegoFeatures is enabled]
	 */
	private void writeEpochSummary(
			ZonedDateTime epochStartTime,
			// Instant epochStartTime,
			List<Long> timeVals /* milliseconds since start of epochStartTime */,
			List<Double> xVals,
			List<Double> yVals,
			List<Double> zVals,
			List<Double> temperatureVals,
			int[] errCounter) {

		int[] clipsCounter = new int[] { 0, 0 }; // before, after (calibration)
		double x;
		double y;
		double z;
        double temp;
		for (int i = 0; i < xVals.size(); i++) {
			Boolean isClipped = false;
			x = xVals.get(i);
			y = yVals.get(i);
			z = zVals.get(i);
			temp = temperatureVals.get(i);
																// temp
			// check if any pre-calibration clipping present
			//use >= range as it's clipped here
			if (Math.abs(x) >= range || Math.abs(y) >= range || Math.abs(z) >= range) {
				clipsCounter[0] += 1;
				isClipped = true;
			}

			// update values to software calibrated values
			x = xyzIntercept[0] + x * xyzSlope[0] + temp * xyzSlopeT[0];
			y = xyzIntercept[1] + y * xyzSlope[1] + temp * xyzSlopeT[1];
			z = xyzIntercept[2] + z * xyzSlope[2] + temp * xyzSlopeT[2];

			// check if any new post-calibration clipping has happened
			// find crossing of range threshold so use > rather than >=
			if (Math.abs(x) > range || Math.abs(y) > range || Math.abs(z) > range) {
				if (!isClipped) {
					clipsCounter[1] += 1;
				}
				// drag post calibration clipped values back to range limit
				if (x < -range || (isClipped && x < 0)) {
					x = -range;
				} else if (x > range || (isClipped && x > 0)) {
					x = range;
				}
				if (y < -range || (isClipped && y < 0)) {
					y = -range;
				} else if (y > range || (isClipped && y > 0)) {
					y = range;
				}
				if (z < -range || (isClipped && z < 0)) {
					z = -range;
				} else if (z > range || (isClipped && z > 0)) {
					z = range;
				}
			}

			xVals.set(i, x);
			yVals.set(i, y);
			zVals.set(i, z);
		}

		// resample values to epochSec * (intended) sampleRate
		long[] timeResampled = new long[epochPeriod * (int) intendedSampleRate];
		double[] xResampled = new double[timeResampled.length];
		double[] yResampled = new double[timeResampled.length];
		double[] zResampled = new double[timeResampled.length];
		for (int i = 0; i < timeResampled.length; i++) {
			timeResampled[i] = Math.round((epochPeriod * 1000d * i) / timeResampled.length);
		}
        if (resampleMethod.equalsIgnoreCase("linear")) {
            Resample.interpLinear(timeVals, xVals, yVals, zVals,
                timeResampled, xResampled, yResampled, zResampled);
        } else if (resampleMethod.equalsIgnoreCase("nearest")) {
            Resample.interpNearest(timeVals, xVals, yVals, zVals,
                timeResampled, xResampled, yResampled, zResampled);
        } else {
			System.err.println("Unknown resample method: " + resampleMethod);
			System.exit(-1);
        }

		//write out raw values ...
		if (rawWriter != null) {
			for (int i = 0; i < xResampled.length; i++) {
				writeLine(
                    rawWriter,
                    timeFormat.format(epochStartTime.plus(timeResampled[i], ChronoUnit.MILLIS))
                    + "," + DF3.format(xResampled[i])
                    + "," + DF3.format(yResampled[i])
                    + "," + DF3.format(zResampled[i]));
			}
        }
		if (npyWriter!=null) {
			for (int i = 0; i < xResampled.length; i++) {
                // Note: For .npy format, we store time in Unix nanoseconds
                long time = toNanos(epochStartTime.plus(timeResampled[i], ChronoUnit.MILLIS));
                writeNpyLine(npyWriter, time, xResampled[i], yResampled[i], zResampled[i]);
            }

		}

		// extract necessary features for this epoch
		double[] stats = AccStats.getAccStats(xResampled, yResampled, zResampled, filter, getFeatures, intendedSampleRate);

		// check if the values have likely been stuck during this epoch
		errCounter[0] += AccStats.countStuckVals(xResampled, yResampled, zResampled);

		// write summary values to file
        String epochSummary = timeFormat.format(epochStartTime);
		for(int i=0; i<stats.length; i++){
			epochSummary += "," + DF6.format(stats[i]);
		}

		// write housekeeping stats
		epochSummary += "," + DF2.format(AccStats.mean(temperatureVals));
		epochSummary += "," + xResampled.length + "," + errCounter[0];
		epochSummary += "," + clipsCounter[0] + "," + clipsCounter[1];
		epochSummary += "," + timeVals.size();

		//write line to file...
		double xStd = stats[8]; //needed to identify stationary episodes
		double yStd = stats[9]; //if running first step of calibration process
		double zStd = stats[10];
		if (!getStationaryBouts || (xStd < stationaryStd && yStd < stationaryStd && zStd < stationaryStd)) {
			writeLine(epochFileWriter, epochSummary);
		}

		timeVals.clear();
		xVals.clear();
		yVals.clear();
		zVals.clear();
		temperatureVals.clear();
		errCounter[0] = 0;
    }


	private static void writeLine(BufferedWriter fileWriter, String line) {
        try {
            fileWriter.write(line + "\n");
        } catch (Exception excep) {
            System.err.println("line write error: " + excep.toString());
        }
    }


    private static void writeNpyLine(NpyWriter npyWriter, long time, double x, double y, double z){
        if (Double.isNaN(x) && Double.isNaN(y) && Double.isNaN(z)) {
            System.err.println("NaN at "+time+","+x+","+y+","+z);
        }

        try {
            npyWriter.writeData(time, (float) x, (float) y, (float) z);
        } catch (Exception excep) {
            System.err.println("line write error: " + excep.toString());
        }
    }


	public void closeWriters(){
		try{
			epochFileWriter.close();
			if (rawWriter != null) rawWriter.close();
			if (npyWriter != null) npyWriter.close();
		} catch (IOException excep) {
			excep.printStackTrace(System.err);
			System.err.println("error closing file writer: " + excep.toString());
			System.exit(-2);
		}
	}


	private static LocalDateTime millisToTimestamp(double d) {
        return LocalDateTime.ofInstant(new Date((long) d).toInstant(), ZoneOffset.UTC);
    }


    private static double timestampToMillis(LocalDateTime ldt) {
        ZonedDateTime zdt = ldt.atZone(ZoneId.of("UTC"));
        long millis = zdt.toInstant().toEpochMilli();
        return millis;
    }


    private static Instant millisToInstant(long t) {
        return Instant.ofEpochMilli(t);
    }


    private ZonedDateTime millisToZonedDateTime(long t) {
        return millisToInstant(t).atZone(this.zoneId);
    }


    private static long toNanos(Instant ins) {
        return (long) TimeUnit.SECONDS.toNanos(ins.getEpochSecond()) + ins.getNano();
    }


    private static long toNanos(ZonedDateTime zdt) {
        return toNanos(zdt.toInstant());
    }


}
