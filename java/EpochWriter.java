import java.io.BufferedWriter;
import java.io.IOException;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Locale;

import org.jtransforms.fft.DoubleFFT_1D;

public class EpochWriter {

	private static final DecimalFormatSymbols decimalFormatSymbol = new DecimalFormatSymbols(Locale.ENGLISH);
	private static DecimalFormat DF8 = new DecimalFormat("0.00000000", decimalFormatSymbol);
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
	private double[] prevXYZT = { -1, -1, -1, -1 }; // x, y, z, and temp at prevTimeVal
	private boolean edgeInterpolation = true;

	// parameters to be initialised
	private long epochStartTime = UNUSED_DATE; // start point of current epoch (in milliseconds from 1970 epoch)
	private int epochPeriod; // duration of epoch (seconds)
	private DateTimeFormatter timeFormat;
	private boolean getEpochCovariance;
	private boolean getAxisMeans;
	private boolean getStationaryBouts;
	private boolean useAbs; // use abs(VM) instead of trunc(VM)
	private double stationaryStd;
	private double[] swIntercept;
	private double[] swSlope;
	private double[] tempCoef;
	private double meanTemp;
	private int intendedSampleRate;
	private int range;
	private LowpassFilter filter;
	private long startTime; // milliseconds since epoch
	private long endTime;

	private boolean getSanDiegoFeatures;
	private boolean getMADFeatures;
	private boolean getUnileverFeatures;
	private boolean get3DFourier;
	private boolean getEachAxis;
	private int numEachAxis;


	// file read/write objects
	private BufferedWriter epochFileWriter;
	private BufferedWriter rawWriter; // raw and fft are null if not used
	private BufferedWriter fftWriter; // otherwise will be created (using epochFile
	private NpyWriter npyWriter;



	// have we added the header to the FFT file? (so can generate one based on number of samples)
	private Boolean fftWriterHasHeader = false;


	public EpochWriter(
		      BufferedWriter epochFileWriter,
		      BufferedWriter rawWriter,
		      BufferedWriter fftWriter,
		      NpyWriter npyWriter,
		      DateTimeFormatter timeFormat,
		      int epochPeriod,
		      int intendedSampleRate,
		      int range,
		      double[] swIntercept,
		      double[] swSlope,
		      double[] tempCoef,
		      double meanTemp,
		      Boolean getStationaryBouts,
		      double stationaryStd,
		      LowpassFilter filter,
		      boolean getEpochCovariance,
		      boolean getAxisMeans,
		      long startTime,
		      long endTime,
		      boolean getSanDiegoFeatures,
		      boolean getMADFeatures,
		      boolean getUnileverFeatures,
		      boolean get3DFourier,
		      boolean getEachAxis,
		      int numEachAxis,
		      boolean useAbs) {
		this.epochFileWriter = epochFileWriter;
		this.rawWriter = rawWriter;
		this.fftWriter = fftWriter;
		this.npyWriter = npyWriter;
		this.timeFormat = timeFormat;
		this.epochPeriod = epochPeriod;
		this.intendedSampleRate = intendedSampleRate;
		this.range = range;
		this.swIntercept = swIntercept;
		this.swSlope = swSlope;
		this.tempCoef = tempCoef;
		this.meanTemp = meanTemp;
		this.getStationaryBouts = getStationaryBouts;
		this.stationaryStd = stationaryStd;
		this.filter = filter;
		this.getEpochCovariance = getEpochCovariance;
		this.getAxisMeans = getAxisMeans;
		this.startTime = startTime;
		this.endTime = endTime;
		this.getSanDiegoFeatures = getSanDiegoFeatures;
		this.getMADFeatures = getMADFeatures;
		this.getUnileverFeatures = getUnileverFeatures;
		this.get3DFourier = get3DFourier;
		this.getEachAxis = getEachAxis;
		this.numEachAxis = numEachAxis;
		this.useAbs = useAbs;

		/* edge interpolation is only useful for resampling. The San Diego features
		   do not use resampling so this results in a different number of samples for
		   the first epoch which will result in slightly different FFT output.
		*/
//		if (this.getSanDiegoFeatures) this.edgeInterpolation = false;

		// NaN's and infinity normally display as non-ASCII characters
		decimalFormatSymbol.setNaN("NaN");
		decimalFormatSymbol.setInfinity("inf");
		DF8.setDecimalFormatSymbols(decimalFormatSymbol);
		DF6.setDecimalFormatSymbols(decimalFormatSymbol);
		DF3.setDecimalFormatSymbols(decimalFormatSymbol);
		DF2.setDecimalFormatSymbols(decimalFormatSymbol);

		DF8.setRoundingMode(RoundingMode.HALF_UP);
		DF6.setRoundingMode(RoundingMode.CEILING);
		DF3.setRoundingMode(RoundingMode.HALF_UP); // To match the mHealth Gt3x implementation
		DF2.setRoundingMode(RoundingMode.CEILING);

//    	System.out.println(DF6.format(Double.NEGATIVE_INFINITY));
//    	System.out.println(DF6.format(Double.NaN));
//    	System.out.println(DF6.format(Double.POSITIVE_INFINITY));

    	/*// testing the new method:
    	double[] inputArray = new double[9];
    	Arrays.setAll(inputArray, p -> (double) p);
    	System.out.println(Arrays.toString(inputArray));
    	double [] percentiles = new double[] {-0.01, 0.0, 0.01,0.011, 0.25, 0.5, 0.75,0.965, 0.99, 1.0, 1.01};

    	System.out.println(Arrays.toString(percentiles(inputArray, percentiles)));
    	// [0.0, 0.0, 1.0, 1.0999999999999996, 25.0, 50.0, 75.0, 96.5, 99.0, 99.0, 99.0]
    	System.out.println(Arrays.toString(percentiles));
		System.exit(0);*/

		String epochHeader = "time,enmoTrunc";
		if (useAbs) {
			epochHeader += ",enmoAbs";
		}
		if (getStationaryBouts || getAxisMeans) {
			epochHeader += ",xMean,yMean,zMean";
		}
		epochHeader += ",xRange,yRange,zRange,xStd,yStd,zStd";
		if (getEpochCovariance) {
			epochHeader += ",xyCov,xzCov,yzCov";
		}
		epochHeader += ",temp,samples";
		epochHeader += ",dataErrors,clipsBeforeCalibr,clipsAfterCalibr,rawSamples";
		if (getSanDiegoFeatures) {
			epochHeader += ",mean,sd,coefvariation,median,min,max,25thp,75thp,autocorr,corrxy,corrxz,corryz,avgroll,avgpitch,avgyaw,sdroll,sdpitch,sdyaw,rollg,pitchg,yawg,fmax,pmax,fmaxband,pmaxband,entropy,fft0,fft1,fft2,fft3,fft4,fft5,fft6,fft7,fft8,fft9,fft10,fft11,fft12,fft13,fft14";
    	}
		if (getMADFeatures) {
			epochHeader += ",MAD,MPD,skew,kurt";
    	}
		if (getUnileverFeatures) {
			epochHeader += ",f1,p1,f2,p2,f625,p625,total";
			int out_n = (int) Math.ceil((epochPeriod * this.intendedSampleRate) /2) + 1; // fft output array size
			out_n = Math.min(out_n, numEachAxis);
			if (getEachAxis && get3DFourier) {
				for (char c: new char[] {'x','y','z','m'}) {
					for (int i=0; i<out_n ; i++) {
						epochHeader += ","+c+"fft"+i;
					}
				}
			} else {
				for (int i=0; i<out_n ; i++) {
					epochHeader += ",ufft"+i;
				}
			}
    	}
		writeLine(epochFileWriter, epochHeader);

		if (rawWriter!=null)
			writeLine(rawWriter, "time,x,y,z");


	}

	// Method which accepts raw values and writes them to an epoch (when enough values collected)
	// Returns true to continue processing, or false if endTime has been reached
	public boolean newValues(
			long time /* milliseconds since start of 1970 epoch */,
			double x, double y, double z, double temperature,
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
				System.out.println("first epochtime set to startTime" + (numSkipped>0 ? " + " + numSkipped + " epochs" : "") + " at " +  millisToTimestamp(epochStartTime).format(timeFormat));
			}
		}

		//System.out.println("reading: " + millisToTimestamp(time).format(timeFormat) + " samples:" + xVals.size());

		if (time<prevTimeVal && prevTimeVal != UNUSED_DATE) {
			//System.err.println("samples not in cronological order: " + time + " is before previous sample: "
			//					+ prevTimeVal + " at epoch" + millisToTimestamp(epochStartTime).format(timeFormat));
			// ignore value
			errCounter[0] += 1;
			return true;
		}
		// check for large discontinuities in time intervals
		if (time-prevTimeVal >= epochPeriod * 2 * 1000 && prevTimeVal != UNUSED_DATE) {
			System.err.println("interrupt of length: " + (time-prevTimeVal)/1000.0 + "s, at epoch "
								+ millisToTimestamp(epochStartTime).format(timeFormat) + " \nfrom:\n" +
								millisToTimestamp(prevTimeVal).format(timeFormat) + "\nto\n" +
								millisToTimestamp(time).format(timeFormat));
			// log that an error occurred, and write epoch with previous values
			errCounter[0] += 1;
			if (timeVals.size()>minSamplesForEpoch) {
				writeEpochSummary(millisToTimestamp(epochStartTime), timeVals, xVals, yVals, zVals, temperatureVals, errCounter);
			} else {
				System.err.println("not enough samples for an epoch.. discarding "+timeVals.size()+" samples");
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
				// this code adds the last sample of the next epoch so we can correctly interpolate to the edges
				timeVals.add(time - epochStartTime);
				xVals.add(x);
				yVals.add(y);
				zVals.add(z);
				temperatureVals.add(temperature);
			}
			writeEpochSummary(millisToTimestamp(epochStartTime), timeVals, xVals, yVals, zVals, temperatureVals, errCounter);

			epochStartTime = epochStartTime + epochPeriod * 1000;

			if (edgeInterpolation) {
				// this code adds the first sample of the previous epoch so we can correctly interpolate to the edges
				timeVals.add(prevTimeVal - epochStartTime);
				xVals.add(prevXYZT[0]);
				yVals.add(prevXYZT[1]);
				zVals.add(prevXYZT[2]);
				temperatureVals.add(prevXYZT[3]);
			}
		}
		if (endTime!=UNUSED_DATE && time>endTime) {
			System.out.println("reached endTime at sample:" + millisToTimestamp(time).format(timeFormat) );
			try {
				if (epochFileWriter!=null) epochFileWriter.close();
				if (rawWriter!=null) rawWriter.close();
				if (fftWriter!=null) fftWriter.close();
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
	 *  -writes the fft output data to the global fftWriter (unless null)
	 *  [the above does not apply if getSanDiegoFeatures is enabled]
	 */
	private void writeEpochSummary(LocalDateTime epochStartTime,
			List<Long> timeVals /* milliseconds since start of epochStartTime */,
			List<Double> xVals, List<Double> yVals, List<Double> zVals, List<Double> temperatureVals,
			int[] errCounter) {

		int[] clipsCounter = new int[] { 0, 0 }; // before, after (calibration)
		double x;
		double y;
		double z;
		for (int c = 0; c < xVals.size(); c++) {
			Boolean isClipped = false;
			x = xVals.get(c);
			y = yVals.get(c);
			z = zVals.get(c);
			double mcTemp = temperatureVals.get(c) - meanTemp; // mean centred
																// temp
			// check if any clipping present, use >= range as it's clipped here
			if (Math.abs(x) >= range || Math.abs(y) >= range || Math.abs(z) >= range) {
				clipsCounter[0] += 1;
				isClipped = true;
			}

			// update values to software calibrated values
			x = swIntercept[0] + x * swSlope[0] + mcTemp * tempCoef[0];
			y = swIntercept[1] + y * swSlope[1] + mcTemp * tempCoef[1];
			z = swIntercept[2] + z * swSlope[2] + mcTemp * tempCoef[2];

			// check if any new clipping has happened
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

			xVals.set(c, x);
			yVals.set(c, y);
			zVals.set(c, z);
		}

		long[] timeResampled = new long[epochPeriod * (int) intendedSampleRate];
		double[] xResampled = new double[timeResampled.length];
		double[] yResampled = new double[timeResampled.length];
		double[] zResampled = new double[timeResampled.length];

		// resample values to epochSec * (intended) sampleRate
		for (int c = 0; c < timeResampled.length; c++) {
			timeResampled[c] = Math.round((epochPeriod * 1000d * c) / timeResampled.length);
		}
		Resample.interpLinear(timeVals, xVals, yVals, zVals, timeResampled, xResampled, yResampled, zResampled);

		// epoch variables
		double accPA = 0;
		// calculate raw x/y/z summary values
		double xMean = mean(xResampled);
		double yMean = mean(yResampled);
		double zMean = mean(zResampled);
		double xRange = range(xResampled);
		double yRange = range(yResampled);
		double zRange = range(zResampled);
		double xStd = std(xResampled, xMean);
		double yStd = std(yResampled, yMean);
		double zStd = std(zResampled, zMean);

		// see if values have been abnormally stuck this epoch
		double stuckVal = 1.5;
		if (xStd == 0 && (xMean < -stuckVal || xMean > stuckVal)) {
			errCounter[0] += 1;
		}
		if (yStd == 0 && (yMean < -stuckVal || yMean > stuckVal)) {
			errCounter[0] += 1;
		}
		if (zStd == 0 && (zMean < -stuckVal || zMean > stuckVal)) {
			errCounter[0] += 1;
		}

		// calculate summary vector magnitude based metrics
		double[] paVals = new double[xResampled.length];
		String MADFeatures = "";
		double accPAAbs = -1;
		if (!getStationaryBouts) {
			for (int c = 0; c < xResampled.length; c++) {
				x = xResampled[c];
				y = yResampled[c];
				z = zResampled[c];

				if (!Double.isNaN(x)) {
					double vm = getVectorMagnitude(x, y, z);
					paVals[c] = vm - 1;
				}
			}
			if (useAbs) {
				double[] paValsAbs = new double[paVals.length];
				for (int c = 0; c < paVals.length; c++) {
					paValsAbs[c] = paVals[c];
				}
				abs(paValsAbs);
				accPAAbs = mean(paValsAbs);
			}
			if (getMADFeatures) {
				// should really be on vm, not vm - 1
				MADFeatures = calculateMADFeatures(paVals);
			}
			// filter AvgVm-1 values
			if (filter != null) {
				filter.filter(paVals);
			}
			// run abs() or trunc() on summary variables after filtering
			trunc(paVals);

			// calculate mean values for each outcome metric
			accPA = mean(paVals);
		}
		if (rawWriter != null) {
			for (int c = 0; c < xResampled.length; c++) {
				writeLine(rawWriter,
						epochStartTime.plusNanos(timeResampled[c] * 1000000).format(timeFormat)
								+ "," + DF3.format(xResampled[c]) + "," + DF3.format(yResampled[c]) + "," + DF3.format(zResampled[c]));
			}
        }
		if (npyWriter!=null) {
			for (int c = 0; c < xResampled.length; c++) {
                // note: For .npy format, we store time in Unix nanoseconds
                long time = (long) (timestampToMillis(epochStartTime.plusNanos(timeResampled[c] * 1000000)) * 1000000);
                writeNpyLine(npyWriter, time, xResampled[c], yResampled[c], zResampled[c]);
            }

		}

		if (fftWriter!=null && !getSanDiegoFeatures) {
			writeFFTEpoch(epochStartTime, paVals);
		}
		// write summary values to file
		String epochSummary = epochStartTime.format(timeFormat);
		epochSummary += "," + DF6.format(accPA);
		if (useAbs) {
			epochSummary += "," + DF6.format(accPAAbs);
		}
		if (getStationaryBouts || getAxisMeans) {
			epochSummary += "," + DF6.format(xMean);
			epochSummary += "," + DF6.format(yMean);
			epochSummary += "," + DF6.format(zMean);
		}
		epochSummary += "," + DF6.format(xRange);
		epochSummary += "," + DF6.format(yRange);
		epochSummary += "," + DF6.format(zRange);
		epochSummary += "," + DF6.format(xStd);
		epochSummary += "," + DF6.format(yStd);
		epochSummary += "," + DF6.format(zStd);
		if (getEpochCovariance) {
			double xyCovariance = covariance(xResampled, yResampled, xMean, yMean, 0);
			double xzCovariance = covariance(xResampled, zResampled, xMean, zMean, 0);
			double yzCovariance = covariance(yResampled, zResampled, yMean, zMean, 0);
			epochSummary += "," + DF6.format(xyCovariance);
			epochSummary += "," + DF6.format(xzCovariance);
			epochSummary += "," + DF6.format(yzCovariance);
		}
		epochSummary += "," + DF2.format(mean(temperatureVals));
		epochSummary += "," + xResampled.length + "," + errCounter[0];
		epochSummary += "," + clipsCounter[0] + "," + clipsCounter[1];
		epochSummary += "," + timeVals.size();

		if (getSanDiegoFeatures) {
			epochSummary += "," + calculateSanDiegoFeatures(timeResampled, xResampled, yResampled, zResampled);
		}

		if (getMADFeatures) {
			epochSummary += "," + MADFeatures;
		}
		if (getUnileverFeatures) {

			epochSummary += "," + unileverFeatures(paVals);
			if (get3DFourier) {
				epochSummary += ","+getFFT3D(xResampled, yResampled, zResampled, paVals);
			}
		}
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

	private String calculateSanDiegoFeatures(
			long[] time /* milliseconds since start of epochStartTime */,
			double[] xResampled, double[] yResampled, double[] zResampled) {


		/*
		 	This function aims to replicate the following R-code:

			  # v = vector magnitude
			  v = sqrt(rowSums(w ^ 2))

			  fMean = mean(v)
			  fStd = sd(v)
			  if (fMean > 0) {
			    fCoefVariation = fStd / fMean
			  } else {
			    fCoefVariation = 0
			  }
			  fMedian = median(v)
			  fMin = min(v)
			  fMax = max(v)
			  f25thP = quantile(v, 0.25)[[1]]
			  f75thP = quantile(v, 0.75)[[1]]

		*/

		int n = xResampled.length;

		// San Diego g values
		// the g matric contains the estimated gravity vector, which is essentially a low pass filter
		double[] gg = sanDiegoGetAvgGravity(xResampled, yResampled, zResampled);
		double gxMean = gg[0];
		double gyMean = gg[1];
		double gzMean = gg[2];


		// subtract column means and get vector magnitude

		double[] v = new double[n]; // vector magnitude
		double[] wx = new double[n]; // gravity adjusted weights
		double[] wy = new double[n];
		double[] wz = new double[n];
		for (int i = 0; i < n; i++) {
			wx[i] = xResampled[i]-gxMean;
			wy[i] = yResampled[i]-gyMean;
			wz[i] = zResampled[i]-gzMean;
			v[i] = getVectorMagnitude( wx[i], wy[i], wz[i]);
//				String str = String.format("[ % 6.5f % 6.5f % 6.5f ] ",wx[i], wy[i], wz[i]);
//				System.out.println(xResampled[i] + ":" + str);
		}

		// Write epoch
		double sdMean = mean(v);
		double sdStd = stdR(v, sdMean);
		double sdCoefVariation = 0.0;
		if (sdMean!=0) sdCoefVariation = sdStd/sdMean;
		double[] paQuartiles = percentiles(v, new double[] {0, 0.25, 0.5, 0.75, 1});

		String sanDiego = "";
		sanDiego += DF8.format(sdMean) + ",";
		sanDiego += DF8.format(sdStd) + ",";
		sanDiego += DF8.format(sdCoefVariation) + ",";
		// median, min, max, 25thp, 75thp
		sanDiego += DF8.format(paQuartiles[2]) + ",";
		sanDiego += DF8.format(paQuartiles[0]) + ",";
		sanDiego += DF8.format(paQuartiles[4]) + ",";
		sanDiego += DF8.format(paQuartiles[1]) + ",";
		sanDiego += DF8.format(paQuartiles[3]) + ",";
		double autoCorrelation = Correlation(v, v, intendedSampleRate);
		sanDiego += DF8.format(autoCorrelation) + ",";
		double xyCorrelation = Correlation(wx, wy);
		double xzCorrelation = Correlation(wx, wz);
		double yzCorrelation = Correlation(wy, wz);
		sanDiego += DF8.format(xyCorrelation) + ",";
		sanDiego += DF8.format(xzCorrelation) + ",";
		sanDiego += DF8.format(yzCorrelation) + ",";

		// Roll, Pitch, Yaw
		double [] angleAvgStdYZ = angleAvgStd(wy, wz);
		double [] angleAvgStdZX = angleAvgStd(wz, wx);
		double [] angleAvgStdYX = angleAvgStd(wy, wx);
		sanDiego += DF8.format(angleAvgStdYZ[0]) + "," +  DF8.format(angleAvgStdZX[0]) + "," +  DF8.format(angleAvgStdYX[0]) + ",";
		sanDiego += DF8.format(angleAvgStdYZ[1]) + "," +  DF8.format(angleAvgStdZX[1]) + "," +  DF8.format(angleAvgStdYX[1]) + ",";

		// gravity component angles
		double gxyAngle = Math.atan2(gyMean,gzMean);
		double gzxAngle = Math.atan2(gzMean,gxMean);
		double gyxAngle = Math.atan2(gyMean,gxMean);
		sanDiego += DF8.format(gxyAngle) + "," +  DF8.format(gzxAngle) + "," +  DF8.format(gyxAngle);

		// FFT
		sanDiego += "," + sanDiegoFFT(v);

		// Finally, write our 'gravity-removed' values to our raw-data file
//		if (rawWriter!=null) {
//			for (int c = 0; c < n; c++) {
//				writeLine(rawWriter, epochStartTime.plusNanos(timeResampled[c] * 1000000).format(timeFormat) + ","+ DF8.format(wx[c]) + "," +DF8.format(wy[c]) + "," + DF8.format(wz[c]) +"," +DF8.format(v[c]));
//			}
//		}
		return sanDiego;
	}

	// returns { x, y, z } averages of gravity
	private double[] sanDiegoGetAvgGravity(double[] xResampled, double[] yResampled, double[] zResampled) {
		// San Diego paper 0.5Hz? low-pass filter approximation
		// this code takes in w and returns gg
		/*    R-code: (Fs is intendedSampleRate, w are x/y/z values)

		  	  g = matrix(0, nrow(w), 3)
			  x = 0.9
			  g[1, ] = (1-x) * w[1, ]
			  for (n in 2:nrow(w)) {
			    g[n, ] = x * g[n-1] + (1-x) * w[n, ]
			  }
			  g = g[Fs:nrow(g), ] # ignore beginning
			  gg = colMeans(g)
			  w = w - gg
		*/
		int n = xResampled.length;
		int gn = n-(intendedSampleRate-1); // number of moving average values to estimate gravity direction with
		int gStartIdx = n - gn; // number of gravity values to discard at beginning

		double[] gx = new double[gn];
		double[] gy = new double[gn];
		double[] gz = new double[gn];

		{
			// calculating moving average of signal
			double x = 0.9;

			double xMovAvg = (1-x)*xResampled[0];
			double yMovAvg = (1-x)*yResampled[0];
			double zMovAvg = (1-x)*zResampled[0];


			for (int c = 1; c < n; c++) {
				if (c < gStartIdx) {
					xMovAvg = xMovAvg * x + (1-x) * xResampled[c];
					yMovAvg = yMovAvg * x + (1-x) * yResampled[c];
					zMovAvg = zMovAvg * x + (1-x) * zResampled[c];
				} else {
					// only store the signal after it has stabilized
					xMovAvg = xMovAvg * x + (1-x) * xResampled[c];
					yMovAvg = yMovAvg * x + (1-x) * yResampled[c];
					zMovAvg = zMovAvg * x + (1-x) * zResampled[c];
					gx[c-gStartIdx] = xMovAvg;
					gy[c-gStartIdx] = yMovAvg;
					gz[c-gStartIdx] = zMovAvg;
				}
			}
		}
		/*for (int c = 0; c < gx.length; c++) {
			System.out.println(c + " - [" + gx[c] + ", " + gy[c]+ ", " + gz[c] + "]");
		}*/

		//System.out.println("size:" + gx.length);

		// column means
		double gxMean = mean(gx);
		double gyMean = mean(gy);
		double gzMean = mean(gz);
		/*System.out.println("xMean = " + gxMean + "");
		System.out.println("yMean = " + gyMean + "");
		System.out.println("zMean = " + gzMean + "");*/

		return new double[] {gxMean, gyMean, gzMean};
	}

	private String sanDiegoFFT(double[] paVals) {

		int n = paVals.length;
		// FFT frequency interval = sample frequency / num samples
		double FFTinterval = intendedSampleRate / (1.0 * n); // (Hz)

		int numBins = 15; // From original implementation
//		System.out.println("n = " + n);
//		System.out.println("Fs =" + intendedSampleRate);

		// set input data array
		double[] vmFFT = new double[n];
        for (int c = 0; c < n; c++) {
        	vmFFT[c] = paVals[c];
        }

        // Hanning window attenuates the signal to zero at it's start and end
        HanningWindow(vmFFT,n);

        DoubleFFT_1D transformer = new DoubleFFT_1D(n);
		transformer.realForward(vmFFT);
		double max = max(vmFFT);


		// find dominant frequency, second dominant frequency, and dominant between .6 - 2.5Hz
		double f1=-1, f33=-1;
		double p1=0, p33=0;

		double totalPLnP = 0.0; // sum of P * ln(P)
		double magDC = vmFFT[0]/max;
		totalPLnP += magDC * Math.log(magDC);
//    	System.out.println(String.format("%d (%.2fHz bin %d) : % .6f + % .6fi = % .6f",0, 0.0, currBin, magDC, 0.0 , magDC));

		for (int i = 1; i < n/2; i++) {
			double freq = FFTinterval * i;
			double Re = vmFFT[i*2];
			double Im = vmFFT[i*2+1];
			double mag = Math.sqrt( Re * Re + Im * Im)/max;

        	totalPLnP += mag * Math.log(mag);

        	if (mag>p1) {
        		f1 = freq;
        		p1 = mag;
        	}
        	if (freq > 0.3 && freq < 3 && mag>p33) {
        		f33 = freq;
        		p33 = mag;
        	}

		}
		// entropy, AKA (Power) Spectral Entropy, measures 'peakyness' of the frequency spectrum
		// This should be higher where there are periodic motions such as walking
		double H = - totalPLnP;// / total - (n/2) * Math.log(total);


		double[] binnedFFT = new double[numBins];

		for (int i = 0; i < numBins; i++) {
			binnedFFT[i] = 0;
		}
		int numWindows = (int) Math.floor(n/intendedSampleRate);
		double[] windowSamples = new double[intendedSampleRate];
        DoubleFFT_1D windowTransformer = new DoubleFFT_1D(intendedSampleRate);
        max = Double.NEGATIVE_INFINITY;
        // do a FFT on each 1 second window (therefore FFT-interval will be one)
        FFTinterval = 1;
		for (int window = 0; window < numWindows; window++ ) {
			for (int i = 0; i < intendedSampleRate; i++) {
				windowSamples[i] = paVals[i+window*(intendedSampleRate/2)];
			}
			HanningWindow(windowSamples, intendedSampleRate);
			for (int i = 0; i < intendedSampleRate; i++) {
				// System.out.println(String.format("%d, %d : % .6f",window, i, windowSamples[i]));
			}
			windowTransformer.realForward(windowSamples);
			for (int i = 0; i < Math.min(intendedSampleRate,10); i++) {
				// System.out.println(String.format("%d, %d : % .6f",window, i, windowSamples[i]));
			}
			for (int i = 0; i < numBins; i++) {
				double mag;
				if (i==0) {
					mag = windowSamples[i];
					// System.out.println(String.format("bin%d: % .6f",i, mag));
				}
				else {
					double Re = windowSamples[i*2];
					double Im = windowSamples[i*2+1];
					mag = Math.sqrt( Re * Re + Im * Im);
					// System.out.println(String.format("bin%d: % .6f + % .6fi = % .6f",i, Re, Im ,mag));
				}
				binnedFFT[i] += mag;
				max = Math.max(max, mag); // find max as we go
			}

		}

		// Divide by the number of windows (to get the mean value)
		// Then divide by the maximum of the windowed FFT values (found before combination)
		// Note this does not mean the new max of binnedFFT will be one, it will be less than one if one window is stronger than the others
		scale(binnedFFT, 1 / (max * numWindows));
		/*for (int i=0; i < numBins; i++) {
			System.out.println(String.format("F%d=% .6f",i, binnedFFT2[i]));
		}*/

		String line = DF8.format(f1) + "," + DF8.format(p1);
		line += "," + DF8.format(f33) + "," + DF8.format(p33);
		line += "," + DF8.format(H); //entropy

		for (int i=0; i < numBins; i++) {
			line += "," + DF8.format(binnedFFT[i]);
        }

		return line;
	}

	/* From table in paper:
	 * A universal, accurate intensity-based classification of different physical
	 * activities using raw data of accelerometer.
	 * Henri Vaha-Ypya, Tommi Vasankari, Pauliina Husu, Jaana Suni and Harri Sievanen
	 */
	private String calculateMADFeatures(
			double[] paVals) {

		// used in calculation
		int n = paVals.length;
		double N = (double) n; // avoid integer arithmetic
		double vmMean = mean(paVals);
		double vmStd = std(paVals, vmMean);

		// features from paper:
		double MAD = 0; // Mean amplitude deviation (MAD) describes the typical distance of data points about the mean
		double MPD = 0; // Mean power deviation (MPD) describes the dispersion of data points about the mean
		double skew = 0; // Skewness (skewR) describes the asymmetry of dispersion of data points about the mean
		double kurt = 0; // Kurtosis (kurtR) describes the peakedness of the distribution of data points
		for (int c = 0; c < n; c++) {
			double diff = paVals[c] - vmMean;
			MAD += Math.abs(diff);
			MPD += Math.pow(Math.abs(diff), 1.5);
			skew += Math.pow(diff/vmStd, 3);
			kurt += Math.pow(diff/vmStd, 4);
		}

		MAD /= N;
		MPD /= Math.pow(N, 1.5);
		skew *= N / ((N-1)*(N-2));
		kurt = kurt * N*(N+1)/((N-1)*(N-2)*(N-3)*(N-4)) - 3*(N-1)*(N-1)/((N-2)*(N-3));
//		System.out.println(DF6.format(MAD)+","
//				+DF6.format(MPD)+","
//				+DF6.format(skew)+","
//				+DF6.format(kurt));
//		System.out.println(DF6.format(Math.pow(N, 1.5)));
//		System.out.println(DF6.format(Math.pow(1.5,2.2)));
//		System.out.println(DF6.format(N));
//		System.out.println(DF6.format(N / (((double) N-1)*((double) N-2))));
//		System.out.println(DF6.format(N*(N+1)/((N-1)*(N-2)*(N-3)*(N-4))));
//		System.exit(0);
		return DF6.format(MAD)+","
				+DF6.format(MPD)+","
				+DF6.format(skew)+","
				+DF6.format(kurt);
	}

	public static void testFFTProperties(int n, double cycles, boolean useWindow, boolean debug, boolean showInput) {
		System.out.println( String.format("n=%d, cycles=%.2f, useWindow=%b debug=%b",n, cycles, useWindow, debug));
		double[] signal = new double[n];
		for (int c = 0; c < n; c++) {
			 signal[c] = Math.sin(c * 2 * cycles * Math.PI / n); //+ Math.cos(c * 2 * Math.PI * 2 / intendedSampleRate);

			 int w = 20;
			 int a = (int) Math.round(Math.abs(signal[c])*10);

			 if (showInput) {
				 if (signal[c]<0) System.out.println(String.format("[%2d]",c) + new String(new char[w-a]).replace("\0", " ") + new String(new char[a]).replace("\0", "=") + ":" +  new String(new char[w]).replace("\0", " "));
				 else System.out.println(String.format("[%2d]",c) + " " + new String(new char[w]).replace("\0", " ") + ":" + new String(new char[a]).replace("\0", "="));
			 }
		}
		if (debug) System.out.println(Arrays.toString(signal).replace(',', '\n'));

		if (useWindow) HanningWindow(signal, n);

		DoubleFFT_1D transformer = new DoubleFFT_1D(n);
		transformer.realForward(signal);

		if (debug) System.out.println(Arrays.toString(signal).replace(',', '\n'));

		int m = (int) Math.ceil(n/2); // output array size
		double[] output = new double[m];
		output[0] = Math.abs(signal[0])/m;
		for (int i = 1; i < m; i++) {
			double Re = signal[i*2];
			double Im = signal[i*2 + 1];
			output[i] = Math.sqrt(Re * Re + Im * Im)/m;
		}
		System.out.println("FFT output:");
		for (int i = 0; i < Math.min(m, 22); i++) System.out.println( String.format("[%2d]",i) + new String(new char[(int) Math.max(0, output[i]*20)]).replace("\0", "="));
		if (debug) System.out.println(Arrays.toString(output).replace(',', '\n'));

	}


	public static double[] HanningWindow(double[] signal_in, int size)
	{
	    for (int i = 0; i < size; i++)
	    {
	        signal_in[i] = (double) (signal_in[i] * 0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / (size-1))));
	    }
	    return signal_in;
	}



	public static double[] getFFTmagnitude(double[] FFT) {
		return getFFTmagnitude(FFT, FFT.length);
	}

	/* converts FFT library's complex output to only absolute magnitude */
	public static double[] getFFTmagnitude(double[] FFT, int n) {
		if (n<1) {
			System.err.println("cannot get FFT magnitude of array with zero elements");
			return new double[] {0.0};
		}

		/*
		if n is even then
		 a[2*k] = Re[k], 0<=k<n/2
		 a[2*k+1] = Im[k], 0<k<n/2
		 a[1] = Re[n/2]
		e.g for n = 6: (yes there will be 7 array elements for 4 magnitudes)
		a = { Re[0], Re[3], Re[1], Im[1], Re[2], Im[2], Im[3]}

		if n is odd then
		 a[2*k] = Re[k], 0<=k<(n+1)/2
		 a[2*k+1] = Im[k], 0<k<(n-1)/2
		 a[1] = Im[(n-1)/2]
		e.g for n = 7: (again there will be 7 array elements for 4 magnitudes)
		a = { Re[0], Im[3], Re[1], Im[1], Re[2], Im[2], Re[3]}

		*/
		int m = 1 + (int) Math.floor(n/2); // output array size
		double[] output = new double[m];
		double Re, Im;


		output[0] = FFT[0];
		for (int i = 1; i < m-1; i++) {
			Re = FFT[i*2];
			Im = FFT[i*2 + 1];
			output[i] = Math.sqrt(Re * Re + Im * Im);
		}
		// highest frequency will be
		output[m-1] = Math.sqrt(FFT[1] * FFT[1] + FFT[m] * FFT[m]);
		return output;
	}

	/*
	 * Gets [numEachAxis] FFT bins for each of the 3 axes and combines them into 'mfft' column.
	 * If getEachAxis is true it will output the FFT bins for each axis separately
	 */
	private String getFFT3D(double[] x, double[] y, double[] z, double[] vm) {
		String featureString = " ";
		int n = x.length;
		DoubleFFT_1D transformer = new DoubleFFT_1D(n);
		double[] input = new double[n*2];

		int m = 1 + (int) Math.floor(n/2); // output array size
		double[] output = new double[m];

        for (int c = 0; c < n; c++) {
        	input[c] = x[c];
        }
		HanningWindow(input, n);
		transformer.realForward(input);
        output = EpochWriter.getFFTmagnitude(input, n);
        if (getEachAxis) {
        	for (int c = 0; c < numEachAxis; c++) {
            	featureString += DF3.format(output[c])+",";
            }
        }

//        System.out.println(n + " < " + output.length + "?");
        input = new double[n*2];
        for (int c = 0; c < n; c++) {
        	input[c] = y[c];
        }
		HanningWindow(input, n);
		transformer.realForward(input);
        input = EpochWriter.getFFTmagnitude(input, n);
        if (getEachAxis) {
        	for (int c = 0; c < numEachAxis; c++) {
        		featureString += DF3.format(input[c])+",";
        	}
        } else {
	        for (int c = 0; c < m; c++) {
	        	output[c] += input[c];
	        }
        }

        input = new double[n*2];
        for (int c = 0; c < n; c++) {
        	input[c] = z[c];
        }
		HanningWindow(input, n);
		transformer.realForward(input);
        input = EpochWriter.getFFTmagnitude(input, n);
        if (getEachAxis) {
        	for (int c = 0; c < numEachAxis; c++) {
            	featureString += DF3.format(input[c])+",";
            }
        	input = new double[n*2];
            for (int c = 0; c < n; c++) {
            	input[c] = vm[c];
            }
    		HanningWindow(input, n);
    		transformer.realForward(input);
            input = EpochWriter.getFFTmagnitude(input, n);
            for (int c = 0; c < numEachAxis; c++) {
            	featureString += DF3.format(input[c])+",";
            }
        } else {
	        for (int c = 0; c < m; c++) {
	        	output[c] = (output[c]+input[c])/3;
	        }


	        for (int c = 0; c < numEachAxis; c++) {
	        	featureString += DF3.format(output[c])+",";
	        }
        }
//        System.out.println(input.length+" - " +Arrays.toString(input));

//        System.out.println(output.length+" - " +Arrays.toString(output));
        // remove trailing ','
        if (featureString.charAt(featureString.length()-1)==',') {
        	featureString = featureString.substring(0, featureString.length()-1);
        }
		return featureString;
	}

	/* Physical Activity Classification using the GENEA Wrist Worn Accelerometer
		Shaoyan Zhang, Alex V. Rowlands, Peter Murray, Tina Hurst
	*/
	private String unileverFeatures(double[] paVals) {
		String output = "";

		int n = paVals.length;
		double FFTinterval = intendedSampleRate / (1.0 * n); // (Hz)
		double binSize = 0.1; // desired size of each FFT bin (Hz)
		double maxFreq = 15; // min and max for searching for dominant frequency
		double minFreq = 0.3;

		int numBins = (int) Math.ceil((maxFreq-minFreq)/binSize);
//		System.out.println("samplerate" + intendedSampleRate + ", n =" + n +", interval " + FFTinterval);

		DoubleFFT_1D transformer = new DoubleFFT_1D(n);
		double[] vmFFT = new double[n * 2];
		// set input data array
        for (int c = 0; c < n; c++) {
        	// NOTE: this code will generate peaks at 10Hz, and 2Hz (as expected).
        	// Math.sin(c * 2 * Math.PI * 10 / intendedSampleRate) + Math.cos(c * 2 * Math.PI * 2 / intendedSampleRate);
        	vmFFT[c] = paVals[c];
        }

		HanningWindow(vmFFT, n);
        // FFT
		transformer.realForward(vmFFT);

        // find dominant frequency, second dominant frequency, and dominant between .6 - 2.5Hz
 		double f1=-1, f2=-1, f625=-1, f33=-1;
 		double p1=0, p2=0, p625=0, p33=0;

 		double totalPower = 0;
		int out_n = (int) Math.ceil(n/2); // output array size

//		for (int i = 1; i < out_n; i++) {
//	    	double mag = Math.sqrt(vmFFT[i*2]*vmFFT[i*2] + vmFFT[i*2+1]*vmFFT[i*2+1]);
//	    	double freq = FFTinterval * i;
// 			if (freq<minFreq || freq>maxFreq) continue;
//
//	    	System.out.println(DF3.format(freq) + ": " + mag);
//		}

 		for (int i = 1; i < out_n; i++) {
 			double freq = FFTinterval * i;
 			if (freq<minFreq || freq>maxFreq) continue;
        	double mag = Math.sqrt(vmFFT[i*2]*vmFFT[i*2] + vmFFT[i*2+1]*vmFFT[i*2+1]);///(n/2);
        	totalPower += mag;
        	if (mag>p1) {
        		f2 = f1;
        		p2 = p1;
        		f1 = freq;
        		p1 = mag;
        	} else if (mag > p2) {
        		f2 = freq;
        		p2 = mag;
        	}
        	if (mag>p625 && freq > 0.6 && freq < 2.5) {
        		f625 = freq;
        		p625 = mag;
        	}

        	int w = 20;
			int a = (int) Math.round(Math.abs(mag)*10);

//			if (mag<0) System.out.println(String.format("[% 4.1f]",freq) + new String(new char[w-a]).replace("\0", " ") + new String(new char[a]).replace("\0", "=") + ":" +  new String(new char[w]).replace("\0", " "));
//			else System.out.println(String.format("[% 4.1f]",freq) + " " + new String(new char[w]).replace("\0", " ") + ":" + new String(new char[a]).replace("\0", "="));

 		}


//		System.out.println("p1: " + DF3.format(p1) + " f1: " + DF2.format(f1));
//		System.out.println("f1:   " + f1 + ", p1: " + p1);
//		System.out.println("f2:   " + f2 + ", p2: " + p2);
//		System.out.println("f625: " + f625 + ", p625: " + p625);
//		System.out.println("total:" + totalPower);
//
// 		System.exit(0);
 		output = DF3.format(f1)+","+DF3.format(p1)+","+DF3.format(f2)+","+DF3.format(p2)+","+DF3.format(f625)+","+DF3.format(p625)+","+DF3.format(totalPower);

 		if (!get3DFourier) {
	 		output += ",";
	 		for(int i=0; i<numEachAxis; i++) {
	        	double mag = Math.sqrt(vmFFT[i*2]*vmFFT[i*2] + vmFFT[i*2+1]*vmFFT[i*2+1]);///(n/2);
	        	output += DF3.format(mag)+",";
			}
	 		output += "0.0,";
 		}
 		return output;
	}

	/* method to write FFT data to a separate _fft.csv file (not used anymore)  */
	private void writeFFTEpoch(LocalDateTime epochStartTime, double[] paVals) {

		// num samples
		int n = paVals.length;
		// FFT frequency interval = sample frequency / num samples
		double FFTinterval = intendedSampleRate / (1.0 * n); // (Hz)
//		System.out.println("samplerate" + intendedSampleRate + ", n =" + n +", interval " + FFTinterval);
		double binSize = 3; // desired size of each FFT bin (Hz)
		int numBins = 15;
		if (!fftWriterHasHeader) {

			// write the header for fft data (number of columns is dependent on sample frequency)

			String line = "Time";
			// x axis is  x * sample-rate / num-samples

			line += ",mag,f1,p1,f2,p2,f625,p625";
	        for (int i = 1; i < numBins + 1; i++) {
				line += "," + (i * FFTinterval) + "Hz";
	        }

	        writeLine(fftWriter,line);
	        fftWriterHasHeader = true;
		}

		DoubleFFT_1D transformer = new DoubleFFT_1D(n);
		double[] vmFFT = new double[n * 2];
		// set input data array
        for (int c = 0; c < n; c++) {
        	// NOTE: this code will generate peaks at 10Hz, and 2Hz (as expected).
        	// Math.sin(c * 2 * Math.PI * 10 / intendedSampleRate) + Math.cos(c * 2 * Math.PI * 2 / intendedSampleRate);
        	vmFFT[c] = paVals[c];
        }

        // FFT
		transformer.realForward(vmFFT);
		// a useful explanation of output:
		// http://stackoverflow.com/questions/4364823/how-do-i-obtain-the-frequencies-of-each-value-in-a-fft

		// find dominant frequency, second dominant frequency, and dominant between .6 - 2.5Hz
		double f1=-1, f2=-1, f625=-1, f33=-1;
		double p1=0, p2=0, p625=0, p33=0;

		double[] binnedFFT = new double[numBins];
		double total = 0.0;     // sum of P
		double totalPLnP = 0.0; // sum of P * ln(P)
		int currBin = 0;
		int numInBin = 0;
//		for (int i = 1; i < n/2 + 1; i++) {
//        	double mag = vmFFT[i*2]*vmFFT[i*2] + vmFFT[i*2+1]*vmFFT[i*2+1];
//        	double freq = FFTinterval * i;
//        	System.out.println(freq + ": " + mag);
//		}
		for (int i = 1; i < n/2 + 1; i++) {
        	double mag = Math.sqrt(vmFFT[i*2]*vmFFT[i*2] + vmFFT[i*2+1]*vmFFT[i*2+1])/(n/2);
        	double freq = FFTinterval * i;
        	total += mag;
        	totalPLnP += mag * Math.log(mag);
        	if (mag>p1) {
        		f2 = f1;
        		p2 = p1;
        		f1 = freq;
        		p1 = mag;
        	} else if (mag > p2) {
        		f2 = freq;
        		p2 = mag;
        	}
        	if (freq > 0.6 && freq < 2.5 && mag>p625) {
        		f625 = freq;
        		p625 = mag;
        	}
        	if (freq > 0.3 && freq < 3 && mag>p33) {
        		f33 = freq;
        		p33 = mag;
        	}


        	binnedFFT[currBin] += mag;
        	numInBin++;

        	if (freq > currBin * binSize) {
        		binnedFFT[currBin] /= numInBin; // get average
//        		System.out.println(currBin + " "+currBin * binSize + "Hz: "+binnedFFT.length + ", " + binnedFFT[currBin]);
        		currBin++;
        		numInBin = 0;
        		if (currBin >= numBins) break;
        		else binnedFFT[currBin] = 0.0;
        	}

		}

		// Feature Extraction of EEG Signals Using Power Spectral Entropy (2008)
		// normalized p = P / sum(P)
		// H = sum(p * ln(p))
		// H = sum(P * ln(P) / sum(P)) - N * ln(sum(P))
		double H = totalPLnP / total - (n/2) * Math.log(total);

//		System.out.println("p1: " + DF3.format(p1) + " f1: " + DF2.format(f1) + " H =" +H);
//		System.out.println("f1:   " + f1 + ", p1: " + p1);
//		System.out.println("f2:   " + f2 + ", p2: " + p2);
//		System.out.println("f625: " + f625 + ", p625: " + p625);

		// print to fftWriter
		String line = epochStartTime.format(timeFormat);
		line += "," + DF6.format(vmFFT[0]);
		line += "," + DF6.format(f1) + "," + DF6.format(p1);
		line += "," + DF6.format(f2) + "," + DF6.format(p2);
		line += "," + DF6.format(f625) + "," + DF6.format(p625);
		line += "," + DF6.format(f33) + "," + DF6.format(p33);

		for (int i=0; i < numBins; i++) {
			line += "," + DF6.format(binnedFFT[i]);
        }
        writeLine(fftWriter,line);
//        if (Math.random()<0.01)
//        	System.exit(1);
        // this line can check that inverse is the same
		// transformer.realInverse(vmFFT, true);
	}

	private static double getVectorMagnitude(double x, double y, double z) {
		return Math.sqrt(x * x + y * y + z * z);
	}

	private static void abs(double[] vals) {
		for (int c = 0; c < vals.length; c++) {
			vals[c] = Math.abs(vals[c]);
		}
	}

	private static void trunc(double[] vals) {
		double tmp;
		for (int c = 0; c < vals.length; c++) {
			tmp = vals[c];
			if (tmp < 0) {
				tmp = 0;
			}
			vals[c] = tmp;
		}
	}

	private static double sum(double[] vals) {
		if (vals.length == 0) {
			return Double.NaN;
		}
		double sum = 0;
		for (int c = 0; c < vals.length; c++) {
			if (!Double.isNaN(vals[c])) {
				sum += vals[c];
			}
		}
		return sum;
	}

	private static double mean(double[] vals) {
		if (vals.length == 0) {
			return Double.NaN;
		}
		return sum(vals) / (double) vals.length;
	}

	private static double mean(List<Double> vals) {
		if (vals.size() == 0) {
			return Double.NaN;
		}
		return sum(vals) / (double) vals.size();
	}

	private static double sum(List<Double> vals) {
		if (vals.size() == 0) {
			return Double.NaN;
		}
		double sum = 0;
		for (int c = 0; c < vals.size(); c++) {
			sum += vals.get(c);
		}
		return sum;
	}

	private static double range(double[] vals) {
		if (vals.length == 0) {
			return Double.NaN;
		}
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		for (int c = 0; c < vals.length; c++) {
			if (vals[c] < min) {
				min = vals[c];
			}
			if (vals[c] > max) {
				max = vals[c];
			}
		}
		return max - min;
	}

	// standard deviation
	private static double std(double[] vals, double mean) {
		if (vals.length == 0) {
			return Double.NaN;
		}
		double var = 0; // variance
		double len = vals.length; // length
		for (int c = 0; c < len; c++) {
			if (!Double.isNaN(vals[c])) {
				var += Math.pow((vals[c] - mean), 2);
			}
		}
		return Math.sqrt(var / len);
	}

	// same as above but matches R's (n-1) denominator (Bessel's correction)
	private static double stdR(double[] vals, double mean) {
		if (vals.length == 0) {
			return Double.NaN;
		}
		double var = 0; // variance
		double len = vals.length; // length
		for (int c = 0; c < len; c++) {
			if (!Double.isNaN(vals[c])) {
				var += Math.pow((vals[c] - mean), 2);
			}
		}
		return Math.sqrt(var / (len-1));
	}

	private static double Correlation(double[] vals1, double[] vals2) {
		return Correlation(vals1, vals2, 0);
	}
	private static double Correlation(double[] vals1, double[] vals2, int lag) {
		lag = Math.abs(lag); // should be identical
		if ( vals1.length <= lag || vals1.length != vals2.length ) {
			return Double.NaN;
		}
		double sx = 0.0;
		double sy = 0.0;
		double sxx = 0.0;
		double syy = 0.0;
		double sxy = 0.0;

		int nmax = vals1.length;
		int n = nmax - lag;

		for(int i = lag; i < nmax; ++i) {
			double x = vals1[i-lag];
			double y = vals2[i];

			sx += x;
			sy += y;
			sxx += x * x;
			syy += y * y;
			sxy += x * y;
		}

		//System.out.println(sx + ", " + sy+", "+sxx+", "+ syy+", "+sxy);
		// covariation
		double cov = sxy / n - sx * sy / n / n;
		// standard error of x
		double sigmax = Math.sqrt(sxx / n -  sx * sx / n / n);
		// standard error of y
		double sigmay = Math.sqrt(syy / n -  sy * sy / n / n);

		// correlation is just a normalized covariation
		return cov / sigmax / sigmay;
	}


	// covariance of two signals (with lag in samples)
	private static double covariance(double[] vals1, double[] vals2, double mean1, double mean2, int lag) {
		lag = Math.abs(lag); // should be identical
		if ( vals1.length <= lag || vals1.length != vals2.length ) {
			return Double.NaN;
		}
		double cov = 0; // covariance
		for (int c = lag; c < vals1.length; c++) {
			if (!Double.isNaN(vals1[c-lag]) && !Double.isNaN(vals2[c])) {
				cov += (vals1[c]-mean1) * (vals2[c]-mean2);
			}
		}
		cov /= vals1.length+1-lag;
		return cov;
	}

	/*
	 * Implementation of the following features aims to match the paper:
	 * Hip and Wrist Accelerometer Algorithms for Free-Living Behavior Classification
	 * Katherine Ellis, Jacqueline Kerr, Suneeta Godbole, John Staudenmayer, and Gert Lanckriet
	 */

	// percentiles = {0.25, 0.5, 0.75}, to calculate 25th, median and 75th percentile
	private static double[] percentiles(double[] vals, double[] percentiles) {
		double[] output = new double[percentiles.length];
		int n = vals.length;
		if (n == 0) {
			Arrays.fill(output, Double.NaN);
			return output;
		}
		if (n == 1) {
			Arrays.fill(output, vals[0]);
			return output;
		}
		double[] sortedVals = vals.clone();
		Arrays.sort(sortedVals);
		for (int i = 0; i<percentiles.length; i++) {
			// this follows the R default (R7) interpolation model
			// https://en.wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample
			double h = percentiles[i] * (n-1) + 1;
			if (h<=1.0) {
				output[i] = sortedVals[0];
				continue;
			}
			if (h>=n) {
				output[i] = sortedVals[n-1];
				continue;
			}
			// interpolate using: x[h] + (h - floor(h)) (x[h + 1] - x[h])
			int hfloor = (int) Math.floor(h);
			double xh = sortedVals[hfloor-1] ;
			double xh2 = sortedVals[hfloor] ;
			output[i] = xh + (h - hfloor) * (xh2 - xh);
			//S ystem.out.println(percentiles[i] + ", h:" + h + ", " + xh + ", " + xh2);
		}
		return output;
	}

	// returns {mean, standard deviation} together to reduce processing time
	private static double[] angleAvgStd(double[] vals1, double[] vals2) {
		int len = vals1.length;
		if ( len < 2 || len != vals2.length ) {
			return new double[] {Double.NaN, Double.NaN};
		}
		double[] angles = new double[len];
		double total = 0.0;
		for (int c = 0; c < len; c++) {
			angles[c] = Math.atan2(vals1[c],vals2[c]);
			total += angles[c];
		}
		double mean = total/len;
		double var = 0.0;
		for (int c = 0; c < len; c++) {
			var += Math.pow(angles[c] - mean, 2);
		}
		double std = Math.sqrt(var/(len-1)); // uses R's (n-1) denominator standard deviation (Bessel's correction)
		return new double[] {mean, std};
	}


	private static double correlation(double[] vals1, double[] vals2, double mean1, double mean2, int lag) {
		return covariance(vals1, vals2, mean1, mean2, lag)/(mean1*mean2);
	}

	private static double max(double[] vals) {
		double max = Double.NEGATIVE_INFINITY;
		for (int c = 0; c < vals.length; c++) {
			max = Math.max(vals[c], max);
		}
		return max;
	}

	private static void scale(double[] vals, double scale) {
		for (int c = 0; c < vals.length; c++) {
			vals[c] = vals[c] * scale;
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

}
