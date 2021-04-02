import java.util.Collections;
import java.util.List;

public class Resample{

    //inspired by http://www.java2s.com/Code/Java/Collections-Data-Structure/LinearInterpolation.htm
    //in small tests this method has matched python scipy.interpolate.interp1d
    //http://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
    public static final void interpLinear(
            List<Long> time, //time in milliseconds
            List<Double> x,
            List<Double> y,
            List<Double> z,
            long[] timeI, //time in milliseconds
            double[] xNew,
            double[] yNew,
            double[] zNew) throws IllegalArgumentException {
        if (time.size() != x.size()) {
            throw new IllegalArgumentException("time and x must be the same length");
        }
        if (time.size() == 1) {
            throw new IllegalArgumentException("time must contain more than one value");
        }
        double dtime;
        double dx;
        double dy;
        double dz;
        double[] xSlope = new double[time.size() - 1];
        double[] ySlope = new double[time.size() - 1];
        double[] zSlope = new double[time.size() - 1];
        double[] xIntercept = new double[time.size() - 1];
        double[] yIntercept = new double[time.size() - 1];
        double[] zIntercept = new double[time.size() - 1];

        // Calculate the line equation (i.e. slope and intercept) between each point
        for (int i = time.size()-2; i >= 0; i--) {
            dtime = time.get(i + 1) - time.get(i);
            if (dtime <= 0){
                time.set(i, time.get(i+1) - 1);
                dtime = 1;
            }
            dx = x.get(i + 1) - x.get(i);
            dy = y.get(i + 1) - y.get(i);
            dz = z.get(i + 1) - z.get(i);
            xSlope[i] = dx / dtime;
            ySlope[i] = dy / dtime;
            zSlope[i] = dz / dtime;
            xIntercept[i] = x.get(i) - time.get(i) * xSlope[i];
            yIntercept[i] = y.get(i) - time.get(i) * ySlope[i];
            zIntercept[i] = z.get(i) - time.get(i) * zSlope[i];
        }

        // Perform the interpolation here
        for (int i = 0; i < timeI.length; i++) {
            if (timeI[i] > time.get(time.size() - 1)) {
                xNew[i] = x.get(time.size() - 1);
                yNew[i] = y.get(time.size() - 1);
                zNew[i] = z.get(time.size() - 1);
            } else if (timeI[i] < time.get(0)) {
                xNew[i] = x.get(0);
                yNew[i] = y.get(0);
                zNew[i] = z.get(0);
            } else {
                int loc = Collections.binarySearch(time, timeI[i]);
                if (loc < -1) {
                    loc = -loc - 2;
                    xNew[i] = xSlope[loc] * timeI[i] + xIntercept[loc];
                    yNew[i] = ySlope[loc] * timeI[i] + yIntercept[loc];
                    zNew[i] = zSlope[loc] * timeI[i] + zIntercept[loc];
                }
                else {
                    xNew[i] = x.get(loc);
                    yNew[i] = y.get(loc);
                    zNew[i] = z.get(loc);
                }
            }
        }
    }


    /** Nearest neighbor interpolation
     */
    public static final void interpNearest(
            List<Long> t, //time in milliseconds
            List<Double> x,
            List<Double> y,
            List<Double> z,
            long[] tNew, //time in milliseconds
            double[] xNew,
            double[] yNew,
            double[] zNew) throws IllegalArgumentException {
        if (t.size() != x.size()) {
            throw new IllegalArgumentException("time and x must be the same length");
        }
        if (t.size() == 1) {
            throw new IllegalArgumentException("time must contain more than one value");
        }

        for (int i = 0; i < tNew.length; i++) {
            int j = nearestIndex(t, tNew[i]);
            xNew[i] = x.get(j);
            yNew[i] = y.get(j);
            zNew[i] = z.get(j);
        }

    }


    /**
     * Find index of the closest element to key
     */
    private static int nearestIndex(List<Long> elems, long key) {
        if (key <= elems.get(0)) { return 0; }
        if (key >= elems.get(elems.size() - 1)) { return elems.size() - 1; }

        int result = Collections.binarySearch(elems, key);
        if (result >= 0) { return result; }

        int insertionPoint = -result - 1;
        return (elems.get(insertionPoint) - key) < (key - elems.get(insertionPoint - 1)) ?
                insertionPoint : insertionPoint - 1;
    }

}
