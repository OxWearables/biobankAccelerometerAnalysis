import java.util.Arrays;
import java.util.List;

// (This inner class should probably be in another file)
// [dgj] Butterworth 4th-order lowpass filter
public class Filter {

	protected final static int BUTTERWORTH4_ORDER = 4;
	protected static int BUTTERWORTH4_NUM_COEFFICIENTS;

    // Filter coefficients
    protected double B[];
    protected double A[];
    
	// Final/initial conditions
    protected double z[];
	
	
	// Reset state tracking
	public void reset() {
		for (int i = 0; i < z.length; i++) { 
			z[i] = 0; 
		}
	}
	
	// Apply the filter to the specified data
	public void filter(double[] X, int offset, int count) {
		int i, j;
		
		z[BUTTERWORTH4_NUM_COEFFICIENTS - 1] = 0;
		for (i = offset; i < offset + count; i++) {
			double oldXm = X[i];
			double newXm = B[0] * oldXm + z[0];
			for (j = 1; j < BUTTERWORTH4_NUM_COEFFICIENTS; j++) {
				z[j - 1] = B[j] * oldXm + z[j] - A[j] * newXm;
			}
			X[i] = newXm;
		}
	}
	
	// Additionally, returns the filtered-out signal
	public double[] filterWithRemainder(double[] X, int offset, int count) {
		double[] remainder = X.clone();
		filter(X, offset, count);
		for (int i = 0; i < X.length; i++) {
			if (i<offset || i>=offset+count) {
				remainder[i] = 0;
			} else {
				remainder[i] = X[i] - remainder[i];
			}
		}
		return remainder;
	}
	
	// Apply the filter to the specified data
	public void filter(double[] X) {
		filter(X, 0, X.length);
	}		

	
}
