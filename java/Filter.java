//BSD 2-Clause, (c) 2014: A.Doherty (Oxford), D.Jackson, N.Hammerla (Newcastle)
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
		int m, i;
		
		z[BUTTERWORTH4_NUM_COEFFICIENTS - 1] = 0;
		for (m = offset; m < offset + count; m++) {
			double oldXm = X[m];
			double newXm = B[0] * oldXm + z[0];
			for (i = 1; i < BUTTERWORTH4_NUM_COEFFICIENTS; i++) {
				z[i - 1] = B[i] * oldXm + z[i] - A[i] * newXm;
			}
			X[m] = newXm;
		}
	}
	
	// Additionally, returns the filtered-out signal
	public double[] filterWithRemainder(double[] X, int offset, int count) {
		double[] remainder = X.clone();
		filter(X, offset, count);
		for (int m = 0; m < X.length; m++) {
			if (m<offset || m>=offset+count) {
				remainder[m] = 0;
			} else {
				remainder[m] = X[m] - remainder[m];
			}
		}
		return remainder;
	}
	
	// Apply the filter to the specified data
	public void filter(double[] X) {
		filter(X, 0, X.length);
	}		

	
}
