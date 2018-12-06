import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;


public class NpyWriter {
	
	private RandomAccessFile file;
	private int linesWritten = 0;
	
	private final static byte[] NPY_HEADER;
	static {
		byte[] hdr = "XNUMPY".getBytes(StandardCharsets.US_ASCII);
		hdr[0] = (byte) 0x93;
		NPY_HEADER = hdr;
	}
	
	private final static byte NPY_MAJ_VERSION = 1;
	private final static byte NPY_MIN_VERSION = 0;
	private final static int BLOCK_SIZE = 16;
	private final static int HEADER_SIZE = BLOCK_SIZE * 16;
	
	// buffer file output so it's faster
	private int bufferLength = 1000; // number of lines to buffer
	private int bytesPerLine = (Long.BYTES + Short.BYTES * 4);
//	private int bufferPosition = 0;
	private ByteOrder nativeByteOrder = ByteOrder.nativeOrder();
	private char numpyByteOrder = nativeByteOrder==ByteOrder.BIG_ENDIAN ? '>' : '<';
	private ByteBuffer lineBuffer = ByteBuffer.allocate(bufferLength * bytesPerLine).order(nativeByteOrder);
	// column names and types (must remain the same after initialization)
	private ArrayList<Class> itemTypes = new ArrayList<Class>(); 
	private ArrayList<String> itemNames= new ArrayList<String>();
	
	/**
	 * Opens a .npy file for writing (contents are erased) and initializes a dummy header.
	 * @param outputFile filename for the .npy file
	 */
	public NpyWriter(String outputFile) {
		try {
			file = new RandomAccessFile(new File(outputFile), "rw");
			file.setLength(0); // clear file
			
			// generate dummy header (maybe just use real header?)
			int hdrLen = NPY_HEADER.length + 3; // +3 for three extra bytes due to NPY_(MIN/MAX)_VERSION and \n
			String filler = new String(new char[HEADER_SIZE + hdrLen]).replace("\0", " ") +"\n";
			file.writeBytes(filler);
			itemTypes.add(Long.class); itemNames.add("time");
			itemTypes.add(Short.class);itemNames.add("x");
			itemTypes.add(Short.class);itemNames.add("y");
			itemTypes.add(Short.class);itemNames.add("z");
			itemTypes.add(Short.class);itemNames.add("temperature");
			System.out.println(itemNames.toString());
			System.out.println(itemTypes.toString());
			System.out.println("order = "+ lineBuffer.order().toString());
			System.out.println("numpy order = "+ numpyByteOrder);
			
		} catch (IOException e) {
			throw new RuntimeException("The .npy file " + outputFile +" could not be created");
		}
	}
	public void writeData(long time, short x, short y, short z, short temperature) throws IOException {
		lineBuffer.putLong(time);
		lineBuffer.putShort(x);
		lineBuffer.putShort(y);
		lineBuffer.putShort(z);
		lineBuffer.putShort(temperature);
		if (!lineBuffer.hasRemaining()) {
			file.write(lineBuffer.array());
			lineBuffer.clear();
		}

		linesWritten+=1;
		if (linesWritten % 10000000 == 0) {this.writeHeader(); System.out.print(linesWritten +" ");}

	}

	/**
	 * Updates the file's header based on the arrayType and number of array elements written thus far.
	 */
	public void writeHeader() {
		try {

			file.seek(0);
		
			file.write(NPY_HEADER);
			file.write(NPY_MAJ_VERSION);
			file.write(NPY_MIN_VERSION);
			
			// Describes the data to be written. Padded with space characters to be an even
			// multiple of the block size. Terminated with a newline. Prefixed with a header length.
			String dataHeader = "{ 'descr': [";
			
			// Now add to the description our predefined itemNames and itemTypes
			for (int i=0; i < itemNames.size(); i++) {
				dataHeader += "('"+ itemNames.get(i)+"','"+ toDataTypeStr (itemTypes.get(i)) + "')";
				if (i+1!=itemNames.size()) dataHeader+=",";
			}
			
			dataHeader	+= "]"
						+ ", 'fortran_order': False"
						+ ", 'shape': (" + linesWritten + ",), "
						+ "}";

			int hdrLen    = dataHeader.length() + 1; // +1 for a terminating newline.
			if (hdrLen > HEADER_SIZE) {
				throw new RuntimeException("header is too big to be written.");
				// Increase HEADER_SIZE if this happens
			}
			String filler = new String(new char[HEADER_SIZE - hdrLen]).replace("\0", " ");
			
			dataHeader = dataHeader + filler + '\n';

			writeLEShort (file, (short) HEADER_SIZE);
			
			
			file.writeBytes(dataHeader);
			file.seek(file.length());
			
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException("The .npy file could not write a header created");
		}
	}
	
	/**
	 * Writes a little-endian short to the given output stream
	 * @param out the stream
	 * @param value the short value
	 * @throws IOException 
	 */
	private void writeLEShort (RandomAccessFile out, short value) throws IOException
	{

		// convert to little endian
		value = (short) ((short) ((short) value << 8) & 0xFF00 | (value >> 8));

		out.writeShort( value );

	}	
	/**
	 * Writes a little-endian int to the given output stream
	 * @param out the stream
	 * @param value the short value
	 * @throws IOException 
	 */
	public static void writeLEInt(RandomAccessFile out, int value) throws IOException
	{		
		System.out.println("writing:" + value);
		out.writeByte(value & 0x00FF);
		out.writeByte((value >> 8) & 0x00FF);
		out.writeByte((value >> 16) & 0x00FF);
		out.writeByte((value >> 24) & 0x00FF); 
	}
	
	/**
	 * Converts a Java class to a python datatype String. Currently only Integer
	 * and Short are supported.
	 */
	private String toDataTypeStr(Class<?> datatype)
	{
		if (datatype == Integer.class || datatype == Integer.TYPE)
			return numpyByteOrder+"i4";
		else if (datatype == Short.class || datatype == Short.TYPE)
			return numpyByteOrder+"i2";
		else if (datatype == Long.class || datatype == Long.TYPE)
			return numpyByteOrder+"i8";
		else if (datatype == Float.class || datatype == Float.TYPE || datatype == Double.class || datatype == Double.TYPE)
			return numpyByteOrder+"f8";
		else
			throw new IllegalArgumentException("Don't know the corresponding Python datatype for " + datatype.getSimpleName());
	}
	
	public void close() {
		this.writeHeader(); // ensure header is correct length

		try {
			// write any remaining data
			this.file.write(lineBuffer.array());
			lineBuffer.clear();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		try {
			this.file.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("npyWriter was shut down correctly");
	}
}
