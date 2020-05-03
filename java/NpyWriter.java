import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.zip.GZIPOutputStream;


public class NpyWriter {

    private static final Boolean COMPRESS = true;
    private String outputFile;
	private File file;
    private RandomAccessFile raf;
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
	private int bytesPerLine = (Long.BYTES + Float.BYTES * 3);
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
        this.outputFile = outputFile;

		try {
            file = new File(outputFile);
			raf = new RandomAccessFile(file, "rw");
			raf.setLength(0); // clear file

			// generate dummy header (maybe just use real header?)
			int hdrLen = NPY_HEADER.length + 3; // +3 for three extra bytes due to NPY_(MIN/MAX)_VERSION and \n
			String filler = new String(new char[HEADER_SIZE + hdrLen]).replace("\0", " ") +"\n";
			raf.writeBytes(filler);
			itemTypes.add(Long.class); itemNames.add("time");
			itemTypes.add(Float.class);itemNames.add("x");
			itemTypes.add(Float.class);itemNames.add("y");
			itemTypes.add(Float.class);itemNames.add("z");

		} catch (IOException e) {
			throw new RuntimeException("The .npy file " + outputFile +" could not be created");
		}
	}


	public void writeData(long time, Float x, Float y, Float z) throws IOException {
		lineBuffer.putLong(time);
		lineBuffer.putFloat(x);
		lineBuffer.putFloat(y);
		lineBuffer.putFloat(z);
		if (!lineBuffer.hasRemaining()) {
			raf.write(lineBuffer.array());
			lineBuffer.clear();
		}

		linesWritten+=1;
		if (linesWritten % 10000000 == 0) {writeHeader(); System.out.print(linesWritten +" ");}
	}


	/**
	 * Updates the file's header based on the arrayType and number of array elements written thus far.
	 */
	public void writeHeader() {
		try {

			raf.seek(0);

			raf.write(NPY_HEADER);
			raf.write(NPY_MAJ_VERSION);
			raf.write(NPY_MIN_VERSION);

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

			writeLEShort (raf, (short) HEADER_SIZE);

			raf.writeBytes(dataHeader);
			raf.seek(raf.length());

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
	 * Converts a Java class to a python datatype String
	 */
	private String toDataTypeStr(Class<?> datatype)
	{
		if (datatype == Integer.class || datatype == Integer.TYPE)
			return numpyByteOrder+"i4";
		else if (datatype == Short.class || datatype == Short.TYPE)
			return numpyByteOrder+"i2";
		else if (datatype == Long.class || datatype == Long.TYPE)
			return numpyByteOrder+"i8";
		else if (datatype == Float.class || datatype == Float.TYPE)
            return numpyByteOrder+"f4";
        else if (datatype == Double.class || datatype == Double.TYPE)
            return numpyByteOrder+"f8";
		else
			throw new IllegalArgumentException("Don't know the corresponding Python datatype for " + datatype.getSimpleName());
    }


    private void compress() {
        try {
            GZIPOutputStream zip = new GZIPOutputStream(new FileOutputStream(new File(outputFile+".gz")));
            byte [] buff = new byte[1024];
            int len;
            raf.seek(0);
            while((len=raf.read(buff)) != -1){
                zip.write(buff, 0, len);
            }
            zip.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


	public void close() {
		writeHeader(); // ensure header is correct length

		try {
			// write any remaining data
			raf.write(lineBuffer.array());
			lineBuffer.clear();
		} catch (IOException e) {
			e.printStackTrace();
        }

        if (COMPRESS) {
            System.out.println("\ncompressing .npy file...");
            compress();  // compress created file
        }

		try {
            raf.close();
		} catch (IOException e) {
			e.printStackTrace();
        }

        if (COMPRESS) {
            System.out.println("deleting uncompressed .npy file...");
            file.delete();  // note: raf must be closed first
        }

		System.out.println("npyWriter was shut down correctly");
    }

    
}
