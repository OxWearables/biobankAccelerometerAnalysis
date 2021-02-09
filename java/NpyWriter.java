import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.GZIPOutputStream;


public class NpyWriter {

	private static final int BUFSIZE = 1024;
	private static final ByteOrder NATIVE_BYTE_ORDER = ByteOrder.nativeOrder();
	private static final char NUMPY_BYTE_ORDER = NATIVE_BYTE_ORDER==ByteOrder.BIG_ENDIAN ? '>' : '<';
    private static final boolean COMPRESS = false;
	private final static byte NPY_MAJ_VERSION = 1;
	private final static byte NPY_MIN_VERSION = 0;
	private final static int BLOCK_SIZE = 16;
	private final static int HEADER_SIZE = BLOCK_SIZE * 16;
	private final static byte[] NPY_HEADER;
	static {
		byte[] hdr = "XNUMPY".getBytes(StandardCharsets.US_ASCII);
		hdr[0] = (byte) 0x93;
		NPY_HEADER = hdr;
	}

    private String outputFile;
	private LinkedHashMap<String, String> itemNamesAndTypes;
	private ByteBuffer buf;
	private File file;
    private RandomAccessFile raf;
	private int linesWritten = 0;


	public NpyWriter(String outputFile, LinkedHashMap<String, String> itemNamesAndTypes) {
        this.outputFile = outputFile;
		this.itemNamesAndTypes = itemNamesAndTypes;
		this.buf = ByteBuffer.allocate(BUFSIZE * getBytesPerLine(itemNamesAndTypes)).order(NATIVE_BYTE_ORDER);

		try {
            file = new File(outputFile);
			raf = new RandomAccessFile(file, "rw");
			raf.setLength(0); // clear file

			// generate dummy header (maybe just use real header?)
			int hdrLen = NPY_HEADER.length + 3; // +3 for three extra bytes due to NPY_(MIN/MAX)_VERSION and \n
			String filler = new String(new char[HEADER_SIZE + hdrLen]).replace("\0", " ") +"\n";
			raf.writeBytes(filler);

		} catch (IOException e) {
			throw new RuntimeException("The .npy file " + outputFile +" could not be created");
		}
	}


	public NpyWriter(String outputFile) {
		this(outputFile, getDefaultItemNamesAndTypes());
	}


	public void write(HashMap<String, Object> items) throws IOException {
		putItems(items);

		if (!buf.hasRemaining()) {
			raf.write(buf.array());
			buf.clear();
		}

		linesWritten++;
	}


	private void putItems(HashMap<String, Object> items) {
		for(Map.Entry<String, String> entry : itemNamesAndTypes.entrySet()) {
			String name = entry.getKey();
			String type = entry.getValue();
			Object item = items.get(name);
			putItem(item, type);
		}
	}


	private void putItem(Object item, String type) {

		switch(type) {

			case "Integer":
				buf.putInt((int) item);
				break;

			case "Short":
				buf.putShort((short) item);
				break;

			case "Long":
				buf.putLong((long) item);
				break;

			case "Float":
				buf.putFloat((float) item);
				break;

			case "Double":
				buf.putDouble((double) item);
				break;

			default:
				throw new IllegalArgumentException("Unrecognized item type: " + type);

		}

	}


	public void writeData(long time, Float x, Float y, Float z) throws IOException {
		buf.putLong(time);
		buf.putFloat(x);
		buf.putFloat(y);
		buf.putFloat(z);

		if (!buf.hasRemaining()) {
			raf.write(buf.array());
			buf.clear();
		}

		linesWritten += 1;
	}


	/**
	 * Updates the file's header based on the arrayType and number of array elements written thus far.
	 */
	private void writeHeader() {
		try {

			raf.seek(0);

			raf.write(NPY_HEADER);
			raf.write(NPY_MAJ_VERSION);
			raf.write(NPY_MIN_VERSION);

			// Describes the data to be written. Padded with space characters to be an even
			// multiple of the block size. Terminated with a newline. Prefixed with a header length.
			String dataHeader = "{ 'descr': [";

			int i = 0;
			for (Map.Entry<String, String> entry : itemNamesAndTypes.entrySet()) {
				dataHeader += "('" + entry.getKey() + "','" + toNpyTypeStr(entry.getValue()) + "')";
				if ((i+1) < itemNamesAndTypes.entrySet().size()) dataHeader += ",";
				i++;
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


    public void compress(String compressedOutputFile) {
		finalFlush();

        try(GZIPOutputStream zip = new GZIPOutputStream(new FileOutputStream(new File(compressedOutputFile)))) {
            byte [] buff = new byte[1024];
            int len;
            raf.seek(0);
            while((len=raf.read(buff)) != -1){
                zip.write(buff, 0, len);
            }
        } catch (IOException e) {
            e.printStackTrace();
		}
    }


	public void compress() {
		compress(outputFile+".gz");
	}


	private void finalFlush() {
		writeHeader();  // ensure header is correct length
		try {
			// write any remaining data
			raf.write(buf.array());
			buf.clear();
		} catch (IOException e) {
			e.printStackTrace();
        }
	}


	public void close() {
		finalFlush();

		try {
            raf.close();
			System.out.println("NpyWriter was shut down correctly");
		} catch (IOException e) {
			e.printStackTrace();
        }

    }


	public void closeAndDelete() {
		close();
		file.delete();
	}


	/**
	 * Writes a little-endian short to the given output stream
	 * @param out the stream
	 * @param value the short value
	 * @throws IOException
	 */
	private static void writeLEShort(RandomAccessFile out, short value) throws IOException
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
	private static void writeLEInt(RandomAccessFile out, int value) throws IOException
	{
		System.out.println("writing:" + value);
		out.writeByte(value & 0x00FF);
		out.writeByte((value >> 8) & 0x00FF);
		out.writeByte((value >> 16) & 0x00FF);
		out.writeByte((value >> 24) & 0x00FF);
	}


	private static LinkedHashMap<String, String> getDefaultItemNamesAndTypes() {
		LinkedHashMap<String, String> itemNamesAndTypes = new LinkedHashMap<String, String>();
		itemNamesAndTypes.put("time", "Long");
		itemNamesAndTypes.put("x", "Float");
		itemNamesAndTypes.put("y", "Float");
		itemNamesAndTypes.put("z", "Float");
		return itemNamesAndTypes;
	}


	private static int getBytesPerLine(LinkedHashMap<String, String> itemNamesAndTypes) {
		int bytesPerLine = 0;
		for(String type : itemNamesAndTypes.values()) {
			bytesPerLine += getBytesPerType(type);
		}
		return bytesPerLine;
	}


	private static int getBytesPerType(String type) {
		switch(type) {

			case "Integer":
				return Integer.BYTES;

			case "Short":
				return Short.BYTES;

			case "Long":
				return Long.BYTES;

			case "Float":
				return Float.BYTES;

			case "Double":
				return Double.BYTES;

			default:
				throw new IllegalArgumentException("Unrecognized item type: " + type);

		}
	}


	private static String toNpyTypeStr(String type) {

		switch(type) {

			case "Integer":
				return NUMPY_BYTE_ORDER+"i4";

			case "Short":
				return NUMPY_BYTE_ORDER+"i2";

			case "Long":
				return NUMPY_BYTE_ORDER+"i8";

			case "Float":
				return NUMPY_BYTE_ORDER+"f4";

			case "Double":
				return NUMPY_BYTE_ORDER+"f8";

			default:
				throw new IllegalArgumentException("Unrecognized item type: " + type);

		}

    }
    

}
