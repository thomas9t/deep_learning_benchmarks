import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.LinkedBlockingDeque;

import org.apache.sysml.api.jmlc.Connection;
import org.apache.sysml.parser.DataExpression;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.runtime.io.MatrixReader;
import org.apache.sysml.runtime.io.MatrixReaderFactory;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.api.jmlc.PreparedScript;

import org.apache.sysml.utils.Statistics;
import org.apache.wink.json4j.JSONObject;

public class JMLCExecutor {

    public static void main(String[] args) throws Exception {
        Connection conn = new Connection();

        String script;
        script = conn.readScript("prediction.dml");

        String weightsDir = "tmp_dir";
        HashMap<String,MatrixBlock> MM = readAllMatrices(weightsDir);
        String[] matrixNames = getKeysAsString(MM);
        String[] allInputs = new String[matrixNames.length+1];
        System.arraycopy(matrixNames, 0, allInputs, 0, matrixNames.length);
        allInputs[matrixNames.length] = "X_full";
        PreparedScript ps = conn.prepareScript(
            script, allInputs, new String[] {"Prob"}, true, true, 0);
        System.out.println(ps.explain());
        ps.setStatistics(true);

        // for (String m : matrixNames)
        //     ps.setMatrix(m, MM.get(m), true);

        long start = System.nanoTime();
        ps.executeScript();
        System.err.println("JMLC TIME: " + (System.nanoTime() - start));
        System.err.println(ps.statistics());
    }

    private static HashMap<String,MatrixBlock> 
        readAllMatrices(String dirname) throws IOException {
        File loc = new File(dirname);
        HashMap<String,MatrixBlock> MM = new HashMap<>();
        for (File path : loc.listFiles()) {
            System.out.println(path);
            String[] chunks = path.getName().split("/");
            String[] clunks = chunks[chunks.length-1].split("\\.");
            if (!clunks[clunks.length-1].equals("mtx"))
                continue;
            String name = clunks[0];
            MM.put(name, readMatrix(path.getPath()));
        }
        return MM;
    }

    private static String[] getKeysAsString(HashMap<String,MatrixBlock> MM) {
        String[] keys = new String[MM.keySet().size()];
        int ix = 0;
        for (String k : MM.keySet()) {
            keys[ix++] = k;
        }
        return keys;
    }

    static double[][] randomMatrix(
        int rows, int cols, double min, double max, double sparsity) {
        double[][] matrix = new double[rows][cols];
        Random random = new Random(System.currentTimeMillis());
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (random.nextDouble() > sparsity) {
                    continue;
                }
                matrix[i][j] = (random.nextDouble() * (max - min) + min);
            }
        }
        return matrix;
    }

    static MatrixBlock readMatrix(String fname) throws IOException {
        try {
            String fnamemtd = DataExpression.getMTDFileName(fname);
            JSONObject jmtd = new DataExpression().readMetadataFile(fnamemtd, false);

            //parse json meta data
            long rows = jmtd.getLong(DataExpression.READROWPARAM);
            long cols = jmtd.getLong(DataExpression.READCOLPARAM);
            int brlen = jmtd.containsKey(DataExpression.ROWBLOCKCOUNTPARAM)?
                    jmtd.getInt(DataExpression.ROWBLOCKCOUNTPARAM) : -1;
            int bclen = jmtd.containsKey(DataExpression.COLUMNBLOCKCOUNTPARAM)?
                    jmtd.getInt(DataExpression.COLUMNBLOCKCOUNTPARAM) : -1;
            long nnz = jmtd.containsKey(DataExpression.READNNZPARAM)?
                    jmtd.getLong(DataExpression.READNNZPARAM) : -1;
            String format = jmtd.getString(DataExpression.FORMAT_TYPE);
            InputInfo iinfo = InputInfo.stringExternalToInputInfo(format);
            return readMatrix(fname, iinfo, rows, cols, brlen, bclen, nnz);
        } catch (Exception ex) {
            throw new IOException(ex);
        }
    }

    static MatrixBlock readMatrix(
        String fname, InputInfo iinfo, 
        long rows, long cols, int brlen, int bclen, long nnz)
            throws IOException
    {
        try {
            MatrixReader reader = MatrixReaderFactory.createMatrixReader(iinfo);
            return reader.readMatrixFromHDFS(fname, rows, cols, brlen, bclen, nnz);

        }
        catch(Exception ex) {
            throw new IOException(ex);
        }
    }
}