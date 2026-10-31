package bench;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.stream.LongStream;
import smile.base.cart.SplitRule;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.type.DataTypes;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.data.vector.IntVector;

/**
 * Trains the Smile Random Forest on the SAME temporally-split, already-encoded CSVs that Python wrote,
 * then writes val/test scores in the format s08_evaluate_scores.py expects.
 *
 * Class imbalance handling: fraud rows in the TRAINING set are replicated `fraudweight` times and Smile's
 * own classWeight argument is left at {1,1}. Passing {1,90} via Smile's classWeight produced near-constant
 * scores (AUC ~0.54) on BAF, so we use explicit oversampling instead, which is library-independent and
 * documentable. Validation and test sets are never modified.
 *
 * The replication factor defaults to "auto" = round(#legitimate / #fraud) counted in the TRAINING file only
 * (identical to the Python side, common.train_class_weight). A number overrides it for the sensitivity check:
 *   train --fraudweight 90 --tag w90   -> writes scores_smile_w90_*.csv only; rf_smile.ser is NOT overwritten.
 */
public final class TrainSmile {
    public static void run(String[] args) throws Exception {
        var o = LatencyBench.parse(args);
        Path outDir = Path.of(o.getOrDefault("out", "../results"));
        int trees = Integer.parseInt(o.getOrDefault("trees", "100"));
        int depth = Integer.parseInt(o.getOrDefault("depth", "20"));
        int maxNodes = Integer.parseInt(o.getOrDefault("maxnodes", "500"));
        int nodeSize = Integer.parseInt(o.getOrDefault("nodesize", "5"));
        String fwArg = o.getOrDefault("fraudweight", "auto");
        String tag = o.getOrDefault("tag", "");
        String name = tag.isEmpty() ? "smile" : "smile_" + tag;
        long seed = Long.parseLong(o.getOrDefault("seed", "42"));

        Csv tr = Csv.read(Path.of(o.getOrDefault("train", "../data/processed/train.csv")));
        Csv va = Csv.read(Path.of(o.getOrDefault("val", "../data/processed/val.csv")));
        Csv te = Csv.read(Path.of(o.getOrDefault("test", "../data/processed/test.csv")));
        String[] names = tr.featureNames();
        int mtry = Math.max(1, (int) Math.sqrt(names.length));   // same as sklearn max_features="sqrt"

        // ---- explicit oversampling of fraud rows (training set only) ----
        double[][] xTr = tr.features();
        int[] yTr = tr.labels();
        int nFraud = 0;
        for (int v : yTr) if (v == 1) nFraud++;
        if (nFraud == 0) throw new IllegalStateException("no fraud cases in the training file");
        int fw = fwArg.equalsIgnoreCase("auto")
                ? (int) Math.round((double) (yTr.length - nFraud) / nFraud)
                : Integer.parseInt(fwArg);
        if (fw < 1) throw new IllegalArgumentException("fraudweight must be >= 1");
        System.out.printf(Locale.ROOT, "fraud weight = %d (%s; training rows: %d legit / %d fraud = %.2f)%n",
                fw, fwArg.equalsIgnoreCase("auto") ? "auto, training months only" : "fixed, sensitivity check",
                yTr.length - nFraud, nFraud, (double) (yTr.length - nFraud) / nFraud);
        int nOver = xTr.length + nFraud * (fw - 1);
        double[][] xs = new double[nOver][];
        int[] ys = new int[nOver];
        int j = 0;
        for (int i = 0; i < xTr.length; i++) {
            int reps = (yTr[i] == 1) ? fw : 1;
            for (int r = 0; r < reps; r++) {
                xs[j] = xTr[i];
                ys[j] = yTr[i];
                j++;
            }
        }
        System.out.printf(Locale.ROOT, "oversampling: %d fraud rows x%d -> %d training rows (fraud share %.2f%%)%n",
                nFraud, fw, nOver, 100.0 * nFraud * fw / nOver);

        DataFrame df = DataFrame.of(xs, names).merge(IntVector.of("label", ys));
        long t0 = System.nanoTime();
        RandomForest model = RandomForest.fit(Formula.lhs("label"), df, trees, mtry, SplitRule.GINI,
                depth, maxNodes, nodeSize, 1.0, new int[]{1, 1}, LongStream.range(seed, seed + trees));
        double trainSec = (System.nanoTime() - t0) / 1e9;
        System.out.printf(Locale.ROOT, "trained %d trees on %d rows in %.1fs%n", trees, nOver, trainSec);

        Files.createDirectories(outDir.resolve("models"));
        Files.createDirectories(outDir.resolve("tables"));
        if (tag.isEmpty()) {
            try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(
                    new FileOutputStream(outDir.resolve("models/rf_smile.ser").toFile())))) {
                oos.writeObject(model);
            }
        }

        StructField[] f = new StructField[names.length];
        for (int i = 0; i < f.length; i++) f[i] = new StructField(names[i], DataTypes.DoubleType);
        StructType schema = new StructType(f);
        score(model, schema, va, outDir.resolve("tables/scores_" + name + "_val.csv"));
        score(model, schema, te, outDir.resolve("tables/scores_" + name + "_test.csv"));
        Files.writeString(outDir.resolve("tables/train_" + name + "_info.json"), String.format(Locale.ROOT,
                "{\"train_seconds\": %.3f, \"trees\": %d, \"max_depth\": %d, \"max_nodes\": %d, \"node_size\": %d, "
                        + "\"mtry\": %d, \"fraud_weight\": %d, \"fraud_weight_source\": \"%s\", \"weighting\": \"oversample\", "
                        + "\"train_rows_original\": %d, \"train_fraud\": %d, \"train_rows_oversampled\": %d, \"seed\": %d, \"java\": \"%s\"}%n",
                trainSec, trees, depth, maxNodes, nodeSize, mtry, fw,
                fwArg.equalsIgnoreCase("auto") ? "auto: round(legit/fraud) in training months only" : "fixed (sensitivity check)",
                xTr.length, nFraud, nOver, seed, System.getProperty("java.version")));
    }

    private static void score(RandomForest m, StructType schema, Csv d, Path out) throws Exception {
        double[][] x = d.features();
        int[] y = d.labels();
        double[] post = new double[2];
        StringBuilder sb = new StringBuilder("label,score\n");
        for (int i = 0; i < x.length; i++) {
            Tuple t = Tuple.of(x[i], schema);
            m.predict(t, post);
            sb.append(y[i]).append(',').append(String.format(Locale.ROOT, "%.9g", post[1])).append('\n');
        }
        Files.writeString(out, sb.toString());
        System.out.println("wrote " + out);
    }
}