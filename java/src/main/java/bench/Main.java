package bench;

import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;

/** Entry point.   java -Xmx8g -XX:+UseG1GC -jar target/bench.jar <train|latency> [options]  */
public final class Main {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) { usage(); return; }
        String[] rest = Arrays.copyOfRange(args, 1, args.length);
        try {
            switch (args[0]) {
                case "train" -> Class.forName("bench.TrainSmile").getMethod("run", String[].class).invoke(null, (Object) rest);
                case "latency" -> LatencyBench.run(rest);
                default -> usage();
            }
        } catch (InvocationTargetException e) {
            if (e.getCause() instanceof Exception ex) throw ex;
            throw e;
        }
    }

    private static void usage() {
        System.out.println("""
            train   --train ../data/processed/train.csv --val ../data/processed/val.csv --test ../data/processed/test.csv
                    --out ../results [--trees 100 --depth 20 --maxnodes 500 --nodesize 5 --fraudweight auto --seed 42]
                    [--fraudweight 90 --tag w90   (sensitivity check; does not overwrite rf_smile.ser)]
            latency --arm smile|onnx|rest --label NAME --rows ../data/processed/bench_rows.csv
                    --warmuprows ../data/processed/warmup_rows.csv --out ../results
                    [--n 5000] [--warmup 2000] [--model ../results/models/rf_smile.ser | rf_sklearn.onnx] [--url http://127.0.0.1:5000/predict]
            """);
    }
}
