package bench;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Closed-loop single-request latency harness (same protocol as the Python arms).
 *  - warm-up: --warmup calls (default 2000, identical for every arm) on VALIDATION rows (warmup_rows.csv,
 *    written by Python); these are never the measured rows
 *  - measured: the first --n rows of bench_rows.csv (TEST month), the SAME rows in the same order as Python
 *  - each measured call = one transaction, timed with System.nanoTime(); raw microseconds are written
 *  - the score of every measured call is written (<label>_scores.csv) for parity checks; their sum is printed,
 *    which also prevents the JIT from eliminating the call
 *  - REST arm: per-call breakdown (<label>_breakdown.csv) with the same columns as the Python client
 */
public final class LatencyBench {
    static Map<String, String> parse(String[] a) {
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i < a.length - 1; i += 2) m.put(a[i].replaceFirst("^--", ""), a[i + 1]);
        return m;
    }

    public static void run(String[] args) throws Exception {
        Map<String, String> o = parse(args);
        String armName = o.getOrDefault("arm", "");
        String label = o.getOrDefault("label", "jvm_" + armName);
        int n = Integer.parseInt(o.getOrDefault("n", "5000"));
        int warmup = Integer.parseInt(o.getOrDefault("warmup", "2000"));
        Path rowsPath = Path.of(o.getOrDefault("rows", "../data/processed/bench_rows.csv"));
        Path warmPath = Path.of(o.getOrDefault("warmuprows", "../data/processed/warmup_rows.csv"));
        Path outDir = Path.of(o.getOrDefault("out", "../results"));

        Csv csv = Csv.read(rowsPath);
        double[][] rows = csv.features();
        double[][] warm = Csv.read(warmPath).features();
        if (n > rows.length) throw new IllegalArgumentException("n exceeds bench_rows.csv length " + rows.length);

        Arm arm;
        switch (armName) {
            case "rest" -> arm = new RestArm(o.getOrDefault("url", "http://127.0.0.1:5000/predict"));
            case "smile" -> arm = load("bench.SmileArm", o.getOrDefault("model", "../results/models/rf_smile.ser"), csv.featureNames());
            case "onnx" -> arm = load("bench.OnnxArm", o.getOrDefault("model", "../results/models/rf_sklearn.onnx"), null);
            default -> throw new IllegalArgumentException("--arm must be smile|onnx|rest");
        }
        RestArm rest = (arm instanceof RestArm r) ? r : null;
        String serverConfig = rest != null ? rest.health() : "null";

        double sink = 0;
        try (arm) {
            double warmSink = 0;
            for (int i = 0; i < warmup; i++) warmSink += arm.score(warm[i % warm.length]);
            if (warmSink == 12345.6789) System.out.print("");

            double[] us = new double[n];
            double[] sc = new double[n];
            double[][] bd = rest != null ? new double[n][8] : null;
            for (int i = 0; i < n; i++) {
                double[] r = rows[i];
                long t0 = System.nanoTime();
                double s = arm.score(r);
                long t1 = System.nanoTime();
                sink += s;
                sc[i] = s;
                us[i] = (t1 - t0) / 1000.0;
                if (rest != null) {
                    bd[i] = new double[]{us[i], rest.lastSerializeUs, rest.lastRoundTripUs, rest.lastParseUs,
                            rest.lastServerHandlerUs, rest.lastServerParseUs, rest.lastModelUs,
                            rest.lastRoundTripUs - rest.lastServerHandlerUs};
                }
            }
            write(outDir, label, us, armName, warmup, o, serverConfig);
            Path raw = outDir.resolve("raw");
            StringBuilder sb = new StringBuilder("score\n");
            double mn = Double.MAX_VALUE, mx = -Double.MAX_VALUE;
            for (double v : sc) { sb.append(String.format(Locale.ROOT, "%.9g%n", v)); mn = Math.min(mn, v); mx = Math.max(mx, v); }
            Files.writeString(raw.resolve(label + "_scores.csv"), sb.toString());
            if (bd != null) {
                StringBuilder b = new StringBuilder("total,client_serialize,round_trip,client_parse,server_handler,server_parse,model,outside_handler\n");
                for (double[] row : bd) {
                    for (int k = 0; k < row.length; k++) { if (k > 0) b.append(','); b.append(String.format(Locale.ROOT, "%.3f", row[k])); }
                    b.append('\n');
                }
                Files.writeString(raw.resolve(label + "_breakdown.csv"), b.toString());
            }
            System.out.printf(Locale.ROOT, "scores: min=%.6f max=%.6f first3=%.6f, %.6f, %.6f%n", mn, mx, sc[0], sc[1], sc[2]);
        }
        System.out.println("score_sum_measured=" + sink + "  (sum of scores over the measured rows only; use for cross-arm parity)");
    }

    private static Arm load(String cls, String model, String[] names) throws Exception {
        try {
            return (Arm) Class.forName(cls).getConstructor(String.class, String[].class).newInstance(model, names);
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException(cls + " not on classpath: build with Maven (mvn -q package) and run the shaded jar.", e);
        }
    }

    private static String esc(String s) { return s == null ? "" : s.replace("\\", "\\\\").replace("\"", "\\\""); }

    private static void write(Path outDir, String label, double[] us, String arm, int warmup, Map<String, String> o,
                              String serverConfig) throws IOException {
        Path raw = outDir.resolve("raw");
        Files.createDirectories(raw);
        StringBuilder sb = new StringBuilder("latency_us\n");
        for (double v : us) sb.append(String.format(Locale.ROOT, "%.3f%n", v));
        Files.writeString(raw.resolve(label + ".csv"), sb.toString());
        String client = arm.equals("rest")
                ? "java.net.http.HttpClient, HTTP/1.1, one client for all calls (pooled keep-alive), connect timeout 10 s, request timeout 30 s"
                : "JVM in-process";
        String json = String.format(Locale.ROOT,
                "{%n  \"label\": \"%s\",%n  \"arm\": \"%s\",%n  \"n\": %d,%n  \"warmup\": %d,%n"
                        + "  \"warmup_source\": \"validation month (%s)\",%n  \"measured_source\": \"test month (%s)\",%n"
                        + "  \"batch_size\": 1,%n  \"client\": \"%s\",%n  \"java\": \"%s\",%n  \"jvm\": \"%s\",%n  \"jvm_args\": \"%s\",%n"
                        + "  \"os\": \"%s\",%n  \"cpu_count_logical\": %d,%n  \"max_heap_mb\": %d,%n  \"server_config\": %s%n}%n",
                label, arm, us.length, warmup, esc(o.getOrDefault("warmuprows", "warmup_rows.csv")),
                esc(o.getOrDefault("rows", "bench_rows.csv")), client,
                System.getProperty("java.version"), System.getProperty("java.vm.name"),
                esc(String.join(" ", java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments())),
                esc(System.getProperty("os.name") + " " + System.getProperty("os.version")),
                Runtime.getRuntime().availableProcessors(), Runtime.getRuntime().maxMemory() / (1024 * 1024),
                serverConfig == null || serverConfig.isBlank() ? "null" : serverConfig);
        Files.writeString(raw.resolve(label + ".json"), json);
        double[] s = us.clone();
        java.util.Arrays.sort(s);
        System.out.printf(Locale.ROOT, "%s: n=%d median=%.1fus p95=%.1fus p99=%.1fus max=%.1fus%n",
                label, s.length, s[s.length / 2], s[(int) (0.95 * (s.length - 1))], s[(int) (0.99 * (s.length - 1))], s[s.length - 1]);
    }
}
