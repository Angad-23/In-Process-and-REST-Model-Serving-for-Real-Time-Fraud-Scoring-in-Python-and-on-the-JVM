package bench;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * The real "bridge": a Java client calling the Python REST service.
 * Client settings: java.net.http.HttpClient, HTTP/1.1, one client instance for all calls (connection pool,
 * keep-alive), connect timeout 10 s, request timeout 30 s, JSON body {"features":[...]}.
 *
 * Per call it records (microseconds, System.nanoTime) the same components as the Python client:
 * client_serialize (build JSON + request), round_trip (send until the full body is received),
 * client_parse (extract "prob"), and the server's X-Handler-us / X-Parse-us / X-Model-us headers.
 */
public final class RestArm implements Arm {
    private final HttpClient client = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .connectTimeout(Duration.ofSeconds(10)).build();
    private final URI uri;

    /** Components of the most recent call, in microseconds. */
    double lastSerializeUs, lastRoundTripUs, lastParseUs, lastServerHandlerUs, lastServerParseUs, lastModelUs;

    public RestArm(String url) { this.uri = URI.create(url); }

    /** Server settings as reported by GET /health (raw JSON), recorded in the sidecar. */
    String health() {
        try {
            URI h = uri.resolve("/health");
            HttpResponse<String> r = client.send(HttpRequest.newBuilder(h).timeout(Duration.ofSeconds(10)).GET().build(),
                    HttpResponse.BodyHandlers.ofString());
            return r.body();
        } catch (Exception e) {
            return "null";
        }
    }

    @Override public double score(double[] f) throws Exception {
        long t0 = System.nanoTime();
        StringBuilder sb = new StringBuilder(f.length * 12 + 20).append("{\"features\":[");
        for (int i = 0; i < f.length; i++) { if (i > 0) sb.append(','); sb.append((float) f[i]); }
        sb.append("]}");
        HttpRequest req = HttpRequest.newBuilder(uri).timeout(Duration.ofSeconds(30))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(sb.toString())).build();
        long t1 = System.nanoTime();
        HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
        long t2 = System.nanoTime();
        if (resp.statusCode() != 200) throw new IllegalStateException("HTTP " + resp.statusCode());
        String b = resp.body();
        int k = b.indexOf("\"prob\":");
        int s = k + 7;
        while (s < b.length() && b.charAt(s) == ' ') s++;
        int e = s;
        while (e < b.length() && "0123456789.eE+-".indexOf(b.charAt(e)) >= 0) e++;
        double p = Double.parseDouble(b.substring(s, e));
        long t3 = System.nanoTime();
        lastSerializeUs = (t1 - t0) / 1000.0;
        lastRoundTripUs = (t2 - t1) / 1000.0;
        lastParseUs = (t3 - t2) / 1000.0;
        lastServerHandlerUs = header(resp, "X-Handler-us");
        lastServerParseUs = header(resp, "X-Parse-us");
        lastModelUs = header(resp, "X-Model-us");
        return p;
    }

    private static double header(HttpResponse<String> r, String name) {
        return r.headers().firstValue(name).map(Double::parseDouble).orElse(Double.NaN);
    }
}
