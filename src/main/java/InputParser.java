import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

class InputParser {
    JavaRDD<LabeledPoint> parse(JavaRDD<String> input) {
        return input
                .map((Function<String, LabeledPoint>) line -> {
                    // Sorry for that lambda
                    String[] parts = line.split(",");
                    String[] pointsStr = parts[1].split(" ");
                    double[] points = new double[pointsStr.length];

                    for (int i = 0; i < pointsStr.length; i++)
                        points[i] = Double.parseDouble(pointsStr[i]);

                    return new LabeledPoint(Double.parseDouble(parts[0]),
                            Vectors.dense(points));
                });
    }
}
