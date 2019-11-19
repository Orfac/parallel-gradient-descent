import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import scala.Tuple2;

public class ErrorCalculator {
    public double MeanSquaredError(LinearRegressionModel model, JavaRDD<LabeledPoint> points){
        JavaRDD<Tuple2<Double, Double>> valuesAndPred = points
                .map(point -> new Tuple2<>(
                        point.label(),
                        model.predict(point.features())));

        return valuesAndPred.mapToDouble(
                tuple -> Math.abs(tuple._1 - tuple._2)).max();
    }
}
