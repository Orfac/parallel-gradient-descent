import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import scala.Tuple2;

import java.util.Arrays;

public class ParallelGD {
    private double maxErrorValue;
    private JavaRDD<LabeledPoint> data;
    private LinearRegressionModel model;
    private JavaSparkContext sparkContext;

    public ParallelGD(double MaxErrorValue, JavaSparkContext context){
        maxErrorValue = MaxErrorValue;
        sparkContext = context;
    }
    public LinearRegressionModel Train(JavaRDD<LabeledPoint> inputData){
        data = sparkContext.parallelize(inputData.collect());
        model = LinearRegressionWithSGD.train(data.rdd(),1);
        while (!isModelCompleted()){
            data.mapPartitions(a -> {
                return a;
            });
        }
        return model;
    }

    private boolean isModelCompleted(){
        JavaRDD<Tuple2<Double, Double>> valuesAndPred = data
                .map(point -> new Tuple2<>(
                        point.label(),
                        model.predict(point.features())));

        double MSE = valuesAndPred.mapToDouble(
                tuple -> Math.pow(tuple._1 - tuple._2, 2)).mean();
        return MSE > maxErrorValue;
    }

}
