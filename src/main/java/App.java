import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import scala.Tuple2;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

public class App {
    public static void main(String[] args) {
        String path = "/home/arseniy/dev/lpsa.data";
        SparkConf conf = new SparkConf().setAppName("ParallelGD");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> input = sc.textFile(path);

        InputParser parser = new InputParser();
        JavaRDD<LabeledPoint> data = parser.parse(input);


        int numIterations = 10000000;
        System.out.println("Training iterations - " + numIterations);
        LinearRegressionModel model = LinearRegressionWithSGD.train(
                data.rdd(), numIterations);

        LinearRegressionModel model2 = model;


        JavaRDD<Tuple2<Double, Double>> valuesAndPred = data
                .map(point -> new Tuple2<>(
                        point.label(),
                        model.predict(point.features())));

        double MSE = valuesAndPred.mapToDouble(
                tuple -> Math.pow(tuple._1 - tuple._2, 2)).mean();
        System.out.println(Arrays.toString(valuesAndPred.collect().toArray()));
        System.out.println("training Mean Squared Error = "
                + String.valueOf(MSE));

    }


}