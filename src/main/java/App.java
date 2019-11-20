import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import scala.Tuple2;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

public class App {
        static String path = "/home/arseniy/dev/learn_projects/math/ParallelGradientDescent/lpsa.data";
        static double possibleDifference = 0.0001;

    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("ParallelGD");
        JavaSparkContext sparkContext = new JavaSparkContext(conf);

        JavaRDD<String> input = sparkContext.textFile(path);

        InputParser parser = new InputParser();
        JavaRDD<LabeledPoint> data = parser.parse(input);

        ParallelGD pgd = new ParallelGD(sparkContext);
        LinearRegressionModel model = pgd.Train(data);

        ErrorCalculator calculator = new ErrorCalculator();

        double mse = calculator.MeanSquaredError(model,data);

        // Count amount of operations for learning
        int counter = 1;
        double oldMse;
        do {
            oldMse = mse;

            model = pgd.Train(data,model);
            mse = calculator.MeanSquaredError(model,data);

            System.out.println(counter + " " + mse);
            counter++;
        } while (Math.abs(mse - oldMse) > possibleDifference);


        System.out.println(counter + " " +  Arrays.toString(model.weights().toArray()));
        sparkContext.stop();

    }


}