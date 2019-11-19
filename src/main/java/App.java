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
        static double possibleDifference = 1e-15;

    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("ParallelGD");
        JavaSparkContext sparkContext = new JavaSparkContext(conf);

        JavaRDD<String>[] input = sparkContext.textFile(path).randomSplit(new double[]{0.7, 0.3});

        InputParser parser = new InputParser();
        JavaRDD<LabeledPoint> trainData = parser.parse(input[0]);
        JavaRDD<LabeledPoint> testData = parser.parse(input[1]);

        ParallelGD pgd = new ParallelGD(sparkContext);
        LinearRegressionModel model = pgd.Train(trainData);

        ErrorCalculator calculator = new ErrorCalculator();

        double mse = calculator.MeanSquaredError(model,testData);

        // Count amount of operations for learning
        int counter = 1;
        double oldMse;
        do {
            oldMse = mse;

            model = pgd.Train(trainData,model);
            mse = calculator.MeanSquaredError(model,testData);

            System.out.println(counter + " " + mse);
            counter++;
        } while (Math.abs(mse - oldMse) > possibleDifference);


        System.out.println(counter);
        sparkContext.stop();

    }


}