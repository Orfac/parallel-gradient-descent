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
        static int workerCount = 8;
        static double possibleError = 0.5;

    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("ParallelGD");
        JavaSparkContext sparkContext = new JavaSparkContext(conf);

        JavaRDD<String> input = sparkContext.textFile(path);
        InputParser parser = new InputParser();
        JavaRDD<LabeledPoint> data = parser.parse(input);

        ParallelGD pgd = new ParallelGD(sparkContext);
        LinearRegressionModel model = pgd.Train(data,workerCount);

        ErrorCalculator calculator = new ErrorCalculator();

        double mse = calculator.MeanSquaredError(model,data);
        System.out.println(mse);
        int i = 1000;
        while (mse > possibleError && i > 5){
            model = pgd.Train(data,workerCount,model);
            mse = calculator.MeanSquaredError(model,data);
            System.out.println(mse);
            i--;
        }

        sparkContext.stop();

    }


}