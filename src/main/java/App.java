import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;

import java.util.Arrays;

public class App {
        static String path = "/home/arseniy/dev/learn_projects/math/ParallelGradientDescent/lpsa.data";
        static int workerCount = 2;
        static double possibleError = 0.5;
    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("ParallelGD");
        JavaSparkContext sparkContext = new JavaSparkContext(conf);

        JavaRDD<String> input = sparkContext.textFile(path);
        InputParser parser = new InputParser();
        JavaRDD<LabeledPoint> data = parser.parse(input);

        ParallelGD pgd = new ParallelGD(sparkContext);
        LinearRegressionModel model = pgd.Train(data, workerCount);

        ErrorCalculator calculator = new ErrorCalculator();

        double mse = calculator.MeanSquaredError(model,data);
        int i = 10;
        while (mse > possibleError && i > 0){
            System.out.println(mse);
            System.out.println(model.intercept());
            System.out.println(Arrays.toString(model.weights().toArray()));
            model = pgd.Train(data,workerCount,model);
            mse = calculator.MeanSquaredError(model,data);
            i = i - 1;
        }



    }


}