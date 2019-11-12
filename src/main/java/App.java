import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.mllib.regression.LabeledPoint;

public class App {
        static String path = "/home/arseniy/dev/lpsa.data";
        static double maxErrorValue = 5f;
        static int workerCount = 2;

    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("ParallelGD");
        JavaSparkContext sparkContext = new JavaSparkContext(conf);

        JavaRDD<String> input = sparkContext.textFile(path);
        InputParser parser = new InputParser();

        JavaRDD<LabeledPoint> data = parser.parse(input);

        ParallelGD pgd = new ParallelGD(maxErrorValue, workerCount, sparkContext);
        pgd.Train(data);


    }


}