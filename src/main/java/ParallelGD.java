import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.util.Arrays;

public class ParallelGD {
    private final int iterationsNum = 10;
    private double maxErrorValue;
    private JavaRDD<LabeledPoint> data;
    private LinearRegressionModel model;
    private JavaSparkContext sparkContext;
    private int workerCount;

    public ParallelGD(double MaxErrorValue, int count, JavaSparkContext context){
        maxErrorValue = MaxErrorValue;
        sparkContext = context;
        workerCount = count;
    }
    public LinearRegressionModel Train(JavaRDD<LabeledPoint> inputData){
        data = sparkContext.parallelize(inputData.collect());
        model = LinearRegressionWithSGD.train(data.rdd(),1);
        while (!isModelCompleted()){
            double[] vectors = model.weights().toArray();
            double[] bufferedVectors = vectors.clone();
            Arrays.fill(bufferedVectors, 0);

            data.mapPartitions(part -> {
                LinearRegressionModel tmpModel =  LinearRegressionWithSGD
                        .train((RDD<LabeledPoint>)part,iterationsNUm,0.001,0.001,model.weights());
                double[] tmpVector = tmpModel.weights().toArray();
                for (int i = 0; i < bufferedVectors.length; i++) {
                    bufferedVectors[i] += tmpVector[i];
                }
                return part;
            });

            for (int i = 0; i < vectors.length; i++) {
                vectors[i] = bufferedVectors[i] / workerCount;
            }
            model = new LinearRegressionModel(Vector.dense(vectors),model.intercept);
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
