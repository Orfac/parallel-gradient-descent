import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LinearRegressionModel;

import java.util.Random;

public class RandomModelGenerator {
    public LinearRegressionModel generate(int weightCount){
        Random r = new Random(5);
        double[] randomWeights = new double[weightCount];
        for (int i = 0; i < weightCount; i++) {
            randomWeights[i] = r.nextDouble();
        }
        return new LinearRegressionModel(Vectors.dense(randomWeights), r.nextDouble());
    }
}
