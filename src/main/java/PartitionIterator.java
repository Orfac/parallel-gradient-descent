import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;

public class PartitionIterator implements PairFlatMapFunction<Iterator<LabeledPoint>, Double, Double[]> {
    private double[] weights;
    private double intercept;
    private double learningRate;
    public PartitionIterator(double[] weights, double intercept, double learningRate){
        this.weights = weights;
        this.intercept = intercept;
        this.learningRate = learningRate;
    }

    @Override
    public Iterator<Tuple2<Double, Double[]>> call(Iterator<LabeledPoint> partedData) throws Exception {
        Double[] partedWeights = new Double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            partedWeights[i] = 0.0;
        }
        double partedIntercept = 0.0;

        int count = 0;
        while (partedData.hasNext()) {
            LabeledPoint point = partedData.next();
            count++;
            // Calculating predicted point
            double[] features = point.features().toArray();
            double predictedY = intercept;
            for (int i = 0; i < features.length; i++) {
                predictedY += features[i] * weights[i];
            }

            // Error calculation
            double error = 2 * (predictedY -point.label() );

            // Updating weights and intercept
            partedIntercept -= learningRate * error;
            for (int i = 0; i < partedWeights.length; i++) {
                partedWeights[i] -= learningRate * error * features[i];
            }
        }

        partedIntercept /= count;
        for (int i = 0; i < partedWeights.length; i++) {
            partedWeights[i] /= count;
        }
        ArrayList list = new ArrayList<Tuple2<Double,Double[]>>();
        list.add(new Tuple2<>(partedIntercept,partedWeights));
        return list.iterator();
    }
}
