# parallel-gradient-descent

# BUILD
use command
```
mvn package
```

# DEPLOY
use command

```
spark-submit --class App --master local[2] target/ParallelGradientDescent-1.0-SNAPSHOT.jar
```
where 2 is number of workers