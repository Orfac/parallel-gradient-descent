# parallel-gradient-descent

## Requirements

- maven
- java 8+
- spark

## Build

```
mvn package
```

## Deploy

```
spark-submit --class App --master local[2] target/ParallelGradientDescent-1.0-SNAPSHOT.jar
```
where 2 is number of workers
