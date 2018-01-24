package com.github.chen.wentao.languageclassifier;

import com.github.chen.wentao.languageclassifier.languages.Language;
import com.github.chen.wentao.mllib.training.BatchFullDataSetStream;
import com.github.chen.wentao.mllib.training.DataSet;
import com.github.chen.wentao.mllib.training.FullDataSet;
import com.github.chen.wentao.mllib.training.NeuralNetwork;
import com.github.chen.wentao.mllib.training.StaticFullDataSetStream;
import com.github.chen.wentao.mllib.training.StreamCostFunction;
import com.github.chen.wentao.mllib.training.StreamSupervisedLearningAlgorithm;
import org.ejml.simple.SimpleMatrix;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;
import java.util.function.IntConsumer;
import java.util.function.Supplier;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class LanguageClassifier implements Serializable {

    private static final long serialVersionUID = -7603513701132703287L;

    private static final Logger LOGGER = Logger.getLogger(LanguageClassifier.class.getName());

    private final int inputLettersCount;
    private final int maxWordLength;
    private final LetterEncoder letterEncoder;
    private final Language[] languages;
    private transient NeuralNetwork network;
    private final LanguageWordsDataSet allWordsDataSet;

    public LanguageClassifier(int inputLettersCount, int maxWordLength, LetterEncoder letterEncoder, Random random, Language... languages) {
        this(inputLettersCount, maxWordLength, letterEncoder, random, new LanguageWordsDataSet(random, languages), languages);
    }

    public LanguageClassifier(int inputLettersCount, int maxWordLength, LetterEncoder letterEncoder, Random random, LanguageWordsDataSet allWordsDataSet, Language... languages) {
        if (inputLettersCount <= 0) throw new IllegalArgumentException("There must be at least 1 letter of input. Given: (" + inputLettersCount + ")");
        if (letterEncoder == null) throw new IllegalArgumentException("letter encoder cannot be null");
        if (languages.length <= 0) throw new IllegalArgumentException("There must be at least 1 language.");
        this.inputLettersCount = inputLettersCount;
        this.maxWordLength = maxWordLength;
        this.letterEncoder = letterEncoder;
        this.languages = languages;
        this.network = NeuralNetwork.emptyNetwork((inputLettersCount + 1) * maxWordLength, inputLettersCount + 1, languages.length);
        this.network.randomlyInitialize(random);
        this.allWordsDataSet = allWordsDataSet;
    }

    public static Supplier<LanguageClassifier> getLanguageRecognizerGenerator(int inputLettersCount, int maxWordLength, LetterEncoder letterEncoder, Random random, Language... languages) {
        LanguageWordsDataSet dataSet = new LanguageWordsDataSet(random, languages);
        return () -> new LanguageClassifier(inputLettersCount, maxWordLength, letterEncoder, random, dataSet, languages);
    }

    public Map<Language, Double> process(String input) {
        double[] data = new double[this.network.numInputs()];
        allWordsDataSet.setDataToArray(inputLettersCount, maxWordLength, letterEncoder, input, data);
        DataSet dataSet = DataSet.single(data);
        SimpleMatrix[] result = this.network.feedForward(dataSet);
        SimpleMatrix output = result[result.length - 1];
        Map<Language, Double> results = new HashMap<>();
        for (int i = 0; i < languages.length; i++) {
            results.put(languages[i], output.get(i, 0));
        }
        return results;
    }

    public Map<Language, Double> processParagraph(String paragraph) {
        return processParagraph(paragraph.split(" "));
    }

    public Map<Language, Double> processParagraph(String[] words) {
        Map<Language, Double> score = Arrays.stream(languages).collect(Collectors.toMap(Function.identity(), l -> 1.0));
        for (int i = 0; i < words.length; i++) {
            String word = words[i].toLowerCase();
            StringBuilder input = new StringBuilder();
            for (int c = 0; c < word.length(); c++) {
                char character = word.charAt(c);
                if (letterEncoder.applyAsInt(character) >= 0) {
                    input.append(character);
                }
            }
            Map<Language, Double> result = process(input.toString());
            for (Map.Entry<Language, Double> entry : result.entrySet()) {
                Language language = entry.getKey();
                score.put(language, Math.pow(score.get(language), i / (i + 1.0)) * Math.pow(entry.getValue(), 1.0 / (i + 1)));
            }
        }
        return score;
    }

    public Language predict(String input) {
        return languages[predictIndex(input)];
    }

    public int predictIndex(String input) {
        double[] data = new double[this.network.numInputs()];
        allWordsDataSet.setDataToArray(inputLettersCount, maxWordLength, letterEncoder, input, data);
        DataSet dataSet = DataSet.single(data);
        return (int) this.network.predict(dataSet).get(0);
    }

    public List<Double> train(double alpha, double lambda, int numIterations, int batchSize) {
        List<Double> runningCost = new ArrayList<>();
        BatchFullDataSetStream batchGenerator = allWordsDataSet.buildDataSetGenerator(inputLettersCount, maxWordLength, letterEncoder, batchSize);
        BatchFullDataSetStream batchGeneratorWithPrint = new StaticFullDataSetStream(batchGenerator.numBatches()) {
            @Override
            public FullDataSet getBatch(int batchIndex) {
                if (batchIndex % (numIterations / 10) == 0) {
                    LOGGER.info(() -> String.format("Training... (%f%%)%n", batchIndex * 100.0 / numIterations));
                    //double cost = network.costFunction(batchGenerator, lambda);
                    double cost = accuracy(10000);
                    LOGGER.info(() -> String.format("\tCurrent cost: %f)%n", cost));
                    runningCost.add(cost);
                }
                return batchGenerator.getBatch(batchIndex);
            }
        };
        this.network.trainMiniBatch(batchGeneratorWithPrint, alpha, lambda, numIterations);
        return runningCost;
    }

    public void train(double alpha, double lambda, int numIterations, int batchSize, int preBatchEventCount, IntConsumer preBatchEvent) {
        train(alpha, lambda, numIterations, batchSize, preBatchEventCount, preBatchEvent, allWordsDataSet);
    }

    public void train(double alpha, double lambda, int numIterations, int batchSize, int preBatchEventCount, IntConsumer preBatchEvent, LanguageWordsDataSet dataSet) {
        boolean runPreBatchEvent = preBatchEventCount > 0;
        int numBatches = Math.max(numIterations / preBatchEventCount, 1);
        BatchFullDataSetStream batchGenerator = dataSet.buildDataSetGenerator(inputLettersCount, maxWordLength, letterEncoder, batchSize);
        BatchFullDataSetStream batchGeneratorWithPrint = new StaticFullDataSetStream(batchGenerator.numBatches()) {
            @Override
            public FullDataSet getBatch(int batchIndex) {
                if (runPreBatchEvent && batchIndex % numBatches == 0) {
                    preBatchEvent.accept(batchIndex);
                }
                return batchGenerator.getBatch(batchIndex);
            }
        };
        this.network.trainMiniBatch(batchGeneratorWithPrint, alpha, lambda, numIterations);
    }

    public void saveToFileBinary(String directoryName) throws IOException {
        File directory = new File(directoryName);
        if (!directory.exists()) {
            if (!directory.mkdir()) {
                throw new IOException(directoryName);
            }
        }
        this.network.saveToFileBinary(directoryName);
        try (FileOutputStream fileOutputStream = new FileOutputStream(directoryName + "/lanrec.jobj");
             ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream)) {
            objectOutputStream.writeObject(this);
        }
    }

    public static LanguageClassifier loadFromFileBinary(String directoryName) throws IOException {
        LanguageClassifier languageClassifier;
        try (FileInputStream fileInputStream = new FileInputStream(directoryName + "/lanrec.jobj");
             ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream)) {
            languageClassifier = (LanguageClassifier) objectInputStream.readObject();
            languageClassifier.network = NeuralNetwork.loadFromFileBinary(directoryName);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
        return languageClassifier;
    }

    public double accuracy(int testWordsCount) {
        return accuracy(testWordsCount, allWordsDataSet);
    }

    public double accuracy(int testWordsCount, LanguageWordsDataSet dataSet) {
        return dataSet.accuracy(this::predictIndex, testWordsCount);
    }

    public double cost(double lambda) {
        return cost(lambda, allWordsDataSet);
    }

    public double cost(double lambda, LanguageWordsDataSet dataSet) {
        int batchSize = 1000;
        BatchFullDataSetStream dataSetStream = dataSet.buildDataSetGenerator(inputLettersCount, maxWordLength, letterEncoder, batchSize);
        return network.costFunction(dataSetStream, lambda);
    }

    public LanguageWordsDataSet getAllWordsDataSet() {
        return allWordsDataSet;
    }

    public static StreamSupervisedLearningAlgorithm<LanguageClassifier> getStreamAlgorithm(Supplier<LanguageClassifier> generator, double alpha, double lambda, int numIterations) {
        return dataSet -> {
            LanguageClassifier languageClassifier = generator.get();
            languageClassifier.network.trainMiniBatch(dataSet, alpha, lambda, numIterations);
            return languageClassifier;
        };
    }

    public static StreamCostFunction<LanguageClassifier> getCostFunction(double lambda) {
        return (languageClassifier, dataSetStream) -> languageClassifier.network.costFunction(dataSetStream, lambda);
    }

    public Language[] getLanguages() {
        return languages;
    }

    public int getMaxWordLength() {
        return maxWordLength;
    }

    public int getInputLettersCount() {
        return inputLettersCount;
    }

    public LetterEncoder getLetterEncoder() {
        return letterEncoder;
    }
}
