package com.github.chen.wentao.languageclassifier.cli;

import com.github.chen.wentao.languageclassifier.LanguageClassifier;
import com.github.chen.wentao.languageclassifier.LanguageWordsDataSet;
import com.github.chen.wentao.languageclassifier.LetterEncoder;
import com.github.chen.wentao.languageclassifier.LetterEncodingFileConverter;
import com.github.chen.wentao.languageclassifier.languages.BasicLanguage;
import com.github.chen.wentao.languageclassifier.languages.Language;
import com.github.chen.wentao.mllib.data.LearningCurve;
import com.github.chen.wentao.mllib.training.BatchFullDataSetStream;
import com.github.chen.wentao.mllib.training.StreamCostFunction;
import com.github.chen.wentao.mllib.training.StreamSupervisedLearningAlgorithm;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.util.StringJoiner;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.IntFunction;
import java.util.function.Supplier;
import java.util.stream.IntStream;

public class Cli {

    private static final String QUIT_COMMAND = "quit";
    private final CliCommand[] COMMANDS = new CliCommand[]{
            new CliCommand("new",
                    "Create a new classifier",
                    "new [<classifier>] {-m <length> | -c <minFraction>} [-d <dataSet>] -l <languages>...",
                    this::newRecognizer, new Options()
                    .addOptionGroup(requiredOptionGroup(
                            option("m", "length", "max word length", 1),
                            option("c", "cover", "the minimum fraction of all words included used to determine max word length", 1)
                    )).addOption(requiredOption("l", "language", "The languages to classify", Option.UNLIMITED_VALUES))
                    .addOption(option("d", "dataset", "Default data set for classifier", 1))
            ),
            new CliCommand("show",
                    "Display information for current classifiers, languages, or data sets",
                    "show {-c | -l | -d}",
                    this::show, new Options()
                    .addOptionGroup(requiredOptionGroup(
                            option("c", "classifiers", "Show classifiers", 0),
                            option("l", "languages", "Show languages", 0),
                            option("d", "datasets", "Show data sets", 0)
                    ))
            ),
            new CliCommand("train",
                    "Train a classifier",
                    "train [<classifier>] -a <alpha> -i <iterations> [-l <lambda>] [-b <batchSize>] [-v <numPrintStatements>] [-c] [-x] [-d <dataSet>]",
                    this::train, new Options()
                    .addOption(requiredOption("a", "alpha", "alpha, learning rate (> 0)", 1))
                    .addOption(option("l", "lambda", "lambda, regularization parameter (>= 0)", 1))
                    .addOption(requiredOption("i", "iterations", "number of iterations (> 0)", 1))
                    .addOption(option("b", "batch", "batch size (> 0)", 1))
                    .addOption(option("v", "verbose", "Display additional information", 1))
                    .addOption(option("c", "cost", "Display running cost", 0))
                    .addOption(option("x", "accuracy", "Display running accuracy", 0))
                    .addOption(option("d", "dataset", "Training data set", 1))
            ),
            new CliCommand("test",
                    "Test a classifier with an input",
                    "test [<classifier>] [-v] {-w <input> | -p <paragraph>}",
                    this::test, new Options()
                    .addOptionGroup(requiredOptionGroup(
                            option("w", "word", "Test an input word", 1),
                            option("p", "paragraph", "Test a paragraph of words", Option.UNLIMITED_VALUES)
                    ))
                    .addOption(option("v", "verbose", "Display additional information", 0))
            ),
            new CliCommand("analyze",
                    "Display analysis information for a classifier",
                    "analyze [<classifier>] [-l <lambda>] [-d <dataSet>] [-a] [-c]",
                    this::analyze, new Options()
                    .addOption(option("l", "lambda", "lambda, regularization parameter (>= 0)", 1))
                    .addOption(option("d", "dataset", "Training data set", 1))
                    .addOption(option("a", "accuracy", "Display accuracy", 0))
                    .addOption(option("c", "cost", "Display cost", 0))
            ),
            new CliCommand("save",
                    "Saves a classifier as a directory",
                    "save -f <file>",
                    this::save, new Options()
                    .addOption(requiredOption("f", "file", "Output directory to save", 1))
            ),
            new CliCommand("load",
                    "Loads a classifier as a directory",
                    "load -f <file>"
                    , this::load, new Options()
                    .addOption(requiredOption("f", "file", "Output directory to load", 1))
            ),
            new CliCommand("add",
                    "Add a new language or data set",
                    "add {-l <iso639-1> <name> <file> | -d <name> <languages>...}",
                    this::add, new Options()
                    .addOptionGroup(requiredOptionGroup(
                            option("l", "language", "Add new language, args: <iso639-1> <name> <file>", 3),
                            option("d", "dataset", "Add new data set, args: <name> <languages>...", Option.UNLIMITED_VALUES)
                    ))
            ),
            new CliCommand("copy",
                    "Copies a classifier, language, or data set",
                    "copy {-c | -l | -d}",
                    this::copy, new Options()
                    .addOptionGroup(requiredOptionGroup(
                            option("c", "classifier", "Copy a classifier", 0),
                            option("l", "language", "Copy a language", 0),
                            option("d", "dataset", "Copy a dataset", 0)
                    ))
            ),
            new CliCommand("delete",
                    "Removes a classifier, language, or data set",
                    "delete {-c | -l | -d}",
                    this::remove, new Options()
                    .addOptionGroup(requiredOptionGroup(
                            option("c", "classifier", "Copy a classifier", 0),
                            option("l", "language", "Copy a language", 0),
                            option("d", "dataset", "Copy a dataset", 0)
                    ))
            ),
            new CliCommand("split",
                    "Splits a data set into multiple data sets",
                    "split <src> {<dest>... -d <fractions>... | -t <trainFraction> <cvFraction> <testFraction>}", this::splitData, new Options()
                    .addOptionGroup(requiredOptionGroup(
                            option("d", "division", "Split fractions", Option.UNLIMITED_VALUES),
                            option("t", "tcvt", "Split set into train, cross-validation and test sets", 3)
                    ))
            ),
            new CliCommand("shuffle",
                    "Shuffles a data set",
                    "shuffle <dataSet>",
                    this::shuffle, new Options()
            ),
            new CliCommand("data",
                    "Add values to a data set",
                    "data <dataSet> [-a <language> <words>...]", this::data, new Options()
                    .addOption(option("a", "add", "Add data to a data set, args: <language> <words>...", Option.UNLIMITED_VALUES))
            ),
            new CliCommand("lcurve",
                    "Computes learning curves for a classifier",
                    "lcurve [<classifier>] -a <alpha> -i <iterations> [-l <lambda>] [-b <batchSize>] [-d <trainDataSet>] [-c <cvDataSet>] [-g] [{-r <start> <end> <incr> | -s <sizes>...}]",
                    this::learningCurve, new Options()
                    .addOption(requiredOption("a", "alpha", "alpha, learning rate (> 0)", 1))
                    .addOption(option("l", "lambda", "lambda, regularization parameter (>= 0)", 1))
                    .addOption(requiredOption("i", "iterations", "number of iterations (> 0)", 1))
                    .addOption(option("b", "batch", "batch size (> 0)", 1))
                    .addOption(option("d", "train", "Training data set", 1))
                    .addOption(option("c", "cv", "Cross validation data set", 1))
                    .addOptionGroup(optionGroup(
                            option("s", "sizes", "Test training data set sizes", Option.UNLIMITED_VALUES),
                            option("r", "range", "Test training data set sizes with range, args: <start> <end> <incr>", 3)
                    ))
                    .addOption(option("g", "gui", "Display data with GUI", 0))
            ),
            new CliCommand("file",
                    "Perform operations on data files",
                    "file -f <file> [-o <output>] [-c]",
                    this::file, new Options()
                    .addOption(option("f", "file", "The input file to use", 1))
                    .addOption(option("o", "output", "The output file", 1))
                    .addOption(option("c", "convert", "Convert the encoding of the file", 0))
            ),
            new CliCommand("help",
                    "Displays help information",
                    "help [<command>]",
                    this::help, new Options()
            )
    };

    private final String prompt;
    private final Random random;
    private final Scanner scanner;
    private final PrintStream out;
    private final PrintStream err;
    private CommandLineParser parser;
    private final LanguageClassifier[] languageClassifiers;
    private final Map<String, Supplier<Language>> languagesLoader;
    private final Map<String, Language> languages = new HashMap<>();
    private final Map<String, LanguageWordsDataSet> dataSets = new HashMap<>();

    private Cli(String prompt, Random random, int languageRecognizersCount) {
        this.prompt = prompt;
        this.random = random;
        this.scanner = new Scanner(System.in);
        this.out = System.out;
        this.err = System.err;
        this.parser = new DefaultParser();
        this.languageClassifiers = new LanguageClassifier[Math.max(languageRecognizersCount, 1)];
        this.languagesLoader = loadLanguages();
    }

    public static void run(String[] args) {
        Options options = new Options().addOption("s", "seed", true, "random seed");
        CommandLine cmd;
        try {
            cmd = new DefaultParser().parse(options, args);
        } catch (ParseException e) {
            throw new RuntimeException(e);
        }
        Random random = cmd.hasOption("s") ? new Random(Long.parseLong(cmd.getOptionValue("s"))) : new Random();
        Cli cli = new Cli(">", random, args.length >= 1 ? Integer.parseInt(args[0]) : 10);
        cli.run();
    }

    private void newRecognizer(CommandLine cmd) {
        Language[] languages = parseLanguages(cmd.getOptionValues("l"));
        LanguageWordsDataSet dataSet = dataSets.get(cmd.getOptionValue("d"));
        boolean createNewDataSet = dataSet == null;
        if (createNewDataSet) {
            dataSet = new LanguageWordsDataSet(random, languages);
        }

        int dest = getDestSlot(cmd, 0, false);
        int maxWordLength = Integer.parseInt(cmd.getOptionValue("m", "-1"));
        double dataSetCoverage = Double.parseDouble(cmd.getOptionValue("c", "-1"));
        if (dataSetCoverage >= 0) {
            maxWordLength = Math.max(maxWordLength, dataSet.findMaxWordLengthForDataSetCoverage(dataSetCoverage));
        }

        int inputLettersCount = Language.countDistinctLetters(languages);
        LetterEncoder letterEncoder = LetterEncoder.fromLanguages(languages);
        this.languageClassifiers[dest] = new LanguageClassifier(inputLettersCount, maxWordLength, letterEncoder, random, dataSet, languages);
        if (createNewDataSet) {
            addDataSet(languages, dataSet);
        }
    }

    private void addDataSet(Language[] languages, LanguageWordsDataSet dataSet) {
        String dataSetName = getDataSetName(languages);
        if (this.dataSets.containsKey(dataSetName)) {
            for (int i = 1;; i++) {
                if (!this.dataSets.containsKey(dataSetName + i)) {
                    this.dataSets.put(dataSetName + i, dataSet);
                    break;
                }
            }
        } else {
            this.dataSets.put(dataSetName, dataSet);
        }
    }

    private String getDataSetName(Language[] languages) {
        StringJoiner str = new StringJoiner("-");
        Language[] languagesSorted = Arrays.copyOf(languages, languages.length);
        Arrays.sort(languagesSorted, Comparator.comparing(Language::getIso6391));
        for (Language language : languagesSorted) {
            str.add(language.getIso6391());
        }
        return str.toString();
    }

    private void show(CommandLine cmd) {
        if (cmd.hasOption("l")) {
            showLanguages();
        } else if (cmd.hasOption("d")) {
            showDataSets();
        } else if (cmd.hasOption("c")) {
            int start = 0;
            int end = languageClassifiers.length;
            String[] args = cmd.getArgs();
            if (args.length >= 1) {
                start = Integer.parseInt(args[0]);
            }
            if (args.length >= 2) {
                end = Math.min(Integer.parseInt(args[1]) + 1, languageClassifiers.length);
            }
            showLanguageRecognizers(start, end);
        }
    }

    private void showLanguages() {
        for (Map.Entry<String, Language> entry : languages.entrySet()) {
            Language language = entry.getValue();
            Set<Character> letters = language.getLetters();
            out.printf("%s) %s, Letters: (%d)%s, Words: %d%n", entry.getKey(), language.getName(), letters.size(), letters, language.getWords().size());
        }
        Set<String> unloadedLanguages = new HashSet<>(languagesLoader.keySet());
        unloadedLanguages.removeAll(languages.keySet());
        for (String language : unloadedLanguages) {
            out.printf("%s) Not yet loaded%n", language);
        }
    }

    private void showDataSets() {
        if (dataSets.isEmpty()) {
            out.println("No data sets");
        }
        for (Map.Entry<String, LanguageWordsDataSet> entry : dataSets.entrySet()) {
            LanguageWordsDataSet dataSet = entry.getValue();
            out.printf("%s) Words: %d, Languages: %s%n", entry.getKey(), dataSet.numWords(), Arrays.toString(dataSet.getLanguages()));
        }
    }

    private void showLanguageRecognizers(int start, int end) {
        for (int i = start; i < end; i++) {
            LanguageClassifier languageClassifier = languageClassifiers[i];
            if (languageClassifier != null) {
                out.printf("%2d) Max Word Length: %d, Languages: %s%n",
                        i, languageClassifier.getMaxWordLength(), Arrays.toString(languageClassifier.getLanguages()));
            } else {
                out.printf("%2d) Empty%n", i);
            }
        }
    }

    private void train(CommandLine cmd) {
        double alpha = Double.parseDouble(cmd.getOptionValue("a"));
        double lambda = Double.parseDouble(cmd.getOptionValue("l", "0"));
        int iterations = Integer.parseInt(cmd.getOptionValue("i"));
        int batchSize = Integer.parseInt(cmd.getOptionValue("b", "1"));
        Integer displayInfo = cmd.hasOption("v") ? Integer.parseInt(cmd.getOptionValue("v", "0")) : null;
        boolean displayRunningCost = cmd.hasOption("c");
        boolean displayRunningAccuracy = cmd.hasOption("x");
        String trainingDataSet = cmd.getOptionValue("d");
        if (alpha <= 0) throw new IllegalArgumentException(String.format("learning rate (%f) must be greater than 0", alpha));
        if (lambda < 0) throw new IllegalArgumentException(String.format("regularization parameter (%f) cannot be less than 0", lambda));
        if (iterations <= 0) throw new IllegalArgumentException(String.format("number of iterations (%d) must be greater than 0", iterations));
        if (batchSize <= 0) throw new IllegalArgumentException(String.format("batch size (%d) must be greater than 0", batchSize));

        int dest = getDestSlot(cmd, 0, true);
        LanguageClassifier languageClassifier = languageClassifiers[dest];
        int numWords = languageClassifier.getAllWordsDataSet().numWords();
        AtomicLong startTrainingTime = new AtomicLong(System.nanoTime());
        languageClassifier.train(alpha, lambda, iterations, batchSize, displayInfo != null ? Math.max(displayInfo, 0) : 0, batchIndex -> {
                long startTime = System.nanoTime();
                if (displayRunningCost) {
                    out.printf("Current cost: %.10f%n", languageClassifier.cost(lambda));
                }
                if (displayRunningAccuracy) {
                    out.printf("Current accuracy: %.10f%n", languageClassifier.accuracy(numWords));
                }
                startTrainingTime.set(startTrainingTime.get() - startTime + System.nanoTime());
                out.printf("Training... %.2f%%%n", batchIndex * 100.0 / iterations);
        }, getDataSet(trainingDataSet, languageClassifier::getAllWordsDataSet));
        if (displayInfo != null) {
            out.println("Training... 100.00%");
            out.printf("Training time: %.3fs%n", (System.nanoTime() - startTrainingTime.get()) / 1000000000.0);
        }
    }

    private LanguageWordsDataSet getDataSet(String dataSetName, Supplier<LanguageWordsDataSet> defaultDataSet) {
        if (dataSetName == null) {
            return defaultDataSet.get();
        }
        LanguageWordsDataSet dataSet = dataSets.get(dataSetName);
        if (dataSet == null) {
            err.printf("Unknown data set: %s%n", dataSetName);
        }
        return dataSet;
    }

    private void test(CommandLine cmd) {
        boolean verbose = cmd.hasOption("v");
        int dest = getDestSlot(cmd, 0, true);
        LanguageClassifier languageClassifier = languageClassifiers[dest];

        String word = cmd.getOptionValue("w");
        if (word == null) {
            String[] paragraph = Arrays.stream(cmd.getOptionValues("p")).map(s -> s.split("_")).flatMap(Arrays::stream).toArray(String[]::new);

            Map<Language, Double> results = languageClassifier.processParagraph(paragraph);
            results.entrySet().stream()
                    .max(Comparator.comparingDouble(Map.Entry::getValue))
                    .ifPresent(maxScoreLanguage ->
                            out.printf("Prediction: %s%n", maxScoreLanguage.getKey())
                    );
            if (verbose) {
                List<Language> rankedResults = new ArrayList<>(results.keySet());
                rankedResults.sort(Comparator.comparingDouble(results::get).reversed());
                for (Language language : rankedResults) {
                    out.printf("%s) %.10f%n", language.getName(), results.get(language));
                }
            }
        } else {
            out.printf("Prediction: %s%n", languageClassifier.predict(word));
            if (verbose) {
                Map<Language, Double> results = languageClassifier.process(word);
                List<Language> rankedResults = new ArrayList<>(results.keySet());
                rankedResults.sort(Comparator.comparingDouble(results::get).reversed());
                for (Language language : rankedResults) {
                    out.printf("%s) %.10f%n", language.getName(), results.get(language));
                }
            }
        }

    }

    private void analyze(CommandLine cmd) {
        boolean displayAccuracy = cmd.hasOption("a");
        boolean displayCost = cmd.hasOption("c");
        double lambda = Double.parseDouble(cmd.getOptionValue("l", "0"));
        String dataSetName = cmd.getOptionValue("d");
        if (lambda < 0) throw new IllegalArgumentException(String.format("regularization parameter (%f) cannot be less than 0", lambda));

        int dest = getDestSlot(cmd, 0, true);
        LanguageClassifier languageClassifier = languageClassifiers[dest];
        LanguageWordsDataSet dataSet = getDataSet(dataSetName, languageClassifier::getAllWordsDataSet);
        if (displayAccuracy) {
            out.printf("Accuracy: %.10f%n", languageClassifier.accuracy(languageClassifier.getAllWordsDataSet().numWords(), dataSet) * 100.0);
        }
        if (displayCost) {
            out.printf("Cost: %.10f%n", languageClassifier.cost(lambda, dataSet));
        }
    }

    private void save(CommandLine cmd) {
        String file = cmd.getOptionValue("f");

        int dest = getDestSlot(cmd, 0, true);
        LanguageClassifier languageClassifier = languageClassifiers[dest];
        try {
            languageClassifier.saveToFileBinary(file);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        out.println("Saved");
    }

    private void load(CommandLine cmd) {
        String file = cmd.getOptionValue("f");

        int dest = getDestSlot(cmd, 0, false);
        try {
            LanguageClassifier languageClassifier = LanguageClassifier.loadFromFileBinary(file);
            languageClassifiers[dest] = languageClassifier;
            addDataSet(languageClassifier.getLanguages(), languageClassifier.getAllWordsDataSet());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        out.println("Loaded");
    }

    private void add(CommandLine cmd) {
        String[] addLanguageValues = cmd.getOptionValues("l");
        String[] addDataSetValues = cmd.getOptionValues("d");

        if (addLanguageValues != null) {
            String iso6391 = addLanguageValues[0];
            String name = addLanguageValues[1];
            String file = addLanguageValues[2];
            if (iso6391.isEmpty()) {
                err.printf("ISO 639-1 code cannot be empty%n");
            } else if (name.isEmpty()) {
                err.printf("Language name cannot be empty%n");
            }
            languages.put(iso6391, new BasicLanguage(iso6391, name, file));
            out.println("Added language " + name);
        } else if (addDataSetValues != null) {
            String name = addDataSetValues[0];
            Language[] languages = parseLanguages(Arrays.copyOfRange(addDataSetValues, 1, addDataSetValues.length));
            dataSets.put(name, new LanguageWordsDataSet(Collections.emptyList(), languages));
        }
    }

    private void copy(CommandLine cmd) {
        boolean copyClassifier = cmd.hasOption("c");
        boolean copyLanguage = cmd.hasOption("l");
        boolean copyDataSet = cmd.hasOption("d");

        if (copyClassifier) {
            int src = getDestSlot(cmd, 0, true);
            int dest = getDestSlot(cmd, 1, false);
            languageClassifiers[dest] = languageClassifiers[src];
            out.println("Copied classifier " + src + " to " + dest);
        } else if (copyLanguage) {
            String[] args = cmd.getArgs();
            String src = args[0];
            String dest = args[1];
            if (dest.isEmpty()) {
                err.printf("Destination name cannot be empty%n");
                return;
            }
                Language language = languages.get(src);
            if (language == null) {
                Supplier<Language> languageSupplier = languagesLoader.get(src);
                if (languageSupplier != null) {
                    language = languageSupplier.get();
                }
            }
            if (language != null) {
                languages.put(dest, language);
                out.println("Copied language " + src + " to " + dest);
            } else {
                err.printf("Language with ISO 639-1 code '%s' does not exist%n", src);
            }
        } else if (copyDataSet) {
            String[] args = cmd.getArgs();
            String src = args[0];
            String dest = args[1];
            LanguageWordsDataSet dataSet = dataSets.get(src);
            if (dest.isEmpty()) {
                err.printf("Destination name cannot be empty%n");
            } else if (dataSet != null) {
                dataSets.put(dest, dataSet);
                out.println("Copied data set " + src + " to " + dest);
            } else {
                err.printf("Data set name '%s' does not exist%n", src);
            }
        }
    }

    private void remove(CommandLine cmd) {
        boolean removeClassifier = cmd.hasOption("c");
        boolean removeLanguage = cmd.hasOption("l");
        boolean removeDataSet = cmd.hasOption("d");

        if (removeClassifier) {
            int slot = getDestSlot(cmd, 0, false);
            languageClassifiers[slot] = null;
            out.println("Classifier " + slot + " removed");
        } else if (removeLanguage) {
            String[] args = cmd.getArgs();
            String language = args[0];
            languages.remove(language);
            out.println("Language " + language + " removed");
        } else if (removeDataSet) {
            String[] args = cmd.getArgs();
            String dataSet = args[0];
            dataSets.remove(dataSet);
            out.println("data set " + dataSet + " removed");
        }
    }

    private void splitData(CommandLine cmd) {
        double[] splitFractions = Arrays.stream(cmd.hasOption("d") ? cmd.getOptionValues("d") : cmd.getOptionValues("t")).mapToDouble(Double::parseDouble).toArray();
        double sum = Arrays.stream(splitFractions).sum();

        String[] args = cmd.getArgs();
        if (args.length == 0) {
            err.println("Required argument 0 missing");
            return;
        }
        String src = args[0];
        String[] destinations = cmd.hasOption("t") ? new String[]{src + "-train", src + "-cv", src + "-test"} : Arrays.copyOfRange(args, 1, args.length);
        if (!dataSets.containsKey(src)) {
            err.println("Unknown data set name: " + src);
            return;
        }
        if (splitFractions.length != destinations.length) {
            err.printf("Split fractions (%d) and destinations length (%d) must match%n", splitFractions.length, destinations.length);
            return;
        }
        LanguageWordsDataSet sourceDataSet = dataSets.get(src);
        int numWords = sourceDataSet.numWords();
        if (cmd.hasOption("t")) {
            sourceDataSet = sourceDataSet.createRandomSubset(numWords, random);
        }
        int offset = 0;
        for (int i = 0; i < splitFractions.length - 1; i++) {
            int size = (int) (splitFractions[i] / sum * numWords);
            LanguageWordsDataSet dataSet = sourceDataSet.createSubset(offset, size);
            dataSets.put(destinations[i], dataSet);
            offset += size;
        }
        if (destinations.length >= 1) {
            LanguageWordsDataSet dataSet = sourceDataSet.createSubset(offset, numWords - offset);
            dataSets.put(destinations[destinations.length - 1], dataSet);
        }
    }

    private void shuffle(CommandLine cmd) {
        String[] args = cmd.getArgs();
        if (args.length == 0) {
            err.println("Required argument 0 missing");
            return;
        }
        String src = args[0];
        LanguageWordsDataSet sourceDataSet = dataSets.get(src);
        if (sourceDataSet == null) {
            err.println("Unknown data set: " + src);
            return;
        }
        dataSets.put(src, sourceDataSet.createRandomSubset(sourceDataSet.numWords(), random));
        out.println("Shuffled");
    }

    private void data(CommandLine cmd) {
        String[] addDataValues = cmd.getOptionValues("a");
        String[] args = cmd.getArgs();
        if (args.length == 0) {
            err.println("Required argument 0 missing");
            return;
        }
        String src = args[0];
        LanguageWordsDataSet sourceDataSet = dataSets.get(src);
        if (sourceDataSet == null) {
            err.println("Unknown data set: " + src);
            return;
        }
        if (addDataValues != null) {
            Language language = getLanguageOrLoad(addDataValues[0]);
            String[] words = Arrays.copyOfRange(addDataValues, 1, addDataValues.length);
            dataSets.put(src, sourceDataSet.addAll(language, words));
            out.println("Added " + words.length + " word" + (words.length == 1 ? "" : "s"));
        }
    }

    private void learningCurve(CommandLine cmd) {
        double alpha = Double.parseDouble(cmd.getOptionValue("a"));
        double lambda = Double.parseDouble(cmd.getOptionValue("l", "0"));
        int iterations = Integer.parseInt(cmd.getOptionValue("i"));
        int batchSize = Integer.parseInt(cmd.getOptionValue("b", "1"));
        String trainingDataSetName = cmd.getOptionValue("d");
        String cvDataSetName = cmd.getOptionValue("c");
        boolean displayWithGui = cmd.hasOption("g");
        if (alpha <= 0) throw new IllegalArgumentException(String.format("learning rate (%f) must be greater than 0", alpha));
        if (lambda < 0) throw new IllegalArgumentException(String.format("regularization parameter (%f) cannot be less than 0", lambda));
        if (iterations <= 0) throw new IllegalArgumentException(String.format("number of iterations (%d) must be greater than 0", iterations));
        if (batchSize <= 0) throw new IllegalArgumentException(String.format("batch size (%d) must be greater than 0", batchSize));
        int[] testSizes = cmd.hasOption("s") ? Arrays.stream(cmd.getOptionValues("s")).mapToInt(Integer::parseInt).toArray() : null;
        int[] sizesParams = cmd.hasOption("r") ? Arrays.stream(cmd.getOptionValues("r")).mapToInt(Integer::parseInt).toArray() : null;
        if (sizesParams != null) {
            int start = sizesParams[0];
            int incr = sizesParams[2];
            testSizes = IntStream.range(0, sizesParams[1] / incr).map(i -> i * incr + start).toArray();
        }
        if (testSizes == null) {
            testSizes = new int[0];
        }

        int dest = getDestSlot(cmd, 0, true);
        LanguageClassifier languageClassifier = languageClassifiers[dest];
        int inputLettersCount = languageClassifier.getInputLettersCount();
        int maxWordLength = languageClassifier.getMaxWordLength();
        LetterEncoder letterEncoder = languageClassifier.getLetterEncoder();

        LanguageWordsDataSet defaultDataSet;
        int numWords;
        if (trainingDataSetName == null || cvDataSetName == null) {
            defaultDataSet = languageClassifier.getAllWordsDataSet().createRandomSubset(languageClassifier.getAllWordsDataSet().numWords(), random);
            numWords = defaultDataSet.numWords();
        } else {
            defaultDataSet = null;
            numWords = 0;
        }
        LanguageWordsDataSet trainingDataSet = getDataSet(trainingDataSetName, () -> defaultDataSet != null ? defaultDataSet.createSubset(0, (int) (numWords * 0.6)) : null);
        LanguageWordsDataSet cvDataSet = getDataSet(cvDataSetName, () -> defaultDataSet != null ? defaultDataSet.createSubset(trainingDataSet.numWords(), (int) (numWords * 0.2)) : null);
        IntFunction<BatchFullDataSetStream> trainingDataSetGenerator = size -> trainingDataSet.createRandomSubset(Math.min(size, trainingDataSet.numWords()), random)
                .buildDataSetGenerator(inputLettersCount, maxWordLength, letterEncoder, batchSize);
        StreamCostFunction<LanguageClassifier> cost = LanguageClassifier.getCostFunction(lambda);

        StreamSupervisedLearningAlgorithm<LanguageClassifier> learningAlgorithm = LanguageClassifier.getStreamAlgorithm(
                LanguageClassifier.getLanguageRecognizerGenerator(inputLettersCount, maxWordLength, letterEncoder, random, languageClassifier.getLanguages()),
                alpha, lambda, iterations
        );
        BatchFullDataSetStream cvDataSetStream = cvDataSet.buildDataSetGenerator(inputLettersCount, maxWordLength, letterEncoder, batchSize);

        LearningCurve learningCurve = LearningCurve.generateSetSizeLearningCurve(learningAlgorithm, cost, trainingDataSetGenerator, cvDataSetStream, testSizes);
        if (displayWithGui) {
            learningCurve.graphWithJFrame("Learning curves", "Training set size", false, 800, 800);
        } else {
            Map<Double, Double> trainingError = learningCurve.getTrainingError();
            Map<Double, Double> cvError = learningCurve.getCrossValidationError();
            for (int testSize : testSizes) {
                out.printf("Size: %10d, train error: %.10f, cv error: %.10f%n", Math.min(testSize, trainingDataSet.numWords()), trainingError.get((double) testSize), cvError.get((double) testSize));
            }
        }
    }

    private void file(CommandLine cmd) {
        String inputFile = cmd.getOptionValue("f");
        String output = cmd.getOptionValue("o");

        if (cmd.hasOption("c")) {
            if (output == null) {
                err.println("Output file must be specified");
                return;
            }
            try {
                LetterEncodingFileConverter.convert(Paths.get(inputFile), output, c -> {
                    if (c >= 'a' && c <= 'z') return String.valueOf(c);
                    out.println("Encoding for: '" + c + "'");
                    out.print(">");
                    String input = scanner.nextLine();
                    return input.isEmpty() ? null : input;
                });
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private void help(CommandLine cmd) {
        String[] args = cmd.getArgs();
        String command = args.length >= 1 ? args[0] : null;
        if (command == null) {
            for (CliCommand c : COMMANDS) {
                out.printf("%s - %s%n", c.getCommand(), c.getHelpHeader());
            }
            return;
        }
        PrintWriter printWriter = new PrintWriter(out, true);
        boolean foundCommand = false;
        for (CliCommand c : COMMANDS) {
            if (command.equals(c.getCommand())) {
                foundCommand = true;
                HelpFormatter helpFormatter = new HelpFormatter();
                helpFormatter.printHelp(printWriter, HelpFormatter.DEFAULT_WIDTH, c.getHelpCmdLineSyntax(), c.getHelpHeader(),
                        c.getOptions(), HelpFormatter.DEFAULT_LEFT_PAD, HelpFormatter.DEFAULT_DESC_PAD, "");
            }
        }
        if (!foundCommand) {
            out.println("Unknown command " + command);
        }
    }

    private Language getLanguageOrLoad(String iso6391) {
        Language language = languages.get(iso6391);
        if (language == null) {
            Supplier<Language> languageSupplier = languagesLoader.get(iso6391);
            if (languageSupplier != null) {
                language = languageSupplier.get();
                languages.put(iso6391, language);
            }
        }
        return language;
    }

    private int getDestSlot(CommandLine cmd, int index, boolean notNullSlot) {
        String[] args = cmd.getArgs();
        int slot = args.length >= 1 ? Integer.parseInt(args[index]) : 0;
        if (slot < 0 || slot >= languageClassifiers.length) {
            throw new RuntimeException("Invalid slot id " + slot + " (min: 0, max: " + (languageClassifiers.length - 1));
        }
        if (notNullSlot && languageClassifiers[slot] == null) {
            throw new RuntimeException("No classifier at slot " + slot);
        }
        return slot;
    }

    private Map<String, Supplier<Language>> loadLanguages() {
        Map<String, Supplier<Language>> languages = new HashMap<>();
        languages.put("zh", () -> new BasicLanguage("zh", "Chinese", "/chinese/zh_50k_latin_script.txt"));
        languages.put("da", () -> new BasicLanguage("da", "Danish", "/danish/da_50k_latin_script.txt"));
        languages.put("nl", () -> new BasicLanguage("nl", "Dutch", "/dutch/nl_50k.txt"));
        languages.put("en", () -> new BasicLanguage("en", "English", "/english/en_50k.txt"));
        languages.put("fr", () -> new BasicLanguage("fr", "French", "/french/fr_50k_latin_script.txt"));
        languages.put("es", () -> new BasicLanguage("es", "Spanish", "/spanish/es_50k_latin_script.txt"));
        languages.put("tr", () -> new BasicLanguage("tr", "Turkish", "/turkish/tr_50k_latin_script.txt"));
        return languages;
    }

    private Language[] parseLanguages(String[] languages) {
        if (languages == null) throw new RuntimeException("No languages specified");
        Set<Language> languagesList = new HashSet<>();
        for (String s : languages) {
            Language language = this.languages.get(s);
            if (language == null) {
                Supplier<Language> languageSupplier = this.languagesLoader.get(s);
                if (languageSupplier != null) {
                    language = languageSupplier.get();
                    this.languages.put(s, language);
                }
            }
            if (language != null) {
                languagesList.add(language);
            } else {
                err.printf("Unrecognized language: %s%n", s);
            }
        }
        if (languagesList.size() < 2) throw new RuntimeException("At least 2 languages must be specified");
        return languagesList.toArray(new Language[languagesList.size()]);
    }

    private void run() {
        String[] input;
        do {
            out.print(prompt);
            input = scanner.nextLine().toLowerCase().split(" ");
            String command = input[0];
            String[] commandInputs = Arrays.copyOfRange(input, 1, input.length);
            boolean foundCommand = false;
            for (CliCommand c : COMMANDS) {
                if (command.equals(c.getCommand())) {
                    foundCommand = true;
                    try {
                        c.accept(parser.parse(c.getOptions(), commandInputs));
                    } catch (Exception e) {
                        err.println(e.getMessage());
                    }
                }
            }
            if (!foundCommand) {
                err.println("Unknown command: " + command);
            }
        } while (!QUIT_COMMAND.equals(input[0]));
    }

    private static Option option(String opt, String longOpt, String description, int args) {
        Option option = new Option(opt, longOpt, args != 0, description);
        option.setRequired(false);
        option.setArgs(args);
        return option;
    }

    private static Option requiredOption(String opt, String longOpt, String description, int args) {
        Option option = option(opt, longOpt, description, args);
        option.setRequired(true);
        return option;
    }

    private static OptionGroup optionGroup(Option... options) {
        OptionGroup optionGroup = new OptionGroup();
        for (Option option : options) {
            optionGroup.addOption(option);
        }
        return optionGroup;
    }

    private static OptionGroup requiredOptionGroup(Option... options) {
        OptionGroup optionGroup = optionGroup(options);
        optionGroup.setRequired(true);
        return optionGroup;
    }
}
