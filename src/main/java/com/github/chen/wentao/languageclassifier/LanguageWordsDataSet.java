package com.github.chen.wentao.languageclassifier;

import com.github.chen.wentao.languageclassifier.languages.Language;
import com.github.chen.wentao.mllib.training.BatchFullDataSetStream;
import com.github.chen.wentao.mllib.training.DataSet;
import com.github.chen.wentao.mllib.training.DataSetTarget;
import com.github.chen.wentao.mllib.training.FullDataSet;
import com.github.chen.wentao.mllib.training.StaticFullDataSetStream;
import org.ejml.data.MatrixType;
import org.ejml.simple.SimpleMatrix;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Predicate;
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;

public class LanguageWordsDataSet implements Serializable {

    private static final long serialVersionUID = 9059505656260974884L;

    private final Language[] languages;
    private transient List<WordAndLanguage> allWords;
    private transient Map<Language, Integer> languagesReverseMap;

    public LanguageWordsDataSet(Random random, Language... languages) {
        this(languages, getAllWords(languages, random));
    }

    public LanguageWordsDataSet(List<WordAndLanguage> allWords, Language... languages) {
        this(languages, Collections.unmodifiableList(new ArrayList<>(allWords)));
    }

    private LanguageWordsDataSet(Language[] languages, List<WordAndLanguage> allWords) {
        assert languages.length > 0;
        this.languages = languages;
        this.allWords = allWords;
        this.languagesReverseMap = getLanguagesReverseMap();
    }

    public int numWords() {
        return allWords.size();
    }

    public int findMaxWordLengthForDataSetCoverage(double fractionWordsCovered) {
        int maxLength = 0;
        Map<Integer, Integer> lengths = new HashMap<>();
        for (WordAndLanguage wordAndLanguage : allWords) {
            String word = wordAndLanguage.word;
            int length = word.length();
            lengths.put(length, lengths.getOrDefault(length, 0) + 1);
            maxLength = Math.max(maxLength, length);
        }
        double cumulativeTotal = 0;
        for (int i = 0; i <= maxLength; i++) {
            cumulativeTotal += lengths.getOrDefault(i, 0);
            if (cumulativeTotal / allWords.size() >= fractionWordsCovered) {
                return i;
            }
        }
        return maxLength;
    }

    public LanguageWordsDataSet filter(Predicate<String> filter) {
        List<WordAndLanguage> allWordsCopy = allWords.stream().filter(word -> filter.test(word.getWord())).collect(Collectors.toList());
        return new LanguageWordsDataSet(languages, allWordsCopy);
    }

    public LanguageWordsDataSet createSubset(int offset, int wordCount) {
        return new LanguageWordsDataSet(languages, allWords.subList(offset, offset + wordCount));
    }

    public LanguageWordsDataSet createRandomSubset(int wordCount, Random random) {
        List<WordAndLanguage> allWordsCopy = new ArrayList<>(allWords);
        Collections.shuffle(allWordsCopy, random);
        return new LanguageWordsDataSet(languages, allWordsCopy.subList(0, wordCount));
    }

    public LanguageWordsDataSet addAll(Language language, String... words) {
        if (!hasLanguage(language)) {
            throw new IllegalArgumentException("Invalid language (" + language.getName() + ") for data set");
        }
        List<WordAndLanguage> allWordsCopy = new ArrayList<>(allWords);
        for (String word : words) {
            allWordsCopy.add(new WordAndLanguage(word, languagesReverseMap, language));
        }
        return new LanguageWordsDataSet(languages, allWordsCopy);
    }

    public boolean hasLanguage(Language language) {
        return Arrays.stream(languages).anyMatch(l -> l.equals(language));
    }

    public double accuracy(ToIntFunction<String> predictor, int testWordsCount) {
        int correct = 0;
        int count = 0;
        for (WordAndLanguage wordAndLanguage : allWords) {
            int prediction = predictor.applyAsInt(wordAndLanguage.getWord());
            if (wordAndLanguage.getLanguageIndices().get(prediction)) {
                correct += 1;
            }
            count += 1;
            if (count >= testWordsCount) {
                break;
            }
        }
        return (double) correct / count;
    }

    public BatchFullDataSetStream buildDataSetGenerator(int inputLettersCount, int maxWordLength, LetterEncoder letterEncoder, int batchSize) {
        List<WordAndLanguage> allWords = this.allWords;
        int inputs = (inputLettersCount + 1) * maxWordLength;
        int outputs = languages.length;
        int batchCount = (allWords.size() - 1) / batchSize + 1;
        return new StaticFullDataSetStream(batchCount) {
            @Override
            public FullDataSet getBatch(int batchIndex) {
                int batchStart = batchIndex % batchCount * batchSize;
                int batchEnd = batchStart + batchSize;
                List<WordAndLanguage> wordBatch = allWords.subList(batchStart, batchEnd <= allWords.size() ? batchEnd : allWords.size());
                int totalWords = wordBatch.size();
                SimpleMatrix dataSet = new SimpleMatrix(totalWords, inputs, MatrixType.DDRM);
                SimpleMatrix target = new SimpleMatrix(totalWords, outputs, MatrixType.DDRM);
                int row = 0;
                for (WordAndLanguage wordAndLanguage : wordBatch) {
                    setDataToMatrix(inputLettersCount, maxWordLength, letterEncoder, wordAndLanguage.getLanguageIndices(), wordAndLanguage.getWord(), row, dataSet, target);
                    row++;
                }
                return new FullDataSet(new DataSet(dataSet), new DataSetTarget(target, languages.length));
            }
        };
    }

    private void setDataToMatrix(int inputLettersCount, int maxWordLength, LetterEncoder letterEncoder, BitSet languageIndices, String word, int row, SimpleMatrix dataSet, SimpleMatrix target) {
        if (word.length() > maxWordLength) {
            word = word.substring(0, maxWordLength);
        }
        for (int i = 0, n = word.length(); i < maxWordLength; i++) {
            int c = letterEncoder.applyAsInt(i < n ? word.charAt(i) : ' ');
            if (c >= 0 && c < inputLettersCount + 1) {
                int col = i * (inputLettersCount + 1) + c;
                dataSet.set(row, col, 1.0);
                for (int targetCol = target.numCols() - 1; targetCol >= 0; targetCol--) {
                    if (languageIndices.get(targetCol)) {
                        target.set(row, targetCol, 1.0);
                    }
                }
            }
        }
    }

    void setDataToArray(int inputLettersCount, int maxWordLength, LetterEncoder letterEncoder, String word, double[] data) {
        if (word.length() > maxWordLength) {
            word = word.substring(0, maxWordLength);
        }
        for (int i = 0, n = word.length(); i < maxWordLength; i++) {
            int c = letterEncoder.applyAsInt(i < n ? word.charAt(i) : ' ');
            if (c >= 0 && c < inputLettersCount + 1) {
                int col = i * (inputLettersCount + 1) + c;
                data[col] = 1.0;
            }
        }
    }

    private static List<WordAndLanguage> getAllWords(Language[] languages, Random random) {
        Map<String, BitSet> allWords = new HashMap<>();
        for (int i = 0; i < languages.length; i++) {
            for (String word : languages[i].getWords()) {
                allWords.computeIfAbsent(word, k -> new BitSet()).set(i);
            }
        }
        List<WordAndLanguage> words = allWords.entrySet().stream().map(entry -> new WordAndLanguage(entry.getKey(), entry.getValue())).collect(Collectors.toList());
        Collections.shuffle(words, random);
        return words;
    }

    public Language[] getLanguages() {
        return languages;
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        this.allWords = getAllWords(languages, new Random());
        this.languagesReverseMap = getLanguagesReverseMap();
    }

    private Map<Language, Integer> getLanguagesReverseMap() {
        Map<Language, Integer> reverseMap = new HashMap<>();
        for (int i = 0; i < languages.length; i++) {
            reverseMap.put(languages[i], i);
        }
        return reverseMap;
    }

    private static class WordAndLanguage {
        private final String word;
        private final BitSet languageIndices;

        private WordAndLanguage(String word, Map<Language, Integer> reverseMap, Language... languages) {
            this(word, languagesToBitSet(reverseMap, languages));
        }

        private static BitSet languagesToBitSet(Map<Language, Integer> reverseMap, Language[] languages) {
            BitSet bitSet = new BitSet();
            for (Language language : languages) {
                Integer index = reverseMap.get(language);
                if (index != null) {
                    bitSet.set(index);
                }
            }
            return bitSet;
        }

        private WordAndLanguage(String word, BitSet languageIndex) {
            this.word = word;
            this.languageIndices = languageIndex;
        }

        public String getWord() {
            return word;
        }

        private BitSet getLanguageIndices() {
            return languageIndices;
        }
    }
}
