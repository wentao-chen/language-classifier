package com.github.chen.wentao.languageclassifier.languages;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class BasicLanguage implements Language {

    private static final long serialVersionUID = -1766189745407596467L;
    private final String iso6391;
    private final String name;
    private final Set<Character> letters;
    private final List<String> words;

    public BasicLanguage(String iso6391, String name, String wordsFile) {
        this(iso6391, name, null, wordsFile);
    }

    public BasicLanguage(String iso6391, String name, String letters, String wordsFile) {
        this.iso6391 = iso6391;
        this.name = name;
        Set<Character> lettersSet = new HashSet<>();
        if (letters != null) {
            letters.chars().forEach(c -> lettersSet.add((char) c));
        }
        List<String> words = new ArrayList<>();
        try (InputStream inputStream = BasicLanguage.class.getResourceAsStream(wordsFile)) {
            try (InputStreamReader inputStreamReader = new InputStreamReader(inputStream)) {
                try (BufferedReader bufferedReader = new BufferedReader(inputStreamReader)) {
                    String line;
                    while ((line = bufferedReader.readLine()) != null) {
                        String word = line.split(" ")[0].toLowerCase();
                        words.add(word);
                        if (letters == null) {
                            word.chars().forEach(c -> lettersSet.add((char) c));
                        }
                    }
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        this.letters = Collections.unmodifiableSet(lettersSet);
        this.words = Collections.unmodifiableList(words);
    }

    @Override
    public String getIso6391() {
        return iso6391;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public Set<Character> getLetters() {
        return letters;
    }

    @Override
    public int numLetters() {
        return letters.size();
    }

    @Override
    public List<String> getWords() {
        return words;
    }

    @Override
    public String toString() {
        return name;
    }

}
