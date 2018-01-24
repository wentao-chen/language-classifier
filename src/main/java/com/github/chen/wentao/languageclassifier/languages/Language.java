package com.github.chen.wentao.languageclassifier.languages;


import java.io.Serializable;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public interface Language extends Serializable {

    String getIso6391();

    String getName();

    Set<Character> getLetters();

    int numLetters();

    List<String> getWords();

    static int countDistinctLetters(Language... languages) {
        Set<Character> letters = new HashSet<>();
        for (Language language : languages) {
            letters.addAll(language.getLetters());
        }
        return letters.size();
    }
}
