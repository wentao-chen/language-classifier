package com.github.chen.wentao.languageclassifier;


import com.github.chen.wentao.languageclassifier.languages.Language;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.IntUnaryOperator;

@FunctionalInterface
public interface LetterEncoder extends IntUnaryOperator, Serializable {

    static LetterEncoder fromLanguages(Language... languages) {
        Set<Character> letters = new HashSet<>();
        for (Language language : languages) {
            letters.addAll(language.getLetters());
        }
        Map<Character, Integer> map = new HashMap<>();
        int i = 0;
        for (char letter : letters) {
            map.put(letter, i++);
        }
        return c -> {
            Integer encoding = map.get((char) c);
            return encoding != null ? encoding : -1;
        };
    }
}
