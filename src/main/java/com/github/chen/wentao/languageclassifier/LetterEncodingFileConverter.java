package com.github.chen.wentao.languageclassifier;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Stream;

public class LetterEncodingFileConverter {
//file -f src/main/resources/turkish/tr.txt -o test.txt -c
    public static void convert(Path inputFile, String outputFile, Function<Character, String> encoder) throws IOException {
        Set<Character> invalidCharacters = new HashSet<>();
        Map<Character, String> characterMap = new HashMap<>();
        Stream<String> stream = Files.lines(inputFile)
                .map(s -> {
                    StringBuilder str = new StringBuilder();
                    for (int i = 0; i < s.length(); i++) {
                        char c = Character.toLowerCase(s.charAt(i));
                        if (invalidCharacters.contains(c)) {
                            return null;
                        }
                        if (!Character.isWhitespace(c) && !Character.isDigit(c)) {
                            String encoding = characterMap.computeIfAbsent(c, ch -> {
                                String mapping = encoder.apply(ch);
                                if (mapping == null) {
                                    invalidCharacters.add(ch);
                                }
                                return mapping;
                            });
                            str.append(encoding);
                        } else {
                            str.append(c);
                        }
                    }
                    return str.toString();
                });
        try (PrintWriter pw = new PrintWriter(outputFile)) {
            stream.filter(Objects::nonNull).forEachOrdered(pw::println);
        }
    }

}
