package com.github.chen.wentao.languageclassifier.cli;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;

import java.util.function.Consumer;

class CliCommand {

    private final String command;
    private final String helpHeader;
    private final String helpCmdLineSyntax;
    private final Consumer<CommandLine> consumer;
    private final Options options;

    CliCommand(String command, String helpHeader, String helpCmdLineSyntax, Consumer<CommandLine> consumer, Options options) {
        this.command = command;
        this.helpHeader = helpHeader;
        this.helpCmdLineSyntax = helpCmdLineSyntax;
        this.consumer = consumer;
        this.options = options;
    }

    public String getCommand() {
        return command;
    }

    public String getHelpHeader() {
        return helpHeader;
    }

    public String getHelpCmdLineSyntax() {
        return helpCmdLineSyntax;
    }

    public void accept(CommandLine cmd) {
        consumer.accept(cmd);
    }

    public Options getOptions() {
        return options;
    }
}
