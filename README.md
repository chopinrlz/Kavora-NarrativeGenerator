# Instructions for creating the Kavora Narrative Generator Python Script

This `README.md` file contains instructions for the Cusor Cloud Agent building this solution.

This repository is a sandbox for creating a Python script which implements a LangChain Agent using LangChain and LangChain Anthropic on Python 3.14 that generates example narratives aligned with the narrative tags found in the `tags.csv` file.

## Instructions for Cursor

You are to create a:

* LangChain Agent Python script which will run on the command line
* Connect the LangChain Agent to the Anthropic Claude LLM for narrative generation
* Configure the Agent to generate sample, random narratives aligned with the narrative tags found in `tags.csv`
* Each narrative should be between 200 and 400 words
* Compile each narrative with its matching tag into a single JSON file

## Purpose and usage

The purpose of this Python script is to use a LangChain Agent to create sample narrative texts to be used in testing a semantic classification engine which needs pseudo-organic input to be validated properly. Each time the Python script is executed, it is expected that a new, fresh set of sample narratives are produced thereby allowing the developers to create a seemingly infinite set of sample narratives for testing.

## Constraints and implementation parameters

Here are the constraints and parameters for the implementation:

* The Python script should operate in two modes: Test mode and Real mode - in Test mode the script will skip the calls to the LLM and instead generate random narrative based on a dictionary of random words that you, the Cursor Agent, will create for this purpose. In Real mode the script will invoke the Anthropic Claude LLM to create real narratives aligned with the narrative tags in the `tags.csv` file.

* The Python script should run from the command line on any Linux, Mac, or Windows PC with the Python 3.14 runtime installed and the LangChain module and LangChain Anthropic module

* The environment variable for the Anthropic API key will be added by the developer to your Cursor Cloud Agent environment variables.