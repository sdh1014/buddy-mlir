//===- buddy-deepseek-r1-cli.cpp ------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "buddy-deepseek-r1-runner.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace buddy;
namespace dsr = buddy::deepseekr1;
namespace fs = std::filesystem;

static llvm::cl::opt<std::string>
    ModelPathOpt("model",
                 llvm::cl::desc("Path to the model parameter file (arg0.data)"),
                 llvm::cl::value_desc("path"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    VocabPathOpt("vocab",
                 llvm::cl::desc("Path to the vocabulary file (vocab.txt)"),
                 llvm::cl::value_desc("path"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    PromptOpt("prompt", llvm::cl::desc("Prompt text passed directly"),
              llvm::cl::value_desc("text"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    PromptFileOpt("prompt-file", llvm::cl::desc("File containing prompt text"),
                  llvm::cl::value_desc("path"), llvm::cl::init(""));

static llvm::cl::opt<bool>
    InteractiveOpt("interactive",
                   llvm::cl::desc("Start REPL-style interactive mode (combine "
                                  "with --prompt for a system prompt)"),
                   llvm::cl::init(false));

static llvm::cl::opt<unsigned>
    MaxTokensOpt("max-tokens",
                 llvm::cl::desc("Maximum number of tokens to generate "
                                "(including the first decoded token)"),
                 llvm::cl::init(256));

static llvm::cl::opt<double> TemperatureOpt(
    "temperature",
    llvm::cl::desc(
        "Sampling temperature (currently only greedy decoding is supported)"),
    llvm::cl::init(0.0));

static llvm::cl::opt<int>
    TopKOpt("top-k",
            llvm::cl::desc(
                "Top-k sampling (currently only greedy decoding is supported)"),
            llvm::cl::init(1));

static llvm::cl::opt<double>
    TopPOpt("top-p",
            llvm::cl::desc(
                "Top-p sampling (currently only greedy decoding is supported)"),
            llvm::cl::init(1.0));

static llvm::cl::opt<long long>
    EosIdOpt("eos-id", llvm::cl::desc("ID of the end-of-sequence token"),
             llvm::cl::init(dsr::DefaultEosToken));

static llvm::cl::opt<bool> SuppressStatsOpt(
    "no-stats",
    llvm::cl::desc("Output text only and hide performance statistics"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    RunnerLogOpt("log-runner",
                 llvm::cl::desc("Print runner internal logs to stderr"),
                 llvm::cl::init(false));

namespace {

struct CLIConfig {
  std::string modelPath;
  std::string vocabPath;
  std::string prompt;
  std::string systemPrompt;
  bool interactive = false;
  unsigned maxNewTokens = 0;
  bool suppressStats = false;
  long long eosId = dsr::DefaultEosToken;
};

std::string readPromptFromFile(const std::string &filePath) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open prompt file: " + filePath);
  }
  std::ostringstream oss;
  oss << file.rdbuf();
  return oss.str();
}

std::string readPromptFromStdin() {
  std::ostringstream oss;
  oss << std::cin.rdbuf();
  return oss.str();
}

std::string getDefaultModelPath() {
#ifdef DEEPSEEKR1_EXAMPLE_BUILD_PATH
  return std::string(DEEPSEEKR1_EXAMPLE_BUILD_PATH) + "arg0.data";
#else
  return "arg0.data";
#endif
}

std::string getDefaultVocabPath() {
#ifdef DEEPSEEKR1_EXAMPLE_PATH
  return std::string(DEEPSEEKR1_EXAMPLE_PATH) + "vocab.txt";
#else
  return "vocab.txt";
#endif
}

void printStats(const dsr::GenerationResult &result) {
  llvm::errs() << "Prompt tokens: " << result.promptTokens << "\n";
  llvm::errs() << "Generated tokens: " << result.generatedTokens << "\n";
  llvm::errs() << "Prefill throughput: "
               << llvm::formatv("{0:F3}", result.prefillTokensPerSec)
               << " tokens/s\n";
  llvm::errs() << "Decode throughput: "
               << llvm::formatv("{0:F3}", result.decodeTokensPerSec)
               << " tokens/s\n";
  llvm::errs() << "Total time: " << llvm::formatv("{0:F2}", result.totalSeconds)
               << " s\n";
}

dsr::LogCallback makeRunnerLogger() {
  if (!RunnerLogOpt) {
    return {};
  }
  return [](const std::string &msg) { llvm::errs() << msg << "\n"; };
}

void runInteractiveSession(const std::string &systemPrompt,
                           const std::string &modelPath,
                           const std::string &vocabPath, int maxNewTokens,
                           long long eosTokenId, bool suppressStats,
                           const dsr::LogCallback &logHook) {
  llvm::errs()
      << "Entering interactive mode. Type :exit or :quit to end the session\n";
  std::string userInput;
  while (true) {
    std::cout << ">>> " << std::flush;
    if (!std::getline(std::cin, userInput)) {
      llvm::errs() << "Input stream ended. Leaving interactive mode\n";
      break;
    }
    if (userInput == ":exit" || userInput == ":quit") {
      llvm::errs() << "Leaving interactive mode\n";
      break;
    }
    if (userInput.empty()) {
      continue;
    }
    std::string finalPrompt = userInput;
    if (!systemPrompt.empty()) {
      finalPrompt = systemPrompt + "\n\n" + userInput;
    }
    dsr::GenerationResult result = dsr::runGeneration(
        finalPrompt, vocabPath, modelPath, maxNewTokens, eosTokenId, std::cout,
        dsr::TokenCallback{}, logHook);
    if (!suppressStats) {
      printStats(result);
    }
  }
}

bool validatePromptOptions() {
  if (!PromptOpt.empty() && !PromptFileOpt.empty()) {
    llvm::errs() << "Cannot use --prompt and --prompt-file at the same time\n";
    return false;
  }
  return true;
}

bool populatePrompt(std::string &prompt) {
  try {
    if (!PromptOpt.empty()) {
      prompt = PromptOpt;
    } else if (!PromptFileOpt.empty()) {
      prompt = readPromptFromFile(PromptFileOpt);
    } else if (!InteractiveOpt) {
      prompt = readPromptFromStdin();
    }
  } catch (const std::exception &ex) {
    llvm::errs() << ex.what() << "\n";
    return false;
  }
  if (!InteractiveOpt && prompt.empty()) {
    llvm::errs() << "Prompt cannot be empty\n";
    return false;
  }
  return true;
}

bool loadSystemPrompt(std::string &systemPrompt) {
  try {
    if (!PromptOpt.empty()) {
      systemPrompt = PromptOpt;
    } else if (!PromptFileOpt.empty()) {
      systemPrompt = readPromptFromFile(PromptFileOpt);
    } else {
      systemPrompt.clear();
    }
  } catch (const std::exception &ex) {
    llvm::errs() << ex.what() << "\n";
    return false;
  }
  return true;
}

bool validateResourcePaths(const std::string &modelPath,
                           const std::string &vocabPath) {
  if (!fs::exists(modelPath)) {
    llvm::errs() << "Model parameter file not found: " << modelPath << "\n";
    return false;
  }
  if (!fs::exists(vocabPath)) {
    llvm::errs() << "Vocabulary file not found: " << vocabPath << "\n";
    return false;
  }
  return true;
}

void warnUnsupportedSampling() {
  if (TemperatureOpt != 0.0 || TopKOpt != 1 || TopPOpt != 1.0) {
    llvm::errs() << "Only greedy decoding is implemented; "
                    "temperature/top-k/top-p arguments are ignored for now\n";
  }
}

dsr::GenerationResult runSinglePrompt(const std::string &prompt,
                                      const std::string &modelPath,
                                      const std::string &vocabPath,
                                      int maxNewTokens, long long eosTokenId,
                                      const dsr::LogCallback &logHook) {
  return dsr::runGeneration(prompt, vocabPath, modelPath, maxNewTokens,
                            eosTokenId, std::cout, dsr::TokenCallback{},
                            logHook);
}

std::optional<CLIConfig> parseConfig() {
  CLIConfig config;
  config.modelPath =
      ModelPathOpt.empty() ? getDefaultModelPath() : ModelPathOpt;
  config.vocabPath =
      VocabPathOpt.empty() ? getDefaultVocabPath() : VocabPathOpt;
  config.interactive = InteractiveOpt;
  config.maxNewTokens = std::min(MaxTokensOpt.getValue(),
                                 static_cast<unsigned>(dsr::MaxTokenLength));
  config.suppressStats = SuppressStatsOpt;
  config.eosId = EosIdOpt;

  if (!validatePromptOptions()) {
    return std::nullopt;
  }

  if (config.interactive) {
    if (!loadSystemPrompt(config.systemPrompt)) {
      return std::nullopt;
    }
  } else {
    if (!populatePrompt(config.prompt)) {
      return std::nullopt;
    }
  }

  return config;
}

bool checkPaths(const CLIConfig &config) {
  return validateResourcePaths(config.modelPath, config.vocabPath);
}

void runSinglePromptMode(const CLIConfig &config,
                         const dsr::LogCallback &logHook) {
  dsr::GenerationResult result = runSinglePrompt(
      config.prompt, config.modelPath, config.vocabPath,
      static_cast<int>(config.maxNewTokens), config.eosId, logHook);
  if (!config.suppressStats) {
    printStats(result);
  }
}

void runInteractiveMode(const CLIConfig &config,
                        const dsr::LogCallback &logHook) {
  runInteractiveSession(config.systemPrompt, config.modelPath, config.vocabPath,
                        static_cast<int>(config.maxNewTokens), config.eosId,
                        config.suppressStats, logHook);
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "buddy DeepSeek R1 CLI\n");

  std::optional<CLIConfig> config = parseConfig();
  if (!config.has_value()) {
    return 1;
  }

  if (!checkPaths(*config)) {
    return 1;
  }

  warnUnsupportedSampling();

  dsr::LogCallback runnerLogHook = makeRunnerLogger();

  try {
    if (config->interactive) {
      runInteractiveMode(*config, runnerLogHook);
    } else {
      runSinglePromptMode(*config, runnerLogHook);
    }
  } catch (const std::exception &ex) {
    llvm::errs() << "Inference failed: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
