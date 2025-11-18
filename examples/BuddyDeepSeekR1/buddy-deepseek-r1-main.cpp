//===- buddy-deepseek-r1-main.cpp -----------------------------------------===//
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

#include <buddy/Core/Container.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace buddy;
namespace dsr = buddy::deepseekr1;
namespace fs = std::filesystem;

namespace {

double total_time = 0.0;

void printTitle() {
  const std::string title = "DeepSeekR1 Inference Powered by Buddy Compiler";
  std::cout << "\033[33;1m" << title << "\033[0m" << std::endl;
}

void getUserInput(std::string &inputStr) {
  std::cout << "\nPlease send a message:" << std::endl;
  std::cout << ">>> ";
  if (!std::getline(std::cin, inputStr)) {
    throw std::runtime_error("Failed to read input prompt.");
  }
  std::cout << std::endl;
}

void printLogLabel() { std::cerr << "\033[34;1m[Log] \033[0m"; }

void printIterInfo(size_t iterIdx, std::string_view token, double time) {
  total_time += time;
  std::cerr << "\033[32;1m[Iteration " << iterIdx << "] \033[0m";
  std::cerr << "Token: " << token << " | "
            << "Time: " << time << "s" << std::endl;
}

} // namespace

void printSummary(const dsr::GenerationResult &result,
                  const std::string &inputStr) {
  std::cout << "\n\033[33;1m[Total time]\033[0m " << total_time << std::endl;
  std::cout << "\033[33;1m[Prefilling]\033[0m " << result.prefillTokensPerSec
            << " tokens/s" << std::endl;
  std::cout << "\033[33;1m[Decoding]\033[0m " << result.decodeTokensPerSec
            << " tokens/s" << std::endl;
  std::cout << "\033[33;1m[Input]\033[0m " << inputStr << std::endl;
  std::cout << "\033[33;1m[Output]\033[0m " << result.finalText << std::endl;
}

int runDemo() {
  printTitle();

  std::string deepSeekR1Dir = DEEPSEEKR1_EXAMPLE_PATH;
  std::string deepSeekR1BuildDir = DEEPSEEKR1_EXAMPLE_BUILD_PATH;
  const std::string vocabDir = deepSeekR1Dir + "vocab.txt";
  const std::string paramsDir = deepSeekR1BuildDir + "arg0.data";

  std::string inputStr;
  getUserInput(inputStr);

  total_time = 0.0;
  auto tokenLogger = [](size_t iterIdx, std::string_view token,
                        double seconds) {
    if (iterIdx == 0)
      return;
    printIterInfo(iterIdx - 1, token, seconds);
  };

  std::ostringstream tokenSink;
  auto runnerLogSink = [](const std::string &msg) {
    printLogLabel();
    std::cerr << msg << std::endl;
  };

  dsr::GenerationResult result = dsr::runGeneration(
      inputStr, vocabDir, paramsDir, static_cast<int>(dsr::MaxTokenLength),
      dsr::DefaultEosToken, tokenSink, tokenLogger, runnerLogSink);
  printSummary(result, inputStr);
  return 0;
}

int main() {
  try {
    return runDemo();
  } catch (const std::exception &ex) {
    std::cerr << "Generation failed: " << ex.what() << std::endl;
    return 1;
  }
}
