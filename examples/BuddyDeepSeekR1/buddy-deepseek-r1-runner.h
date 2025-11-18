//===- buddy-deepseek-r1-runner.h -----------------------------------------===//
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

#ifndef BUDDY_EXAMPLES_DEEPSEEK_R1_RUNNER_H
#define BUDDY_EXAMPLES_DEEPSEEK_R1_RUNNER_H

#include <buddy/Core/Container.h>
#include <buddy/LLM/TextContainer.h>

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <string>
#include <string_view>

namespace buddy {
namespace deepseekr1 {

constexpr size_t ParamsSize = 1777088064;
constexpr size_t MaxVocabSize = 151936;
constexpr size_t MaxTokenLength = 1024;

constexpr size_t HiddenSize = 128;
constexpr size_t HeadNum = 2;
constexpr long long DefaultEosToken = 151643;

struct GenerationResult {
  size_t promptTokens = 0;
  size_t generatedTokens = 0;
  double prefillTokensPerSec = 0.0;
  double decodeTokensPerSec = 0.0;
  double totalSeconds = 0.0;
  std::string finalText;
};

using TokenCallback = std::function<void(
    size_t iterationIdx, std::string_view token, double seconds)>;
using LogCallback = std::function<void(const std::string &message)>;

void loadParameters(const std::string &paramFilePath, MemRef<float, 1> &params,
                    LogCallback logHook = {});

GenerationResult runGeneration(const std::string &inputStr,
                               const std::string &vocabPath,
                               const std::string &paramsPath, int maxNewTokens,
                               long long eosTokenId, std::ostream &tokenStream,
                               TokenCallback onToken = {},
                               LogCallback logHook = {});

} // namespace deepseekr1
} // namespace buddy

#endif // BUDDY_EXAMPLES_DEEPSEEK_R1_RUNNER_H
