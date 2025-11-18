//===- buddy-deepseek-r1-runner.cpp ---------------------------------------===//
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

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

using namespace std::chrono;

struct MemRefContainer {
  MemRef<float, 4> kv0;
  MemRef<float, 4> kv1;
  MemRef<float, 4> kv2;
  MemRef<float, 4> kv3;
  MemRef<float, 4> kv4;
  MemRef<float, 4> kv5;
  MemRef<float, 4> kv6;
  MemRef<float, 4> kv7;
  MemRef<float, 4> kv8;
  MemRef<float, 4> kv9;
  MemRef<float, 4> kv10;
  MemRef<float, 4> kv11;
  MemRef<float, 4> kv12;
  MemRef<float, 4> kv13;
  MemRef<float, 4> kv14;
  MemRef<float, 4> kv15;
  MemRef<float, 4> kv16;
  MemRef<float, 4> kv17;
  MemRef<float, 4> kv18;
  MemRef<float, 4> kv19;
  MemRef<float, 4> kv20;
  MemRef<float, 4> kv21;
  MemRef<float, 4> kv22;
  MemRef<float, 4> kv23;
  MemRef<float, 4> kv24;
  MemRef<float, 4> kv25;
  MemRef<float, 4> kv26;
  MemRef<float, 4> kv27;
  MemRef<float, 4> kv28;
  MemRef<float, 4> kv29;
  MemRef<float, 4> kv30;
  MemRef<float, 4> kv31;
  MemRef<float, 4> kv32;
  MemRef<float, 4> kv33;
  MemRef<float, 4> kv34;
  MemRef<float, 4> kv35;
  MemRef<float, 4> kv36;
  MemRef<float, 4> kv37;
  MemRef<float, 4> kv38;
  MemRef<float, 4> kv39;
  MemRef<float, 4> kv40;
  MemRef<float, 4> kv41;
  MemRef<float, 4> kv42;
  MemRef<float, 4> kv43;
  MemRef<float, 4> kv44;
  MemRef<float, 4> kv45;
  MemRef<float, 4> kv46;
  MemRef<float, 4> kv47;
  MemRef<float, 4> kv48;
  MemRef<float, 4> kv49;
  MemRef<float, 4> kv50;
  MemRef<float, 4> kv51;
  MemRef<float, 4> kv52;
  MemRef<float, 4> kv53;
  MemRef<float, 4> kv54;
  MemRef<float, 4> kv55;

  MemRef<float, 3> logits;

  std::array<MemRef<float, 4> *, 56> kvPtrs;

  MemRefContainer(
      MemRef<float, 4> k0, MemRef<float, 4> k1, MemRef<float, 4> k2,
      MemRef<float, 4> k3, MemRef<float, 4> k4, MemRef<float, 4> k5,
      MemRef<float, 4> k6, MemRef<float, 4> k7, MemRef<float, 4> k8,
      MemRef<float, 4> k9, MemRef<float, 4> k10, MemRef<float, 4> k11,
      MemRef<float, 4> k12, MemRef<float, 4> k13, MemRef<float, 4> k14,
      MemRef<float, 4> k15, MemRef<float, 4> k16, MemRef<float, 4> k17,
      MemRef<float, 4> k18, MemRef<float, 4> k19, MemRef<float, 4> k20,
      MemRef<float, 4> k21, MemRef<float, 4> k22, MemRef<float, 4> k23,
      MemRef<float, 4> k24, MemRef<float, 4> k25, MemRef<float, 4> k26,
      MemRef<float, 4> k27, MemRef<float, 4> k28, MemRef<float, 4> k29,
      MemRef<float, 4> k30, MemRef<float, 4> k31, MemRef<float, 4> k32,
      MemRef<float, 4> k33, MemRef<float, 4> k34, MemRef<float, 4> k35,
      MemRef<float, 4> k36, MemRef<float, 4> k37, MemRef<float, 4> k38,
      MemRef<float, 4> k39, MemRef<float, 4> k40, MemRef<float, 4> k41,
      MemRef<float, 4> k42, MemRef<float, 4> k43, MemRef<float, 4> k44,
      MemRef<float, 4> k45, MemRef<float, 4> k46, MemRef<float, 4> k47,
      MemRef<float, 4> k48, MemRef<float, 4> k49, MemRef<float, 4> k50,
      MemRef<float, 4> k51, MemRef<float, 4> k52, MemRef<float, 4> k53,
      MemRef<float, 4> k54, MemRef<float, 4> k55, MemRef<float, 3> l)
      : kv0(k0), kv1(k1), kv2(k2), kv3(k3), kv4(k4), kv5(k5), kv6(k6), kv7(k7),
        kv8(k8), kv9(k9), kv10(k10), kv11(k11), kv12(k12), kv13(k13), kv14(k14),
        kv15(k15), kv16(k16), kv17(k17), kv18(k18), kv19(k19), kv20(k20),
        kv21(k21), kv22(k22), kv23(k23), kv24(k24), kv25(k25), kv26(k26),
        kv27(k27), kv28(k28), kv29(k29), kv30(k30), kv31(k31), kv32(k32),
        kv33(k33), kv34(k34), kv35(k35), kv36(k36), kv37(k37), kv38(k38),
        kv39(k39), kv40(k40), kv41(k41), kv42(k42), kv43(k43), kv44(k44),
        kv45(k45), kv46(k46), kv47(k47), kv48(k48), kv49(k49), kv50(k50),
        kv51(k51), kv52(k52), kv53(k53), kv54(k54), kv55(k55), logits(l),
        kvPtrs{&kv0,  &kv1,  &kv2,  &kv3,  &kv4,  &kv5,  &kv6,  &kv7,
               &kv8,  &kv9,  &kv10, &kv11, &kv12, &kv13, &kv14, &kv15,
               &kv16, &kv17, &kv18, &kv19, &kv20, &kv21, &kv22, &kv23,
               &kv24, &kv25, &kv26, &kv27, &kv28, &kv29, &kv30, &kv31,
               &kv32, &kv33, &kv34, &kv35, &kv36, &kv37, &kv38, &kv39,
               &kv40, &kv41, &kv42, &kv43, &kv44, &kv45, &kv46, &kv47,
               &kv48, &kv49, &kv50, &kv51, &kv52, &kv53, &kv54, &kv55} {}
};

extern "C" void _mlir_ciface_forward_prefill(MemRefContainer *result,
                                             MemRef<float, 1> *arg0,
                                             buddy::Text<size_t, 2> *arg1);

extern "C" void _mlir_ciface_forward_decode(
    MemRefContainer *result, MemRef<float, 1> *arg0, MemRef<long long, 2> *arg1,
    MemRef<long long, 1> *arg2, MemRef<float, 4> *kv0, MemRef<float, 4> *kv1,
    MemRef<float, 4> *kv2, MemRef<float, 4> *kv3, MemRef<float, 4> *kv4,
    MemRef<float, 4> *kv5, MemRef<float, 4> *kv6, MemRef<float, 4> *kv7,
    MemRef<float, 4> *kv8, MemRef<float, 4> *kv9, MemRef<float, 4> *kv10,
    MemRef<float, 4> *kv11, MemRef<float, 4> *kv12, MemRef<float, 4> *kv13,
    MemRef<float, 4> *kv14, MemRef<float, 4> *kv15, MemRef<float, 4> *kv16,
    MemRef<float, 4> *kv17, MemRef<float, 4> *kv18, MemRef<float, 4> *kv19,
    MemRef<float, 4> *kv20, MemRef<float, 4> *kv21, MemRef<float, 4> *kv22,
    MemRef<float, 4> *kv23, MemRef<float, 4> *kv24, MemRef<float, 4> *kv25,
    MemRef<float, 4> *kv26, MemRef<float, 4> *kv27, MemRef<float, 4> *kv28,
    MemRef<float, 4> *kv29, MemRef<float, 4> *kv30, MemRef<float, 4> *kv31,
    MemRef<float, 4> *kv32, MemRef<float, 4> *kv33, MemRef<float, 4> *kv34,
    MemRef<float, 4> *kv35, MemRef<float, 4> *kv36, MemRef<float, 4> *kv37,
    MemRef<float, 4> *kv38, MemRef<float, 4> *kv39, MemRef<float, 4> *kv40,
    MemRef<float, 4> *kv41, MemRef<float, 4> *kv42, MemRef<float, 4> *kv43,
    MemRef<float, 4> *kv44, MemRef<float, 4> *kv45, MemRef<float, 4> *kv46,
    MemRef<float, 4> *kv47, MemRef<float, 4> *kv48, MemRef<float, 4> *kv49,
    MemRef<float, 4> *kv50, MemRef<float, 4> *kv51, MemRef<float, 4> *kv52,
    MemRef<float, 4> *kv53, MemRef<float, 4> *kv54, MemRef<float, 4> *kv55);

namespace buddy {
namespace deepseekr1 {

using ::MemRefContainer;

namespace {

int findMaxIndex(const float *start, const float *end) {
  return std::distance(start, std::max_element(start, end));
}

void copyKVByCachePositionBlock(const MemRefContainer &prefill,
                                MemRefContainer &decode, int cachePosition) {
  constexpr int numKV = 56;
  const int copyLen = std::min(cachePosition, static_cast<int>(MaxTokenLength));

  for (int k = 0; k < numKV; ++k) {
    auto &src = *prefill.kvPtrs[k];
    auto &dst = *decode.kvPtrs[k];

    for (int h = 0; h < static_cast<int>(HeadNum); ++h) {
      const size_t bytesToCopy =
          static_cast<size_t>(copyLen) * HiddenSize * sizeof(float);
      float *srcPtr = src.getData() + h * MaxTokenLength * HiddenSize;
      float *dstPtr = dst.getData() + h * MaxTokenLength * HiddenSize;
      std::memcpy(dstPtr, srcPtr, bytesToCopy);
    }
  }
}

void streamNewText(Text<size_t, 2> &outputContainer, std::string &lastPrinted,
                   std::ostream &tokenStream) {
  std::string current = outputContainer.revertDeepSeekR1();
  if (current.size() > lastPrinted.size()) {
    tokenStream.write(current.data() + lastPrinted.size(),
                      current.size() - lastPrinted.size());
    tokenStream.flush();
  }
  lastPrinted = std::move(current);
}

bool shouldStopAfterPrefill(int availableByContext, int maxNewTokens,
                            double prefillSeconds, std::string &streamed,
                            GenerationResult &stats,
                            std::ostream &tokenStream) {
  if (availableByContext == 0 || maxNewTokens == 0) {
    tokenStream << std::endl;
    stats.totalSeconds = prefillSeconds;
    stats.finalText = streamed;
    stats.generatedTokens = 0;
    return true;
  }
  return false;
}

bool emitTokenCallback(TokenCallback onToken, size_t iterationIdx,
                       const std::string &token, double seconds) {
  if (onToken) {
    onToken(iterationIdx, token, seconds);
  }
  return static_cast<bool>(onToken);
}

void updateDecodeStats(double decodeSeconds, size_t decodeTokens,
                       GenerationResult &stats) {
  if (decodeSeconds > 0.0 && decodeTokens > 0) {
    stats.decodeTokensPerSec =
        static_cast<double>(decodeTokens) / decodeSeconds;
  }
}

void finalizeResult(GenerationResult &stats, double totalSeconds,
                    Text<size_t, 2> &outputContainer, std::string &streamed,
                    std::ostream &tokenStream) {
  tokenStream << std::endl;
  stats.generatedTokens = outputContainer.getTokenCnt();
  stats.finalText = streamed;
  stats.totalSeconds = totalSeconds;
}

} // namespace

/// Tokenize input data in the container. Optionally emits log messages.
void tokenizeInput(const std::string &vocabFile,
                   Text<size_t, 2> &inputContainer,
                   const LogCallback &logHook) {
  if (logHook) {
    logHook("Vocab file: " + std::filesystem::canonical(vocabFile).string());
  }
  const auto buddyTokenizeStart = std::chrono::high_resolution_clock::now();
  inputContainer.tokenizeDeepSeekR1(vocabFile, MaxTokenLength);
  const auto buddyTokenizeEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> buddyTokenizeTime =
      buddyTokenizeEnd - buddyTokenizeStart;
  if (logHook) {
    std::ostringstream oss;
    oss << "Tokenize time: " << buddyTokenizeTime.count() << "ms";
    logHook(oss.str());
  }
}

/// Load parameters into data container.
void loadParameters(const std::string &paramFilePath, MemRef<float, 1> &params,
                    LogCallback logHook) {
  const auto loadStart = std::chrono::high_resolution_clock::now();
  std::ifstream paramFile(paramFilePath, std::ios::in | std::ios::binary);
  if (!paramFile.is_open()) {
    throw std::runtime_error("[Error] Failed to open params file!");
  }
  if (logHook) {
    logHook("Loading params...");
    logHook("Params file: " +
            std::filesystem::canonical(paramFilePath).string());
  }
  paramFile.read(reinterpret_cast<char *>(params.getData()),
                 sizeof(float) * (params.getSize()));
  if (paramFile.fail()) {
    throw std::runtime_error("Error occurred while reading params file!");
  }
  paramFile.close();
  const auto loadEnd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> loadTime =
      loadEnd - loadStart;
  if (logHook) {
    std::ostringstream oss;
    oss << "Params load time: " << (loadTime.count()) / 1000 << "s";
    logHook(oss.str());
  }
}

GenerationResult runGeneration(const std::string &inputStr,
                               const std::string &vocabPath,
                               const std::string &paramsPath, int maxNewTokens,
                               long long eosTokenId, std::ostream &tokenStream,
                               TokenCallback onToken, LogCallback logHook) {
  GenerationResult stats;

  Text<size_t, 2> outputContainer;
  Text<size_t, 2> inputContainerPrefill(inputStr);
  MemRef<long long, 2> inputContainerDecode({1, 1}, 0LL);
  MemRef<float, 1> paramsContainer({ParamsSize});
  MemRef<long long, 1> cachePosition({1}, 0LL);

  MemRef<float, 3> logitsPrefill({1, MaxTokenLength, MaxVocabSize});

  auto makeKV = []() {
    return MemRef<float, 4>({1, HeadNum, MaxTokenLength, HiddenSize}, 0);
  };

  MemRef<float, 4> kv0 = makeKV();
  MemRef<float, 4> kv1 = makeKV();
  MemRef<float, 4> kv2 = makeKV();
  MemRef<float, 4> kv3 = makeKV();
  MemRef<float, 4> kv4 = makeKV();
  MemRef<float, 4> kv5 = makeKV();
  MemRef<float, 4> kv6 = makeKV();
  MemRef<float, 4> kv7 = makeKV();
  MemRef<float, 4> kv8 = makeKV();
  MemRef<float, 4> kv9 = makeKV();
  MemRef<float, 4> kv10 = makeKV();
  MemRef<float, 4> kv11 = makeKV();
  MemRef<float, 4> kv12 = makeKV();
  MemRef<float, 4> kv13 = makeKV();
  MemRef<float, 4> kv14 = makeKV();
  MemRef<float, 4> kv15 = makeKV();
  MemRef<float, 4> kv16 = makeKV();
  MemRef<float, 4> kv17 = makeKV();
  MemRef<float, 4> kv18 = makeKV();
  MemRef<float, 4> kv19 = makeKV();
  MemRef<float, 4> kv20 = makeKV();
  MemRef<float, 4> kv21 = makeKV();
  MemRef<float, 4> kv22 = makeKV();
  MemRef<float, 4> kv23 = makeKV();
  MemRef<float, 4> kv24 = makeKV();
  MemRef<float, 4> kv25 = makeKV();
  MemRef<float, 4> kv26 = makeKV();
  MemRef<float, 4> kv27 = makeKV();
  MemRef<float, 4> kv28 = makeKV();
  MemRef<float, 4> kv29 = makeKV();
  MemRef<float, 4> kv30 = makeKV();
  MemRef<float, 4> kv31 = makeKV();
  MemRef<float, 4> kv32 = makeKV();
  MemRef<float, 4> kv33 = makeKV();
  MemRef<float, 4> kv34 = makeKV();
  MemRef<float, 4> kv35 = makeKV();
  MemRef<float, 4> kv36 = makeKV();
  MemRef<float, 4> kv37 = makeKV();
  MemRef<float, 4> kv38 = makeKV();
  MemRef<float, 4> kv39 = makeKV();
  MemRef<float, 4> kv40 = makeKV();
  MemRef<float, 4> kv41 = makeKV();
  MemRef<float, 4> kv42 = makeKV();
  MemRef<float, 4> kv43 = makeKV();
  MemRef<float, 4> kv44 = makeKV();
  MemRef<float, 4> kv45 = makeKV();
  MemRef<float, 4> kv46 = makeKV();
  MemRef<float, 4> kv47 = makeKV();
  MemRef<float, 4> kv48 = makeKV();
  MemRef<float, 4> kv49 = makeKV();
  MemRef<float, 4> kv50 = makeKV();
  MemRef<float, 4> kv51 = makeKV();
  MemRef<float, 4> kv52 = makeKV();
  MemRef<float, 4> kv53 = makeKV();
  MemRef<float, 4> kv54 = makeKV();
  MemRef<float, 4> kv55 = makeKV();

  MemRefContainer prefillResult(
      kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12, kv13,
      kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24, kv25,
      kv26, kv27, kv28, kv29, kv30, kv31, kv32, kv33, kv34, kv35, kv36, kv37,
      kv38, kv39, kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logitsPrefill);
  MemRefContainer *prefillPtr = &prefillResult;

  tokenizeInput(vocabPath, inputContainerPrefill, logHook);
  outputContainer.loadVocab(vocabPath);
  loadParameters(paramsPath, paramsContainer, logHook);
  if (inputContainerPrefill.getTokenCnt() == 0) {
    tokenStream << std::endl;
    stats.finalText.clear();
    return stats;
  }
  stats.promptTokens = MaxTokenLength;

  const auto prefillStart = high_resolution_clock::now();
  _mlir_ciface_forward_prefill(prefillPtr, &paramsContainer,
                               &inputContainerPrefill);
  const auto prefillEnd = high_resolution_clock::now();
  const duration<double, std::milli> prefillMs = prefillEnd - prefillStart;
  const double prefillSeconds = prefillMs.count() / 1000.0;
  if (prefillSeconds > 0.0) {
    stats.prefillTokensPerSec =
        static_cast<double>(MaxTokenLength) / prefillSeconds;
  }

  std::string streamed;
  int availableByContext =
      std::max(0, static_cast<int>(MaxTokenLength) -
                      static_cast<int>(inputContainerPrefill.getTokenCnt()));
  if (shouldStopAfterPrefill(availableByContext, maxNewTokens, prefillSeconds,
                             streamed, stats, tokenStream)) {
    return stats;
  }

  int remainingBudget = std::min(maxNewTokens, availableByContext);

  MemRef<float, 3> logitsDecode({1, 1, MaxVocabSize});
  MemRefContainer decodeResult(
      kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7, kv8, kv9, kv10, kv11, kv12, kv13,
      kv14, kv15, kv16, kv17, kv18, kv19, kv20, kv21, kv22, kv23, kv24, kv25,
      kv26, kv27, kv28, kv29, kv30, kv31, kv32, kv33, kv34, kv35, kv36, kv37,
      kv38, kv39, kv40, kv41, kv42, kv43, kv44, kv45, kv46, kv47, kv48, kv49,
      kv50, kv51, kv52, kv53, kv54, kv55, logitsDecode);
  MemRefContainer *decodePtr = &decodeResult;

  const int tokenIndex =
      static_cast<int>(inputContainerPrefill.getTokenCnt()) - 1;
  const float *startPtr =
      prefillPtr->logits.getData() + tokenIndex * MaxVocabSize;
  const float *endPtr = startPtr + MaxVocabSize;
  int maxIndex = findMaxIndex(startPtr, endPtr);

  copyKVByCachePositionBlock(prefillResult, decodeResult,
                             inputContainerPrefill.getTokenCnt());

  cachePosition.getData()[0] = inputContainerPrefill.getTokenCnt();
  inputContainerDecode.getData()[0] = static_cast<long long>(maxIndex);
  std::string tok = inputContainerPrefill.getStr(maxIndex);
  emitTokenCallback(onToken, 0, tok, prefillSeconds);
  if (maxIndex == eosTokenId) {
    tokenStream << std::endl;
    stats.totalSeconds = prefillSeconds;
    stats.finalText = streamed;
    stats.generatedTokens = outputContainer.getTokenCnt();
    return stats;
  }

  size_t nextTokenIter = 1;
  if (remainingBudget > 0) {
    outputContainer.appendTokenIdx(maxIndex);
    streamNewText(outputContainer, streamed, tokenStream);
    --remainingBudget;
  }

  if (remainingBudget == 0) {
    tokenStream << std::endl;
    stats.generatedTokens = outputContainer.getTokenCnt();
    stats.finalText = streamed;
    stats.totalSeconds = prefillSeconds;
    return stats;
  }

  const int decodeBudget = remainingBudget;
  const auto maxDecodeSteps = std::min(
      decodeBudget,
      std::max(0, static_cast<int>(MaxTokenLength) -
                      static_cast<int>(inputContainerPrefill.getTokenCnt())));

  double decodeTimeAccumMs = 0.0;
  size_t decodeTokens = 0;

  for (int i = 0; i < maxDecodeSteps; ++i) {
    const auto decodeStart = high_resolution_clock::now();
    _mlir_ciface_forward_decode(
        decodePtr, &paramsContainer, &inputContainerDecode, &cachePosition,
        &decodePtr->kv0, &decodePtr->kv1, &decodePtr->kv2, &decodePtr->kv3,
        &decodePtr->kv4, &decodePtr->kv5, &decodePtr->kv6, &decodePtr->kv7,
        &decodePtr->kv8, &decodePtr->kv9, &decodePtr->kv10, &decodePtr->kv11,
        &decodePtr->kv12, &decodePtr->kv13, &decodePtr->kv14, &decodePtr->kv15,
        &decodePtr->kv16, &decodePtr->kv17, &decodePtr->kv18, &decodePtr->kv19,
        &decodePtr->kv20, &decodePtr->kv21, &decodePtr->kv22, &decodePtr->kv23,
        &decodePtr->kv24, &decodePtr->kv25, &decodePtr->kv26, &decodePtr->kv27,
        &decodePtr->kv28, &decodePtr->kv29, &decodePtr->kv30, &decodePtr->kv31,
        &decodePtr->kv32, &decodePtr->kv33, &decodePtr->kv34, &decodePtr->kv35,
        &decodePtr->kv36, &decodePtr->kv37, &decodePtr->kv38, &decodePtr->kv39,
        &decodePtr->kv40, &decodePtr->kv41, &decodePtr->kv42, &decodePtr->kv43,
        &decodePtr->kv44, &decodePtr->kv45, &decodePtr->kv46, &decodePtr->kv47,
        &decodePtr->kv48, &decodePtr->kv49, &decodePtr->kv50, &decodePtr->kv51,
        &decodePtr->kv52, &decodePtr->kv53, &decodePtr->kv54, &decodePtr->kv55);
    const auto decodeEnd = high_resolution_clock::now();
    const duration<double, std::milli> decodeTime = decodeEnd - decodeStart;
    decodeTimeAccumMs += decodeTime.count();
    ++decodeTokens;

    const float *decodeStartPtr = decodePtr->logits.getData();
    const float *decodeEndPtr = decodeStartPtr + MaxVocabSize;
    maxIndex = findMaxIndex(decodeStartPtr, decodeEndPtr);
    tok = inputContainerPrefill.getStr(maxIndex);
    const double iterationSeconds = decodeTime.count() / 1000.0;
    emitTokenCallback(onToken, nextTokenIter++, tok, iterationSeconds);

    if (maxIndex == eosTokenId) {
      break;
    }

    inputContainerDecode.getData()[0] = maxIndex;
    outputContainer.appendTokenIdx(maxIndex);
    streamNewText(outputContainer, streamed, tokenStream);
    cachePosition.getData()[0] += 1;
  }

  const double decodeSeconds = decodeTimeAccumMs / 1000.0;
  updateDecodeStats(decodeSeconds, decodeTokens, stats);
  finalizeResult(stats, prefillSeconds + decodeSeconds, outputContainer,
                 streamed, tokenStream);
  return stats;
}

} // namespace deepseekr1
} // namespace buddy
