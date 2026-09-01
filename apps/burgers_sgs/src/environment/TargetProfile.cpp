#include "burgers/environment/TargetProfile.h"

#include "json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace burgers {
namespace environment {
namespace {

using Json = nlohmann::json;

std::string directoryName(const std::string& path) {
  const std::string::size_type separator = path.find_last_of("/\\");
  return separator == std::string::npos ? std::string(".")
                                        : path.substr(0, separator);
}

std::string joinPath(const std::string& directory,
                     const std::string& filename) {
  if(filename.empty()) {
    throw std::invalid_argument("accepted target filename must not be empty");
  }
  if(filename.front() == '/' || filename.front() == '\\') {
    return filename;
  }
  if(directory.empty() || directory == ".") {
    return std::string("./") + filename;
  }
  const char last = directory.back();
  return last == '/' || last == '\\' ? directory + filename
                                     : directory + '/' + filename;
}

std::string readFile(const std::string& path, bool binary) {
  std::ifstream input(path, binary ? std::ios::binary : std::ios::in);
  if(!input) {
    throw std::runtime_error("unable to open accepted target file: " + path);
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  if(!input.eof() && input.fail()) {
    throw std::runtime_error("unable to read accepted target file: " + path);
  }
  return contents.str();
}

std::uint32_t rotateRight(std::uint32_t value,
                          unsigned int count) noexcept {
  return (value >> count) | (value << (32u - count));
}

std::string sha256(const std::string& bytes) {
  static const std::array<std::uint32_t, 64> constants = {{
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
  }};
  std::array<std::uint32_t, 8> hash = {{
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
  }};

  std::vector<std::uint8_t> message(bytes.begin(), bytes.end());
  const std::uint64_t bit_length =
    static_cast<std::uint64_t>(message.size()) * 8u;
  message.push_back(0x80u);
  while(message.size() % 64u != 56u) {
    message.push_back(0u);
  }
  for(int shift = 56; shift >= 0; shift -= 8) {
    message.push_back(
      static_cast<std::uint8_t>((bit_length >> shift) & 0xffu));
  }

  for(std::size_t offset = 0; offset < message.size(); offset += 64u) {
    std::array<std::uint32_t, 64> words = {{0u}};
    for(std::size_t index = 0; index < 16u; ++index) {
      const std::size_t byte = offset + 4u * index;
      words[index] =
        (static_cast<std::uint32_t>(message[byte]) << 24u) |
        (static_cast<std::uint32_t>(message[byte + 1u]) << 16u) |
        (static_cast<std::uint32_t>(message[byte + 2u]) << 8u) |
        static_cast<std::uint32_t>(message[byte + 3u]);
    }
    for(std::size_t index = 16u; index < words.size(); ++index) {
      const std::uint32_t s0 =
        rotateRight(words[index - 15u], 7u) ^
        rotateRight(words[index - 15u], 18u) ^
        (words[index - 15u] >> 3u);
      const std::uint32_t s1 =
        rotateRight(words[index - 2u], 17u) ^
        rotateRight(words[index - 2u], 19u) ^
        (words[index - 2u] >> 10u);
      words[index] = words[index - 16u] + s0 + words[index - 7u] + s1;
    }

    std::uint32_t a = hash[0];
    std::uint32_t b = hash[1];
    std::uint32_t c = hash[2];
    std::uint32_t d = hash[3];
    std::uint32_t e = hash[4];
    std::uint32_t f = hash[5];
    std::uint32_t g = hash[6];
    std::uint32_t h = hash[7];
    for(std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t upper_sigma_one =
        rotateRight(e, 6u) ^ rotateRight(e, 11u) ^ rotateRight(e, 25u);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t first =
        h + upper_sigma_one + choice + constants[index] + words[index];
      const std::uint32_t upper_sigma_zero =
        rotateRight(a, 2u) ^ rotateRight(a, 13u) ^ rotateRight(a, 22u);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t second = upper_sigma_zero + majority;
      h = g;
      g = f;
      f = e;
      e = d + first;
      d = c;
      c = b;
      b = a;
      a = first + second;
    }
    hash[0] += a;
    hash[1] += b;
    hash[2] += c;
    hash[3] += d;
    hash[4] += e;
    hash[5] += f;
    hash[6] += g;
    hash[7] += h;
  }

  std::ostringstream digest;
  digest << std::hex << std::setfill('0');
  for(const std::uint32_t word : hash) {
    digest << std::setw(8) << word;
  }
  return digest.str();
}

std::vector<std::string> splitCsv(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  std::istringstream stream(line);
  while(std::getline(stream, field, ',')) {
    if(!field.empty() && field.back() == '\r') {
      field.pop_back();
    }
    fields.push_back(field);
  }
  return fields;
}

double parseFiniteDouble(const std::string& text,
                         const std::string& description) {
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &consumed);
  } catch(const std::exception&) {
    throw std::invalid_argument(description + " is not a number");
  }
  if(consumed != text.size() || std::isfinite(value) == 0) {
    throw std::invalid_argument(description + " must be finite");
  }
  return value;
}

bool nearlyEqual(double left, double right) noexcept {
  const double scale = std::max(1.0, std::max(std::abs(left), std::abs(right)));
  return std::abs(left - right) <=
         128.0 * std::numeric_limits<double>::epsilon() * scale;
}

}  // namespace

TargetProfile TargetProfile::loadAccepted(
  const std::string& metadata_path) {
  if(metadata_path.empty()) {
    throw std::invalid_argument(
      "accepted target metadata path must not be empty");
  }
  const std::string metadata_text = readFile(metadata_path, false);
  Json metadata;
  try {
    metadata = Json::parse(metadata_text);
  } catch(const Json::exception& error) {
    throw std::invalid_argument(
      std::string("unable to parse accepted target metadata: ") +
      error.what());
  }

  TargetProfile result;
  try {
    if(metadata.at("status").get<std::string>() != "accepted_dns_target") {
      throw std::invalid_argument(
        "target metadata status is not accepted_dns_target");
    }
    const std::size_t declared_cells =
      metadata.at("cell_count").get<std::size_t>();
    const Json& domain = metadata.at("domain");
    result.x_begin_ = domain.at("x_begin").get<double>();
    result.x_end_ = domain.at("x_end").get<double>();
    if(domain.at("cell_count").get<std::size_t>() != declared_cells) {
      throw std::invalid_argument(
        "target domain and metadata cell counts differ");
    }
    if(std::isfinite(result.x_begin_) == 0 ||
       std::isfinite(result.x_end_) == 0 ||
       result.x_end_ <= result.x_begin_) {
      throw std::invalid_argument("accepted target domain is invalid");
    }

    const std::string directory = directoryName(metadata_path);
    const std::string profile_path = joinPath(
      directory, metadata.at("target_csv_filename").get<std::string>());
    const std::string spectrum_path = joinPath(
      directory,
      metadata.at("target_spectrum_filename").get<std::string>());
    result.profile_hash_ =
      metadata.at("target_csv_sha256").get<std::string>();
    result.spectrum_hash_ =
      metadata.at("target_spectrum_sha256").get<std::string>();
    const std::string profile_text = readFile(profile_path, true);
    const std::string spectrum_text = readFile(spectrum_path, true);
    if(sha256(profile_text) != result.profile_hash_) {
      throw std::invalid_argument(
        "accepted DNS target profile hash does not match metadata");
    }
    if(sha256(spectrum_text) != result.spectrum_hash_) {
      throw std::invalid_argument(
        "accepted DNS target spectrum hash does not match metadata");
    }

    std::istringstream csv(profile_text);
    std::string line;
    if(!std::getline(csv, line)) {
      throw std::invalid_argument("accepted DNS target profile is empty");
    }
    const std::vector<std::string> header = splitCsv(line);
    std::size_t x_column = header.size();
    std::size_t mean_column = header.size();
    for(std::size_t column = 0; column < header.size(); ++column) {
      if(header[column] == "x") x_column = column;
      if(header[column] == "mean_velocity") mean_column = column;
    }
    if(x_column == header.size() || mean_column == header.size()) {
      throw std::invalid_argument(
        "accepted DNS target profile lacks x or mean_velocity");
    }
    std::size_t row = 0;
    while(std::getline(csv, line)) {
      if(line.empty()) continue;
      const std::vector<std::string> fields = splitCsv(line);
      if(fields.size() != header.size()) {
        throw std::invalid_argument(
          "accepted DNS target profile row has the wrong column count");
      }
      result.cell_centers_.push_back(parseFiniteDouble(
        fields[x_column], "accepted target x at row " + std::to_string(row)));
      result.mean_velocity_.push_back(parseFiniteDouble(
        fields[mean_column],
        "accepted target mean velocity at row " + std::to_string(row)));
      ++row;
    }
    if(result.mean_velocity_.size() != declared_cells) {
      throw std::invalid_argument(
        "accepted DNS target profile size does not match metadata");
    }
  } catch(const Json::exception& error) {
    throw std::invalid_argument(
      std::string("accepted target metadata is incomplete or invalid: ") +
      error.what());
  }

  result.metadata_path_ = metadata_path;
  const double width =
    (result.x_end_ - result.x_begin_) /
    static_cast<double>(result.mean_velocity_.size());
  for(std::size_t cell = 0; cell < result.cell_centers_.size(); ++cell) {
    const double expected =
      result.x_begin_ + (static_cast<double>(cell) + 0.5) * width;
    if(!nearlyEqual(result.cell_centers_[cell], expected)) {
      throw std::invalid_argument(
        "accepted DNS target coordinates are not uniform cell centers");
    }
  }
  return result;
}

const std::string& TargetProfile::metadataPath() const noexcept {
  return metadata_path_;
}

const std::string& TargetProfile::profileHash() const noexcept {
  return profile_hash_;
}

const std::string& TargetProfile::spectrumHash() const noexcept {
  return spectrum_hash_;
}

std::size_t TargetProfile::cellCount() const noexcept {
  return mean_velocity_.size();
}

double TargetProfile::xBegin() const noexcept {
  return x_begin_;
}

double TargetProfile::xEnd() const noexcept {
  return x_end_;
}

const std::vector<double>& TargetProfile::meanVelocity() const noexcept {
  return mean_velocity_;
}

std::vector<double> TargetProfile::restrictMeanVelocity(
  const Grid& grid) const {
  if(!nearlyEqual(grid.xBegin(), x_begin_) ||
     !nearlyEqual(grid.xEnd(), x_end_)) {
    throw std::invalid_argument(
      "LES and accepted DNS target domains do not match");
  }
  if(cellCount() % grid.cellCount() != 0u) {
    throw std::invalid_argument(
      "accepted DNS target cannot be conservatively restricted to the LES "
      "grid");
  }
  const std::size_t ratio = cellCount() / grid.cellCount();
  std::vector<double> restricted(grid.cellCount(), 0.0);
  for(std::size_t coarse = 0; coarse < grid.cellCount(); ++coarse) {
    double sum = 0.0;
    for(std::size_t offset = 0; offset < ratio; ++offset) {
      sum += mean_velocity_[coarse * ratio + offset];
    }
    restricted[coarse] = sum / static_cast<double>(ratio);
  }
  return restricted;
}

}  // namespace environment
}  // namespace burgers
