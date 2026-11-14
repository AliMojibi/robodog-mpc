// Helpers shared by the GoogleTests: modified copies of the real config.yaml in temporary files.
// CONFIG_YAML (path of config/config.yaml) is defined by CMakeLists.txt.
#pragma once

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace test_helpers {

inline std::string readFile(const std::string& path) {
  std::ifstream f(path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

inline const std::string& baseText() {
  static const std::string text = readFile(CONFIG_YAML);
  return text;
}

// The config text with the value of `key` (first line "<indent>key: ...") replaced; comments are kept.
inline std::string withValue(const std::string& key, const std::string& value, std::string text = baseText()) {
  const std::regex line("(\\n\\s+" + key + ":)[^#\\n]*");
  std::smatch m;
  if (!std::regex_search(text, m, line)) throw std::runtime_error("test bug: key not found: " + key);
  return text.replace(m.position(0), m.length(0), m.str(1) + " " + value + "   ");
}

inline std::string writeTemp(const std::string& text) {
  static int counter = 0;
  const auto path = std::filesystem::temp_directory_path() /
                    ("robodog_nmpc_test_config_" + std::to_string(::getpid()) + "_" + std::to_string(counter++) + ".yaml");
  std::ofstream(path) << text;
  return path.string();
}

// A temporary config file that is deleted again when the object goes out of scope.
struct TempConfig {
  explicit TempConfig(const std::string& text) : path(writeTemp(text)) {}
  ~TempConfig() { std::filesystem::remove(path); }
  TempConfig(const TempConfig&) = delete;
  TempConfig& operator=(const TempConfig&) = delete;
  const std::string path;
};

}  // namespace test_helpers
