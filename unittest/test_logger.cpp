// -*- C++ -*-

#include "logger.hpp"

#include "catch.hpp"

using namespace nix;

class TestLogger : public Logger
{
public:
  using Logger::Logger;

  void test_config(std::string prefix, std::string path, int interval, float64 flush)
  {
    REQUIRE(config["prefix"] == prefix);
    REQUIRE(config["path"] == path);
    REQUIRE(config["interval"] == interval);
    REQUIRE(config["flush"] == flush);
  }

  void test_is_flush_required(float64 last_flushed, float64 flush, bool expected)
  {
    config["flush"]    = flush;
    this->last_flushed = last_flushed;
    REQUIRE(is_flush_required() == expected);
  }

  void test_log()
  {
    config["path"]   = ".";
    config["prefix"] = "test_logger";
    config["interval"] = 10;
    config["flush"] = 0.0;

    // step is satisfied, but flush is not satisfied
    std::filesystem::remove(get_filename());
    last_flushed = wall_clock() + 10;
    log(10);
    REQUIRE(std::filesystem::is_regular_file(get_filename()) == true);

    // step is not satisfied, but flush is satisfied
    std::filesystem::remove(get_filename());
    last_flushed = wall_clock() - 10;
    log(1);
    REQUIRE(std::filesystem::is_regular_file(get_filename()) == true);

    // both step and flush are not satisfied
    std::filesystem::remove(get_filename());
    last_flushed = wall_clock() + 10;
    log(1);
    REQUIRE(std::filesystem::is_regular_file(get_filename()) == false);

    // cleanup
    std::filesystem::remove(get_filename());
  }

  void test_append()
  {
    json object = {{"foo", "bar"}};
    json result = {};

    initialize_content();

    // initial record
    append(0, "test1", object);
    result = {{"rank", thisrank}, {"step", 0}, {"test1", object}};
    REQUIRE(content.back() == result);

    // append record
    append(1, "test1", object);
    result = {{"rank", thisrank}, {"step", 1}, {"test1", object}};
    REQUIRE(content.back() == result);

    // another record with the same step
    append(1, "test2", object);
    result = {{"rank", thisrank}, {"step", 1}, {"test1", object}, {"test2", object}};
    REQUIRE(content.back() == result);

    // append yet another record to the next step
    append(2, "test3", object);
    result = {{"rank", thisrank}, {"step", 2}, {"test3", object}};
    REQUIRE(content.back() == result);
  }

  void test_flush()
  {
    json object = {{"foo", "bar"}};
    json result = {};

    initialize_content();

    // added some record
    append(0, "test1", object);
    append(1, "test2", object);
    append(2, "test3", object);
    result = content;

    config["path"]   = ".";
    config["prefix"] = "test_logger";
    std::filesystem::remove(get_filename());
    flush();

    // check the binary file content
    {
      std::ifstream                  ifs(get_filename(), std::ios::binary);
      std::istream_iterator<uint8_t> begin(ifs);
      std::istream_iterator<uint8_t> end;
      std::vector<uint8_t>           restored_buffer(begin, end);

      int restored_index = 0;
      for (auto it = result.begin(); it != result.end(); ++it) {
        std::vector<uint8_t> buffer = json::to_msgpack(*it);
        for (int i = 0; i < buffer.size(); i++, restored_index++) {
          REQUIRE(buffer[i] == restored_buffer[restored_index]);
        }
      }
    }

    // cleanup
    std::filesystem::remove(get_filename());
  }
};

TEST_CASE("test_config")
{
  SECTION("null")
  {
    json object = {};

    TestLogger logger(object);

    logger.test_config("log", ".", 100, 10.0);
  }

  SECTION("fully specified")
  {
    json object = {{"prefix", "foo"}, {"path", "bar"}, {"interval", 1}, {"flush", 1.0}};

    TestLogger logger(object);

    logger.test_config("foo", "bar", 1, 1.0);
  }

  SECTION("partially specified")
  {
    json object = {{"interval", 10}, {"flush", 60.0}};

    TestLogger logger(object);

    logger.test_config("log", ".", 10, 60.0);
  }
}

TEST_CASE("test_is_flush_required")
{
  json object = {};

  TestLogger logger(object);

  SECTION("flush required")
  {
    logger.test_is_flush_required(wall_clock() - 10, 0.0, true);
  }

  SECTION("flush not required")
  {
    logger.test_is_flush_required(wall_clock() + 10, 100, false);
  }
}

TEST_CASE("test_log")
{
  json object = {};

  TestLogger logger(object);

  logger.test_log();
}

TEST_CASE("test_append")
{
  json object = {};

  TestLogger logger(object);

  logger.test_append();
}

TEST_CASE("test_flush")
{
  json object = {};

  TestLogger logger(object, "", 0, true);

  logger.test_flush();
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
