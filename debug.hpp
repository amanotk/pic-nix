// -*- C++ -*-
#ifndef _DEBUG_HPP_
#define _DEBUG_HPP_

#include "tinyformat.hpp"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/ConsoleInitializer.h>
#include <plog/Log.h>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

//
// error message printing
//
#define ERROR PLOG(plog::none)
#define ERROR_IF(condition) PLOG(plog::none, (condition))

#ifndef DISABLE_DEBUG
//
// enable debug message printing
//
#define DEBUG0 PLOG(plog::info)
#define DEBUG1 PLOG(plog::debug)
#define DEBUG2 PLOG(plog::verbose)
#define DEBUG_IF0(condition) PLOG(plog::info, (condition))
#define DEBUG_IF1(condition) PLOG(plog::debug, (condition))
#define DEBUG_IF2(condition) PLOG(plog::verbose, (condition))
#else
//
// disable debug message printing
//
#define DEBUG0
#define DEBUG1
#define DEBUG2
#define DEBUG_IF0(condition)
#define DEBUG_IF1(condition)
#define DEBUG_IF2(condition)
#endif

namespace plog
{
//
// custom formatter for debug message printing
//
class DebugFormatter
{
public:
  static util::nstring header()
  {
    return util::nstring();
  }

  static util::nstring format(const Record& record)
  {
    static const char* message[] = {"ERROR!", "-----",  "-----", "------",
                                    "DEBUG0", "DEBUG1", "DEUBG2"};

    util::nostringstream ss;

    ss << std::setfill(PLOG_NSTR(' ')) << std::setw(6) << std::left << message[record.getSeverity()]
       << PLOG_NSTR("");
    ss << PLOG_NSTR("[") << record.getFunc() << PLOG_NSTR("@") << record.getLine()
       << PLOG_NSTR("] ");
    ss << record.getMessage() << PLOG_NSTR("\n");

    return ss.str();
  }
};
} // namespace plog

//
// utility for debug message printing
//
class DebugPrinter
{
public:
  static void init()
  {
    static plog::ConsoleAppender<plog::DebugFormatter> consoleAppender(plog::streamStdErr);
    plog::init(plog::verbose, &consoleAppender);
  }

  static void set_level(int level = 0)
  {
    if (level > 2) {
      // ignored
    } else if (level == 2) {
      plog::get()->setMaxSeverity(plog::verbose);
    } else if (level == 1) {
      plog::get()->setMaxSeverity(plog::debug);
    } else if (level == 0) {
      plog::get()->setMaxSeverity(plog::info);
    }
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
