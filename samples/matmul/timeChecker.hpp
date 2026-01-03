#ifndef TIME_CHECKER_HPP
#define TIME_CHECKER_HPP

#include <chrono>
#include <format>


class TimeChecker {
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
    bool separateOutput = false;

    static constexpr std::string_view durationToken = "<DURATION>";

    static size_t countBraces(std::string_view fmt) {
        size_t count = 0;
        for (size_t i = 0; i + 1 < fmt.size(); ++i) {
            if (fmt[i] == '{' && fmt[i + 1] == '}') {
                ++count;
                ++i;
            }
        }
        return count;
    }

    /*static int countBraces(std::string_view fmt)
    {
        std::regex brace_re(R"(\{[^}]*\})");
        return std::distance(
            std::sregex_iterator(fmt.begin(), fmt.end(), brace_re),
            std::sregex_iterator());
    }*/

    /*template <std::string_view Str, std::size_t I = 0, std::size_t Count = 0>
    consteval std::size_t count_placeholders()
    {
        if constexpr (I + 1 >= Str.size()) {
            return Count;
        }
        else if constexpr (Str[I] == '{' && Str[I + 1] == '}') {
            return count_placeholders<Str, I + 2, Count + 1>();
        }
        else {
            return count_placeholders<Str, I + 1, Count>();
        }
    }*/


public:
    template <typename... Args>
    TimeChecker(std::string_view fmt, Args&&... args)
        : start(std::chrono::high_resolution_clock::now())
    {
        std::string fmtStr(fmt);
        size_t braceCount = countBraces(fmtStr);
        size_t argCount = sizeof...(Args);

        if (braceCount == argCount) {
            separateOutput = true;
            name = std::vformat(fmtStr, std::make_format_args(std::forward<Args>(args)...));
        }
        else {
            size_t pos = fmtStr.rfind("{}");
            if (pos != std::string::npos) {
                fmtStr.replace(pos, 2, durationToken);
            }
            name = std::vformat(fmtStr, std::make_format_args(std::forward<Args>(args)...));
        }
    }

    ~TimeChecker() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (separateOutput) {
            std::printf("[%s] => %lldms\n", name.c_str(), duration);
        }
        else {
            std::string output = name;
            size_t pos = output.find(durationToken);
            if (pos != std::string::npos) {
                output.replace(pos, durationToken.size(), std::to_string(duration));
            }
            std::printf("%s\n", output.c_str());
        }
        std::fflush(stdout);
    }
};

#endif // TIME_CHECKER_HPP
