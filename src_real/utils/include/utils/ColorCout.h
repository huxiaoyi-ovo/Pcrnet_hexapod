#ifndef COLORFUL_OUTPUT_H
#define COLORFUL_OUTPUT_H

#include <iostream>

namespace Co{
    // 定义文本颜色
    const char* const COLOR_RESET   = "\033[0m";
    const char* const COLOR_BLACK   = "\033[30m";
    const char* const COLOR_RED     = "\033[31m";
    const char* const COLOR_GREEN   = "\033[32m";
    const char* const COLOR_YELLOW  = "\033[33m";
    const char* const COLOR_BLUE    = "\033[34m";
    const char* const COLOR_MAGENTA = "\033[35m";
    const char* const COLOR_CYAN    = "\033[36m";
    const char* const COLOR_WHITE   = "\033[37m";

    // 定义背景颜色
    const char* const BG_BLACK      = "\033[40m";
    const char* const BG_RED        = "\033[41m";
    const char* const BG_GREEN      = "\033[42m";
    const char* const BG_YELLOW     = "\033[43m";
    const char* const BG_BLUE       = "\033[44m";
    const char* const BG_MAGENTA    = "\033[45m";
    const char* const BG_CYAN       = "\033[46m";
    const char* const BG_WHITE      = "\033[47m";
    // 定义彩色输出宏

    // 定义彩色输出函数
}
#define COUT(txt,color,bg_color) std::cout<<color<<bg_color<<txt<<Co::COLOR_RESET<<std::endl
#define RED_WHITE(txt) std::cout<<Co::COLOR_RED<<Co::BG_WHITE<<txt<<Co::COLOR_RESET<<std::endl
#define GREEN_WHITE(txt) std::cout<<Co::COLOR_GREEN<<Co::BG_WHITE<<txt<<Co::COLOR_RESET<<std::endl
#define YELLOW_WHITE(txt) std::cout<<Co::COLOR_YELLOW<<Co::BG_WHITE<<txt<<Co::COLOR_RESET<<std::endl
#define BLUE_WHITE(txt) std::cout<<Co::COLOR_BLUE<<Co::BG_WHITE<<txt<<Co::COLOR_RESET<<std::endl
#define MAGENTA_WHITE(txt) std::cout<<Co::COLOR_MAGENTA<<Co::BG_WHITE<<txt<<Co::COLOR_RESET<<std::endl
#define CYAN_WHITE(txt) std::cout<<Co::COLOR_CYAN<<Co::BG_WHITE<<txt<<Co::COLOR_RESET<<std::endl

#define RED(txt) std::cout<<Co::COLOR_RED<<txt<<Co::COLOR_RESET<<std::endl
#define GREEN(txt) std::cout<<Co::COLOR_GREEN<<txt<<Co::COLOR_RESET<<std::endl
#define YELLOW(txt) std::cout<<Co::COLOR_YELLOW<<txt<<Co::COLOR_RESET<<std::endl
#define BLUE(txt) std::cout<<Co::COLOR_BLUE<<txt<<Co::COLOR_RESET<<std::endl
#define MAGENTA(txt) std::cout<<Co::COLOR_MAGENTA<<txt<<Co::COLOR_RESET<<std::endl
#define CYAN(txt) std::cout<<Co::COLOR_CYAN<<txt<<Co::COLOR_RESET<<std::endl


#endif // COLORFUL_OUTPUT_H

