//
// Created by paja on 8/29/24.
//
#include "progess_stats.cuh"
#include <iomanip>
#include <sstream>
#include <iostream>

void ProgressStats::start() {
    _start_time = std::chrono::steady_clock::now();
}

void ProgressStats::stop(const int total_iter, const int gaussian_cnt, const float psnr) {
    auto cur_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_elapsed = cur_time - _start_time;

    std::cout << std::endl
              << "All done in "
              << std::fixed << std::setw(7) << std::setprecision(3) << time_elapsed.count() << "sec, avg "
              << std::fixed << std::setw(4) << std::setprecision(1) << 1.0 * total_iter / time_elapsed.count() << " iter/sec, "
              << gaussian_cnt << " splats, "
              << std::fixed << std::setw(7) << std::setprecision(6) << " psrn: " << psnr << std::endl
              << std::endl
              << std::endl;
}

void ProgressStats::print(const int current_iter, const int gaussian_cnt, const float loss) {

    auto cur_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_elapsed = cur_time - _start_time;

    std::stringstream status_line;
    // XXX Use thousand separators, but doesn't work for some reason
    status_line.imbue(std::locale(""));
    status_line
        << "\rIter: " << std::setw(6) << current_iter
        << "  Loss: " << std::fixed << std::setw(9) << std::setprecision(6) << loss;
    auto time_count = time_elapsed.count();
    status_line
        << "  Splats: " << std::setw(10) << gaussian_cnt
        << "  Time: " << std::fixed << std::setw(8) << std::setprecision(3) << time_elapsed.count() << "s"
        << "  Avg iter/s: " << std::fixed << std::setw(5) << std::setprecision(1) << 1.0 * current_iter / time_elapsed.count()
        << "  "; // Some extra whitespace, in case a "Pruning ... points" message gets printed after

    const int curlen = status_line.str().length();
    const int ws = _last_status_len - curlen;
    if (ws > 0) {
        status_line << std::string(ws, ' ');
    }

    std::cout << status_line.str() << std::flush;
    _last_status_len = curlen;
}
