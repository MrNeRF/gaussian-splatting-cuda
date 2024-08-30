//
// Created by paja on 8/29/24.
//

#ifndef GAUSSIAN_SPLATTING_CUDA_PROGESS_STATS_CUH
#define GAUSSIAN_SPLATTING_CUDA_PROGESS_STATS_CUH

#include <chrono>

class ProgressStats {
    public:
        ProgressStats() {start();};
        void start();
        void stop(int total_iter, int gaussian_cnt, float psnr);
        void print(int current_iter, int gaussian_cnt, float loss);

    private:
       std::chrono::steady_clock::time_point _start_time;
       std::chrono::steady_clock _elapsed_time;
       int _last_status_len = 0;
};

#endif
