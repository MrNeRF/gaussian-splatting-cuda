#pragma once

#include "external/indicators.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

class TrainingProgress {
private:
    std::unique_ptr<indicators::ProgressBar> progress_bar_;
    std::chrono::steady_clock::time_point start_time_;
    int total_iterations_;
    int update_frequency_;

public:
    TrainingProgress(int total_iterations, int update_frequency = 100, bool enable_early_stopping = false)
        : total_iterations_(total_iterations),
          update_frequency_(update_frequency) {

        // Create progress bar with proper syntax for your indicators version
        progress_bar_ = std::make_unique<indicators::ProgressBar>();

        // Configure the progress bar after creation using constructor syntax
        progress_bar_->set_option(indicators::option::BarWidth(40));
        progress_bar_->set_option(indicators::option::Start("["));
        progress_bar_->set_option(indicators::option::Fill("█"));
        progress_bar_->set_option(indicators::option::Lead("▌"));
        progress_bar_->set_option(indicators::option::Remainder("░"));
        progress_bar_->set_option(indicators::option::End("]"));
        progress_bar_->set_option(indicators::option::PrefixText("Training "));
        progress_bar_->set_option(indicators::option::PostfixText("Initializing..."));
        progress_bar_->set_option(indicators::option::ShowPercentage(true));
        progress_bar_->set_option(indicators::option::ShowElapsedTime(true));
        progress_bar_->set_option(indicators::option::ShowRemainingTime(true));

        // Set color using the correct syntax for your indicators version
        progress_bar_->set_option(indicators::option::ForegroundColor(indicators::Color::cyan));

        // Set font styles
        std::vector<indicators::FontStyle> styles;
        styles.push_back(indicators::FontStyle::bold);
        progress_bar_->set_option(indicators::option::FontStyles(styles));

        start_time_ = std::chrono::steady_clock::now();
    }

    void update(int current_iteration, float loss, int splat_count, bool is_refining = false) {
        if (current_iteration % update_frequency_ != 0)
            return;

        float progress = static_cast<float>(current_iteration) / total_iterations_ * 100;
        progress_bar_->set_progress(static_cast<size_t>(progress));

        std::ostringstream postfix;
        postfix << current_iteration << "/" << total_iterations_
                << " | Loss: " << std::fixed << std::setprecision(4) << loss
                << " | Splats: " << splat_count;

        if (is_refining) {
            postfix << " (+)";
        }

        progress_bar_->set_option(indicators::option::PostfixText(postfix.str()));
    }

    void pause() {
        if (!progress_bar_->is_completed()) {
            progress_bar_->mark_as_completed();
            std::cout << std::endl;
        }
    }

    void resume(int current_iteration, float loss, int splat_count) {
        // Reset the progress bar
        progress_bar_->set_progress(static_cast<size_t>(
            static_cast<float>(current_iteration) / total_iterations_ * 100));
        update(current_iteration, loss, splat_count, false);
    }

    void complete() {
        if (!progress_bar_->is_completed()) {
            progress_bar_->set_progress(100);
            progress_bar_->mark_as_completed();
            std::cout << std::endl;
        }
    }

    void print_final_summary(int final_splats, int actual_iterations = -1) {
        complete(); // Ensure progress bar is completed first

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(end_time - start_time_).count();

        int iterations_used = (actual_iterations > 0) ? actual_iterations : total_iterations_;

        std::cout << std::endl
                  << "✓ Training completed in "
                  << std::fixed << std::setprecision(3) << elapsed << "s"
                  << " (avg " << std::fixed << std::setprecision(1)
                  << iterations_used / elapsed << " iter/s)"
                  << std::endl
                  << "✓ Final splats: " << final_splats
                  << std::endl;
    }

    // Destructor ensures completion
    ~TrainingProgress() {
        complete();
    }
};