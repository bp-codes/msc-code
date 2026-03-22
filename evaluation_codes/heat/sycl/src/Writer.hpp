#ifndef WRITER_HPP
#define WRITER_HPP

#include <cstddef>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <filesystem>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "Grid.hpp"

/**
 * @file Writer.hpp
 * @brief Background writer for periodically saving Grid snapshots to CSV.
 *
 * Provides an asynchronous queue-based writer:
 * - enqueue() pushes Grid snapshots into a thread-safe queue
 * - a worker thread drains the queue and writes files to disk
 * - stop() flushes the queue and joins the worker thread
 *
 * This design decouples solver compute from I/O latency.
 */
class Writer
{
private:
    /**
     * @brief Work item queued for file output.
     */
    struct Item
    {
        double t{};
        Grid grid_data{};
        std::filesystem::path file_path{};
    };

    std::mutex _mutex{};
    std::condition_variable _condition_variable{};
    std::queue<Item> _queue_items{};
    std::atomic<bool> _done{false};
    std::thread _worker_thread{};

public:
    /**
     * @brief Construct a Writer and start the worker thread.
     */
    Writer()
        : _done(false),
          _worker_thread([this]{ this->run(); })
    {
    }

    // Non-copyable
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    /**
     * @brief Enqueue a snapshot for writing (copies the Grid).
     *
     * @param prefix Output filename prefix.
     * @param outdir Output directory.
     * @param t Simulation time.
     * @param step Time step index.
     * @param grid Grid snapshot (copied internally).
     */
    void enqueue(std::string prefix,
                 std::filesystem::path outdir,
                 double t,
                 std::size_t step,
                 const Grid& grid)
    {
        enqueue(std::move(prefix), std::move(outdir), t, step, Grid(grid));
    }

    /**
     * @brief Enqueue a snapshot for writing (moves the Grid).
     *
     * @param prefix Output filename prefix.
     * @param outdir Output directory.
     * @param t Simulation time.
     * @param step Time step index.
     * @param grid Grid snapshot (moved into the queue).
     */
    void enqueue(std::string prefix,
                 std::filesystem::path outdir,
                 const double t,
                 const std::size_t step,
                 Grid&& grid)
    {
        {
            std::lock_guard<std::mutex> lk(_mutex);

            std::ostringstream oss;
            oss << prefix << "_" << std::setw(6) << std::setfill('0') << step << ".csv";
            const auto fname {oss.str()};
            auto file_path {outdir / fname};

            _queue_items.push(Item{t, std::move(grid), std::move(file_path)});
        }

        _condition_variable.notify_one();
    }

    /**
     * @brief Flush any queued work and stop the worker thread.
     *
     * Safe to call multiple times.
     */
    void stop()
    {
        {
            std::lock_guard<std::mutex> lk(_mutex);
            _done = true;
        }

        _condition_variable.notify_one();

        if (_worker_thread.joinable())
        {
            _worker_thread.join();
        }
    }

    /**
     * @brief Destructor: calls stop().
     */
    ~Writer()
    {
        stop();
    }

private:
    /**
     * @brief Worker loop: drains the queue and writes snapshots to disk.
     */
    void run()
    {
        while (true)
        {
            auto item {Item{}};

            {
                std::unique_lock<std::mutex> lock(_mutex);

                _condition_variable.wait(
                    lock,
                    [this]
                    {
                        return _done || !_queue_items.empty();
                    });

                if (_done && _queue_items.empty())
                {
                    break;
                }

                item = std::move(_queue_items.front());
                _queue_items.pop();
            }

            // Outside lock: try to write file
            try
            {
                grid_to_csv(item.t, item.grid_data, item.file_path);
            }
            catch (const std::exception& e)
            {
                std::cerr << "Writer error for '" << item.file_path
                          << "': " << e.what() << "\n";
            }
        }
    }

public:
    /**
     * @brief Convenience overload: build a CSV filename and write the snapshot.
     *
     * @param prefix Output filename prefix.
     * @param outdir Output directory.
     * @param t Simulation time.
     * @param step Time step index.
     * @param grid_data Grid snapshot to write.
     */
    static void grid_to_csv(std::string prefix,
                            std::filesystem::path outdir,
                            double t,
                            std::size_t step,
                            const Grid& grid_data)
    {
        std::ostringstream oss;
        oss << prefix << "_" << std::setw(6) << std::setfill('0') << step << ".csv";
        const auto fname {oss.str()};
        const auto file_path {outdir / fname};

        grid_to_csv(t, grid_data, file_path);
    }

    /**
     * @brief Write the Grid snapshot to a CSV file and print timing information.
     *
     * @param t Simulation time to record in the file header.
     * @param grid_data Grid snapshot to write.
     * @param file_path Output file path.
     *
     * @throws std::runtime_error If the output file cannot be opened.
     */
    static void grid_to_csv(double t,
                            const Grid& grid_data,
                            const std::filesystem::path& file_path)
    {
        // Start timer
        const auto start {std::chrono::high_resolution_clock::now()};

        std::ofstream f(file_path);
        if (!f)
        {
            throw std::runtime_error("Cannot open output file: " + file_path.string());
        }

        f.setf(std::ios::fixed);
        f << std::setprecision(8);
        f << "# t=" << t << ", nx=" << grid_data.nx << ", ny=" << grid_data.ny
          << ", length_x=" << grid_data.length_x << ", length_y=" << grid_data.length_y << "\n";

        for (auto j {std::size_t(0)}; j < grid_data.ny; j++)
        {
            for (auto i {std::size_t(0)}; i < grid_data.nx; i++)
            {
                f << grid_data.at(i, j);
                if (i + 1 < grid_data.nx)
                {
                    f << ",";
                }
            }
            f << "\n";
        }

        // Print out file name and time
        const auto end {std::chrono::high_resolution_clock::now()};
        const std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Saved " << file_path.string() << " (" << duration.count() << " ms)" << std::endl;
    }
};

#endif
