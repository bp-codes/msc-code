#ifndef WRITER_HPP
#define WRITER_HPP

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <filesystem>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <omp.h>
#include "Grid.hpp"


class Writer 
{
    
private:

    // Item to queue and save
    struct Item 
    {
        double t;
        Grid grid_data;
        std::filesystem::path file_path;
    };

    // Private class attributes
    std::mutex _mutex;
    std::condition_variable _condition_variable;
    std::queue<Item> _queue_items;
    std::atomic<bool> _done;
    std::thread _worker_thread;

public:

    // Constructor
    Writer()
        : _done(false), 
          _worker_thread([this]{ this->run(); }) {}   // Lambda function to capture this (current object) and call run()

    // Non-copyable
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    // Overload to construct new Grid
    void enqueue(std::string prefix,
             std::filesystem::path outdir,
             double t,
             std::size_t step,
             const Grid& grid)  // takes const ref
    {
        enqueue(prefix, outdir, t, step, Grid(grid));
    }

    // Create the file name and add to the queued items queue
    void enqueue( std::string prefix,
                  std::filesystem::path outdir,
                  const double t,
                  const std::size_t step,
                  Grid&& grid) 
    {
        {
            std::lock_guard<std::mutex> lk(_mutex);

            std::ostringstream oss;
            oss << prefix << "_" << std::setw(6) << std::setfill('0') << step << ".csv";
            std::string fname = oss.str();
            std::filesystem::path file_path = outdir / fname;   

            _queue_items.push(Item{t, std::move(grid), std::move(file_path)});
        }
        _condition_variable.notify_one();
    }

    // Flush queue and stop the worker.
    void stop() {
        {
            std::lock_guard<std::mutex> lk(_mutex);
            _done = true;
        }
        _condition_variable.notify_one();
        if (_worker_thread.joinable()) _worker_thread.join();
    }

    // On destruction call stop
    ~Writer() { stop(); }

private:

    // Run (
    void run() 
    {

        // Keep looping through items
        while (true) 
        {
            // Local item
            Item item {};

            // Lock section
            {
                // Get a unique lock
                std::unique_lock<std::mutex> lock(_mutex);

                // wait - block thread until done or until there's an item queued
                _condition_variable.wait( 
                        lock, [this]
                        { 
                            return _done || !_queue_items.empty(); 
                        });

                // If done and nothing left to process break out of loop
                if (_done && _queue_items.empty())
                { 
                    break;
                }

                // move front item in queue to item and pop from the queue
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

    // Overload to save grid to csv
    static void grid_to_csv(    std::string prefix,
                                std::filesystem::path outdir,
                                double t,
                                std::size_t step,
                                const Grid& grid_data)  
    {
        std::ostringstream oss;
        oss << prefix << "_" << std::setw(6) << std::setfill('0') << step << ".csv";
        std::string fname = oss.str();
        std::filesystem::path file_path = outdir / fname;   

        grid_to_csv(t, grid_data, file_path);
    }

    // save grid to csv
    static void grid_to_csv(    double t,
                                const Grid& grid_data, 
                                const std::filesystem::path& file_path
                                )
    {
        // Start timer
        auto start = std::chrono::high_resolution_clock::now();
        
        std::ofstream f(file_path);
        if (!f) 
        {            
            throw std::runtime_error("Cannot open output file: " + file_path.string());
        }

        f.setf(std::ios::fixed);
        f << std::setprecision(8);
        f << "# t=" << t << ", nx=" << grid_data.nx << ", ny=" << grid_data.ny
        << ", length_x=" << grid_data.length_x << ", length_y=" << grid_data.length_y << "\n";

        for (std::size_t j = 0; j < grid_data.ny; ++j) 
        {
            for (std::size_t i = 0; i < grid_data.nx; ++i) 
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
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Saved " << file_path.string() << " (" << duration.count() << " ms)" << std::endl;

    }
};


#endif
